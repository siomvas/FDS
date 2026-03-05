import argparse
import base64
import json
import logging
import re
import sys
import importlib
from pathlib import Path
from typing import Dict, Any

import torch
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

# -----------------------------------------------------------------------------
# 1. Path Setup for Imports
# -----------------------------------------------------------------------------
# We need to ensure we can import:
# 1. Local modules (models.factory, engines)
# 2. External modules (data_loading, which is one level up)

# Add current directory (fl_training) to sys.path
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Add project root (parent of fl_training) to sys.path to allow 'data_loading' import
project_root = current_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Now we can safely import local modules
from models.factory import create_model
from engines.classification import ClassificationEngine
from engines.segmentation import SegmentationEngine
from engines.detection import DetectionEngine

# Configure Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ClientEntry")

def decode_payload(payload: str) -> Dict[str, Any]:
    """
    Decodes the Base64 configuration payload passed from the Server/Job.
    """
    try:
        json_str = base64.b64decode(payload).decode("utf-8")
        return json.loads(json_str)
    except Exception as e:
        log.error("Failed to decode config payload. Ensure base64 string is valid.")
        raise e

def get_data_module(class_path: str, args: Dict[str, Any]):
    """
    Dynamically imports and instantiates the DataModule.
    Expects class_path like 'data_loading.data_modules.TinyImageNetDataModule'
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**args)
    except (ImportError, AttributeError) as e:
        log.error(f"Failed to load DataModule: {class_path}")
        raise e

def extract_client_idx(client_name: str) -> int:
    """Convert NVFlare client name to zero-based numeric index."""
    if client_name.startswith("site-"):
        return int(client_name.split("-")[-1]) - 1
    if client_name.startswith("client_"):
        return int(client_name.split("_")[-1])
    match = re.search(r"(\d+)$", client_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract client index from '{client_name}'")


def metric_callback(metrics: Dict[str, float], writer: SummaryWriter):
    step_value = metrics.get("step", metrics.get("round", 0))
    try:
        step = int(step_value)
    except (TypeError, ValueError):
        step = 0

    for key, value in metrics.items():
        if key in {"step", "round"}:
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, global_step=step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_payload", type=str, required=True, help="Base64 encoded full configuration")
    parser.add_argument("--task_type", type=str, required=True, help="classification, segmentation, or detection")
    # Note: We can ignore other args like --lr or --epochs if they were passed mostly for display 
    # or fallback, because the truth is in the payload.
    args, unknown = parser.parse_known_args()


    # 1. Configuration
    flare_ctx = flare.init()
    client_name = flare.get_site_name(flare_ctx)
    cfg = decode_payload(args.config_payload)
    writer = SummaryWriter()
    
    log.info(f"Client {client_name} Started with Task: {args.task_type}")
    log.info(f"Loaded Configuration:\n{json.dumps(cfg, indent=2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}") # comment after testing

    # 4. Resources: Data & Model
    dm_cfg = cfg['dataset']
    dm_args = {
        **dm_cfg['args'],
        "client_idx": extract_client_idx(client_name),
        "seed": cfg['seed'],
        "batch_size": cfg['training']['batch_size']
    }
    if "root" in dm_args and "data_dir" not in dm_args:
        dm_args["data_dir"] = dm_args.pop("root")
    if "dataset_splits" in dm_cfg and "split_file" not in dm_args:
        dm_args["split_file"] = dm_cfg["dataset_splits"]
    if "label_map" in dm_cfg and "label_map" not in dm_args:
        dm_args["label_map"] = dm_cfg["label_map"]

    log.info(f"Instantiating DataModule: {dm_cfg['class_path']} with args: {dm_args}")
    data_module = get_data_module(dm_cfg['class_path'], dm_args)

    # 5. DM setup and Dataloader storing
    data_module.setup()

    train_loader = data_module.train_dataloader
    val_loader = data_module.val_dataloader

    # 6. Model Instantiation via Factory
    model = create_model(
        model_cfg=cfg['model'],
        task_type=args.task_type,
        num_classes=cfg['training']['num_classes']
    )

    # 5. Engine Selection
    task_type = args.task_type.lower()
    if task_type == "classification":
        engine = ClassificationEngine(
            model=model,
            config=cfg,
            device=device
        )
    elif task_type == "segmentation":
        engine = SegmentationEngine(
            model=model,
            config=cfg,
            device=device
        )
    elif task_type == "detection":
        engine = DetectionEngine(
            model=model,
            config=cfg,
            device=device
        )
    else:
        raise ValueError(f"Unknown task type: {args.task_type}")

    # 6. Federated Learning Loop
    while flare.is_running():
        # Receive Global Weights
        input_model = flare.receive()
        log.info(f"Client {client_name} received model for round {input_model.current_round}")

        engine.model.load_state_dict(input_model.params)
        engine.current_round = input_model.current_round
        engine.model.to(device)

        if val_loader is not None:
            pre_metrics = engine.evaluate(val_loader=val_loader)
            metric_callback({f"pre_{k}": v for k, v in pre_metrics.items()}, writer)

        # Local Training
        train_metrics = engine.train_one_round(
            train_loader,
            metric_callback=lambda m: metric_callback(m, writer)
        )

        # D. Evaluation (Optional but recommended for Model Selection)
        val_metrics = {}
        if val_loader is not None:
            val_metrics = engine.evaluate(val_loader)
            log.info(f"Round {input_model.current_round} Validation: {val_metrics}")
            metric_callback({f"val_{k}": v for k, v in val_metrics.items()}, writer)

        # Send Updates
        executed_steps = int(train_metrics.get("num_steps_current_round", 0))
        output_model = flare.FLModel(
            params=engine.model.cpu().state_dict(),
            metrics={**train_metrics, **val_metrics},
            meta={"NUM_STEPS_CURRENT_ROUND": executed_steps}
        )
        
        flare.send(output_model)
        log.info(f"Round {input_model.current_round} completed. Model sent.")

if __name__ == "__main__":
    main()
