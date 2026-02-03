import argparse
import base64
import json
import logging
import sys
import os
import importlib
from pathlib import Path
from typing import Dict, Any, Optional

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
# from segmentation_engine import SegmentationEngine  # Uncomment when implemented

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
    
def metric_callback(metrics: Dict[str, float], writer: SummaryWriter):
        for key, value in metrics.items():
            writer.add_scalar(key, value, global_step=int(metrics.get("epoch", 0)))

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
        "client_name": client_name,
        "seed": cfg['seed'], # access for reproducibility if needed
        "batch_size": cfg['training']['batch_size']
    }

    log.info(f"Instantiating DataModule: {dm_cfg['class_path']} with args: {dm_args}")
    data_module = get_data_module(dm_cfg['class_path'], dm_args)

    # 5. DM setup and Dataloader storing
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # 6. Model Instantiation via Factory
    model = create_model(
        model_cfg=cfg['model'],
        task_type=args.task_type,
        num_classes=cfg['training']['num_classes']
    )

    # 5. Engine Selection
    if args.task_type == "classification":
        engine = ClassificationEngine(
            model=model,
            config=cfg,
            device=device
        )
    elif args.task_type == "segmentation":
        raise NotImplementedError("Segmentation Engine not yet linked.")
        # engine = SegmentationEngine(model, cfg['training'], device)
    elif args.task_type == "detection":
        raise NotImplementedError("Detection Engine not yet linked.")
        # engine = SegmentationEngine(model, cfg['training'], device)
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

        # Evaluate before training for logging purposes
        accuracy = engine.evaluate(test_loader=val_loader, device).get("accuracy", 0.0)

        # Local Training
        engine.train_one_round(train_loader, metric_callback=metric_callback)

        # D. Evaluation (Optional but recommended for Model Selection)
        val_metrics = engine.evaluate(val_loader)
        log.info(f"Round {input_model.current_round} Validation: {val_metrics}")

        metric_callback({f"val_{k}": v for k, v in val_metrics.items()}, writer)

        # Send Updates
        output_model = flare.FLModel(
            params=engine.model.cpu().state_dict(),
            metrics=val_metrics,
            meta={"NUM_STEPS_CURRENT_ROUND": len(train_loader)} # Useful for FedAvg weighting
        )
        
        flare.send(output_model)
        log.info(f"Round {input_model.current_round} completed. Model sent.")

if __name__ == "__main__":
    main()