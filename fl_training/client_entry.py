import argparse
import base64
import binascii
import json
import logging
import re
import importlib
from typing import Dict, Any

import torch
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

from fl_training.fed_methods import MethodContext, method_hook_from_config, validate_method_task_compatibility
from fl_training.models.factory import create_model, validate_model_contract
from fl_training.engines.classification import ClassificationEngine
from fl_training.engines.segmentation import SegmentationEngine
from fl_training.engines.detection import DetectionEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("ClientEntry")

def decode_payload(payload: str) -> Dict[str, Any]:
    """
    Decodes the Base64 configuration payload passed from the Server/Job.
    """
    try:
        json_str = base64.b64decode(payload, validate=True).decode("utf-8")
        return json.loads(json_str)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
        log.error("Failed to decode config payload. Ensure base64 string is valid.")
        raise ValueError("Invalid config payload passed to client entrypoint.") from e


def validate_client_config(cfg: Dict[str, Any], task_type: str) -> None:
    required_sections = ("dataset", "training", "model", "fed")
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"Client config is missing required sections: {missing}")

    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    fed_cfg = cfg["fed"]

    if not isinstance(dataset_cfg, dict):
        raise ValueError("Client config 'dataset' section must be a mapping.")
    if not isinstance(training_cfg, dict):
        raise ValueError("Client config 'training' section must be a mapping.")
    if not isinstance(cfg["model"], dict):
        raise ValueError("Client config 'model' section must be a mapping.")
    if not isinstance(fed_cfg, dict):
        raise ValueError("Client config 'fed' section must be a mapping.")

    if not isinstance(dataset_cfg.get("class_path"), str) or not dataset_cfg["class_path"].strip():
        raise ValueError("dataset.class_path must be a non-empty string.")

    dataset_args = dataset_cfg.get("args", {})
    if not isinstance(dataset_args, dict):
        raise ValueError("dataset.args must be a mapping.")

    has_split_in_args = "split_file" in dataset_args
    has_split_in_dataset = "dataset_splits" in dataset_cfg
    if not (has_split_in_args or has_split_in_dataset):
        raise ValueError(
            "Missing dataset split path. Provide dataset.dataset_splits or dataset.args.split_file."
        )

    if "batch_size" not in training_cfg:
        raise ValueError("training.batch_size is required.")
    if "num_classes" not in training_cfg:
        raise ValueError("training.num_classes is required.")
    if "rounds" not in fed_cfg:
        raise ValueError("fed.rounds is required.")

    declared_task = str(training_cfg.get("task_type", "")).lower().strip()
    requested_task = str(task_type).lower().strip()
    if declared_task and declared_task != requested_task:
        raise ValueError(
            f"Task mismatch: CLI requested '{requested_task}' but config declares '{declared_task}'."
        )

    if "seed" not in cfg:
        raise ValueError("Config is missing required 'seed' value.")

    validate_model_contract(
        cfg["model"],
        task_type=requested_task,
        num_classes=int(training_cfg["num_classes"]),
    )
    validate_method_task_compatibility(cfg, requested_task)


def build_datamodule_args(cfg: Dict[str, Any], client_name: str) -> Dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    class_path = str(dataset_cfg.get("class_path", ""))
    task_type = str(training_cfg.get("task_type", "")).lower()

    dataset_args = dataset_cfg.get("args", {})
    if not isinstance(dataset_args, dict):
        raise ValueError("dataset.args must be a mapping.")

    dm_args = dict(dataset_args)
    dm_args["client_idx"] = extract_client_idx(client_name)
    dm_args["seed"] = cfg["seed"]
    dm_args["batch_size"] = training_cfg["batch_size"]
    dm_args["drop_last"] = bool(training_cfg.get("drop_last", False))

    if "root" in dm_args and "data_dir" not in dm_args:
        dm_args["data_dir"] = dm_args.pop("root")
    if "dataset_splits" in dataset_cfg and "split_file" not in dm_args:
        dm_args["split_file"] = dataset_cfg["dataset_splits"]
    if "label_map" in dataset_cfg and "label_map" not in dm_args:
        dm_args["label_map"] = dataset_cfg["label_map"]
    if (
        "provider" in model_cfg
        and "detection_provider" not in dm_args
        and (task_type == "detection" or class_path.endswith("PascalVOCDataModule"))
    ):
        dm_args["detection_provider"] = str(model_cfg["provider"]).lower()

    return dm_args

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
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        log.error(f"Failed to load DataModule: {class_path}")
        raise RuntimeError(f"Unable to instantiate DataModule '{class_path}'.") from e

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


def metric_callback(
    metrics: Dict[str, float],
    writer: SummaryWriter,
    *,
    prefix: str = "",
    step_override: int | None = None,
):
    step_value = step_override if step_override is not None else metrics.get("round", metrics.get("step", 0))
    try:
        step = int(step_value)
    except (TypeError, ValueError):
        step = 0

    for key, value in metrics.items():
        if key in {"step", "round", "num_steps_current_round"}:
            continue
        if isinstance(value, (int, float)):
            metric_name = key
            if prefix == "train":
                if metric_name.startswith("train_"):
                    metric_name = metric_name[len("train_") :]
            elif prefix in {"pre", "val"}:
                if metric_name.startswith("val_"):
                    metric_name = metric_name[len("val_") :]

            tag = f"{prefix}/{metric_name}" if prefix else metric_name
            writer.add_scalar(tag, value, global_step=step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_payload", type=str, required=True, help="Base64 encoded full configuration")
    parser.add_argument("--task_type", type=str, required=True, help="classification, segmentation, or detection")
    # Note: We can ignore other args like --lr or --epochs if they were passed mostly for display 
    # or fallback, because the truth is in the payload.
    args, _ = parser.parse_known_args()


    # 1. Configuration
    flare_ctx = flare.init()
    client_name = flare.get_site_name(flare_ctx)
    cfg = decode_payload(args.config_payload)
    validate_client_config(cfg, args.task_type)
    writer = SummaryWriter()
    
    log.info(f"Client {client_name} Started with Task: {args.task_type}")
    log.info(f"Loaded Configuration:\n{json.dumps(cfg, indent=2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}") # comment after testing

    # 4. Resources: Data & Model
    dm_cfg = cfg['dataset']
    dm_args = build_datamodule_args(cfg, client_name)

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

    method_hook = method_hook_from_config(cfg, task_type)

    pre_round_eval = bool(cfg["training"].get("pre_round_eval", False))

    # 6. Federated Learning Loop
    while flare.is_running():
        # Receive Global Weights
        input_model = flare.receive()
        log.info(f"Client {client_name} received model for round {input_model.current_round}")

        engine.model.load_state_dict(input_model.params)
        engine.current_round = input_model.current_round
        engine.model.to(device)
        round_idx = int(input_model.current_round)
        input_meta = dict(getattr(input_model, "meta", {}) or {})
        global_params = {name: tensor.detach().clone().cpu() for name, tensor in input_model.params.items()}
        method_ctx = MethodContext(
            method_name=str(cfg.get("fed", {}).get("method", "fedavg")).lower(),
            round_idx=round_idx,
            global_params=global_params,
            input_meta=input_meta,
        )
        method_hook.on_round_start(engine.model, method_ctx)

        if pre_round_eval and val_loader is not None:
            pre_metrics = engine.evaluate(val_loader=val_loader)
            metric_callback(pre_metrics, writer, prefix="pre", step_override=round_idx)

        # Local Training
        train_metrics = engine.train_one_round(
            train_loader,
            metric_callback=lambda m: metric_callback(m, writer, prefix="train"),
            method_hook=method_hook,
            method_ctx=method_ctx,
        )

        # D. Evaluation (Optional but recommended for Model Selection)
        val_metrics = {}
        if val_loader is not None:
            val_metrics = engine.evaluate(val_loader)
            log.info(f"Round {input_model.current_round} Validation: {val_metrics}")
            metric_callback(val_metrics, writer, prefix="val", step_override=round_idx)

        # Send Updates
        executed_steps = int(train_metrics.get("num_steps_current_round", 0))
        method_hook.on_round_end(engine.model, method_ctx)
        train_samples = len(getattr(train_loader, "dataset", []))
        round_meta = method_hook.export_round_meta(
            model=engine.model,
            ctx=method_ctx,
            executed_steps=executed_steps,
            train_samples=int(train_samples),
        )
        output_model = flare.FLModel(
            params=engine.model.cpu().state_dict(),
            metrics={**train_metrics, **val_metrics},
            meta={"NUM_STEPS_CURRENT_ROUND": executed_steps, **round_meta}
        )
        
        flare.send(output_model)
        log.info(f"Round {input_model.current_round} completed. Model sent.")

if __name__ == "__main__":
    main()
