import logging
from typing import Dict, Any
import torch.nn as nn

log = logging.getLogger(__name__)

def validate_model_contract(model_cfg: Dict[str, Any], task_type: str, num_classes: int) -> Dict[str, Any]:
    """Validate and normalize model factory inputs without creating a model."""
    if not isinstance(model_cfg, dict):
        raise TypeError(f"model_cfg must be a mapping, got {type(model_cfg).__name__}.")
    if "name" not in model_cfg:
        raise KeyError("model_cfg is missing required key 'name'.")

    task_type = str(task_type).lower()
    valid_tasks = {"classification", "segmentation", "detection"}
    if task_type not in valid_tasks:
        raise ValueError(f"Unknown task type '{task_type}'. Valid options: {sorted(valid_tasks)}")

    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}.")

    name = str(model_cfg["name"])
    provider = str(model_cfg.get("provider", "timm")).lower()
    args = model_cfg.get("args", {})
    if not isinstance(args, dict):
        raise TypeError(f"model_cfg.args must be a mapping, got {type(args).__name__}.")

    provider_task_support = {
        "timm": {"classification"},
        "smp": {"segmentation"},
        "hf": {"classification", "segmentation", "detection"},
        "torchvision": {"classification", "segmentation", "detection"},
    }
    if provider not in provider_task_support:
        raise ValueError(f"Unknown model provider '{provider}'. Supported: {sorted(provider_task_support)}")
    if task_type not in provider_task_support[provider]:
        supported = sorted(provider_task_support[provider])
        raise ValueError(
            f"Provider '{provider}' does not support task '{task_type}'. Supported tasks: {supported}."
        )

    return {
        "name": name,
        "provider": provider,
        "args": args,
        "task_type": task_type,
        "num_classes": num_classes,
    }


def create_model(model_cfg: Dict[str, Any], task_type: str, num_classes: int) -> nn.Module:
    """
    Universal Model Factory.
    Dispatches configuration to the specific library adapter.
    """
    validated = validate_model_contract(model_cfg, task_type, num_classes)
    name = validated["name"]
    provider = validated["provider"]
    args = validated["args"]
    task_type = validated["task_type"]

    log.info(f"Factory dispatching: {name} | Task: {task_type} | Provider: {provider}")

    # -------------------------------------------------------------------------
    # Dispatch Logic
    # -------------------------------------------------------------------------
    try:
        if provider == "timm":
            from .adapters import timm as timm_adapter
            return timm_adapter.create_model(name, num_classes, task_type, **args)
        
        elif provider == "smp":
            from .adapters import smp as smp_adapter
            return smp_adapter.create_model(name, num_classes, **args)
        
        elif provider == "hf":
            from .adapters import hf as hf_adapter

            return hf_adapter.create_model(name, num_classes, task_type, **args)
            
        elif provider == "torchvision":
            from .adapters import torchvision as torchvision_adapter

            return torchvision_adapter.create_model(name, num_classes, task_type, **args)
            
    except (ImportError, TypeError, RuntimeError, ValueError, OSError) as e:
        log.error(f"Failed to create model {name} with provider {provider}")
        raise RuntimeError(
            f"Model factory failed for provider '{provider}', model '{name}', task '{task_type}'."
        ) from e
