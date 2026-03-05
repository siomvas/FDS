import logging
from typing import Dict, Any
import torch.nn as nn

log = logging.getLogger(__name__)

def create_model(model_cfg: Dict[str, Any], task_type: str, num_classes: int) -> nn.Module:
    """
    Universal Model Factory.
    Dispatches configuration to the specific library adapter.
    """
    name = model_cfg['name']
    provider = model_cfg.get('provider', 'timm')
    args = model_cfg.get('args', {})

    log.info(f"Factory dispatching: {name} | Task: {task_type} | Provider: {provider}")

    # -------------------------------------------------------------------------
    # Dispatch Logic
    # -------------------------------------------------------------------------
    try:
        if provider == "timm":
            from .adapters import timm as timm_adapter
            # timm supports classification primarily
            if task_type != "classification":
                raise ValueError(f"Provider 'timm' is only valid for classification, got {task_type}")
            return timm_adapter.create_model(name, num_classes, task_type, **args)
        
        elif provider == "smp":
            from .adapters import smp as smp_adapter
            # SMP is exclusively for segmentation
            if task_type != "segmentation":
                raise ValueError(f"Provider 'smp' is only valid for segmentation, got {task_type}")
            return smp_adapter.create_model(name, num_classes, **args)
        
        elif provider == "hf":
            from .adapters import hf as hf_adapter

            return hf_adapter.create_model(name, num_classes, task_type, **args)
            
        elif provider == "torchvision":
            from .adapters import torchvision as torchvision_adapter

            return torchvision_adapter.create_model(name, num_classes, task_type, **args)
            
        else:
            raise ValueError(f"Unknown model provider: {provider}")

    except Exception as e:
        log.error(f"Failed to create model {name} with provider {provider}")
        raise e
