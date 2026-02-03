import logging
from typing import Dict, Any
import torch.nn as nn
from .adapters import timm
from .adapters import smp
from .adapters import hf
from .adapters import torchvision

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
            # timm supports classification primarily
            if task_type != "classification":
                raise ValueError(f"Provider 'timm' is only valid for classification, got {task_type}")
            return timm.create_model(name, num_classes, task_type, **args)
        
        elif provider == "smp":
            # SMP is exclusively for segmentation
            if task_type != "segmentation":
                raise ValueError(f"Provider 'smp' is only valid for segmentation, got {task_type}")
            return smp.create_model(name, num_classes, **args)
        
        elif provider == "hf":
            return hf.create_model(name, num_classes, task_type, **args)
            
        elif provider == "torchvision":
            # If you implement the dedicated adapter
            # return torchvision_adapter.create_model(name, num_classes, task_type, **args)
            raise NotImplementedError("Torchvision adapter not linked. Use 'timm' provider instead.")
            
        else:
            raise ValueError(f"Unknown model provider: {provider}")

    except Exception as e:
        log.error(f"Failed to create model {name} with provider {provider}")
        raise e