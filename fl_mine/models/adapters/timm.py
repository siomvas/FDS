import logging
import timm
import torch.nn as nn
from utils.proxy_config import hpc_proxy

log = logging.getLogger(__name__)

def create_model(name: str, num_classes: int, task_type: str, **kwargs) -> nn.Module:

    log.info(f"Initializing Timm Model: {name} | Classes: {num_classes}")

    # Timm handles head replacement automatically via num_classes.
    # It also standardizes 'pretrained=True' vs 'pretrained=False'.
    
    # Extract 'pretrained' from kwargs if present, for now force user to specify it
    try:
        pretrained = kwargs["pretrained"]
    except KeyError:
        pretrained = False
        print("Pretrained not specified for Timm model; defaulting to False.")
    
    with hpc_proxy():
        try:
            model = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=num_classes,
                **kwargs
            )
            return model
        except Exception as e:
            log.error(f"Failed to create timm model '{name}'. Available models: timm.list_models()")
            raise e