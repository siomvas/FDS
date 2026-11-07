import logging
import timm
import torch.nn as nn
from utils.proxy_config import hpc_proxy

log = logging.getLogger(__name__)

def create_model(name: str, num_classes: int, task_type: str, **kwargs) -> nn.Module:
    log.info(f"Initializing Timm Model: {name} | Classes: {num_classes}")

    pretrained = bool(kwargs.pop("pretrained", False))
    proxy = kwargs.pop("proxy", None)

    with hpc_proxy(proxy):
        try:
            return timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=num_classes,
                **kwargs,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            log.error(f"Failed to create timm model '{name}'. Available models: timm.list_models()")
            raise RuntimeError(f"Unable to create timm model '{name}'.") from e
