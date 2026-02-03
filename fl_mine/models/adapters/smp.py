import logging
import torch.nn as nn
import segmentation_models_pytorch as smp

from utils.proxy_config import hpc_proxy

log = logging.getLogger(__name__)

def create_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Adapter for Segmentation Models Pytorch (SMP).
    
    Args:
        name (str): The architecture name, e.g., "Unet", "FPN", "DeepLabV3".
                    Case-insensitive in this adapter logic for robustness.
        num_classes (int): The number of output classes (mask channels).
        **kwargs: Additional args passed from YAML (e.g., encoder_name, encoder_weights).
    
    Returns:
        nn.Module: The instantiated segmentation model.
    """
    
    # 1. Normalize Architecture Name
    # SMP uses CamelCase class names (e.g., smp.Unet), but users might write "unet" in YAML.
    # We define a mapping or use case-insensitive getattr lookup.
    arch_name_map = {
        "unet": "Unet",
        "unetplusplus": "UnetPlusPlus",
        "manet": "MAnet",
        "linknet": "Linknet",
        "fpn": "FPN",
        "pspnet": "PSPNet",
        "deeplabv3": "DeepLabV3",
        "deeplabv3plus": "DeepLabV3Plus",
        "pan": "PAN"
    }
    
    clean_name = arch_name_map.get(name.lower(), name)
    
    if not hasattr(smp, clean_name):
        raise ValueError(
            f"Architecture '{name}' (resolved as '{clean_name}') not found in SMP. "
            f"Available options: {list(arch_name_map.keys())}"
        )

    # 2. Extract specific args with defaults appropriate for FL
    # In FL, we often want to start with pretrained encoders but random decoders.
    # encoder_name = kwargs.pop("encoder_name", "resnet34")
    # encoder_weights = kwargs.pop("encoder_weights", "imagenet")
    # in_channels = kwargs.pop("in_channels", 3)
    
    # To start, better to force users to specify these explicitly in YAML
    try:
        encoder_name = kwargs["encoder_name"]
        encoder_weights = kwargs["encoder_weights"]
        in_channels = kwargs.get("in_channels", 3)
    except KeyError as e:
        raise ValueError(
            f"Missing required argument '{e.args[0]}' for SMP model '{name}'. "
            "Please specify 'encoder_name' and 'encoder_weights' in the model config."
        )

    log.info(f"Initializing SMP {clean_name} | Encoder: {encoder_name} | Classes: {num_classes}")

    # 3. Instantiate
    # Note: We translate our standard 'num_classes' to SMP's 'classes'
    model_fn = getattr(smp, clean_name)
    
    with hpc_proxy():
        model = model_fn(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs  # Pass any remaining args (e.g. activation, encoder_depth)
        )

    return model