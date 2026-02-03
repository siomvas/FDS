import logging
import torch.nn as nn
from torchvision import models

from utils.proxy_config import hpc_proxy

log = logging.getLogger(__name__)

def _replace_head(model: nn.Module, num_classes: int, model_name: str):
    """
    Heuristics to replace the final classification layer for common Torchvision families.
    """
    
    # Case 1: ResNet, RegNet, GoogLeNet, ShuffleNet, Inception3
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    # Case 2: VGG, AlexNet, MobileNetV2/V3, ConvNeXt, EfficientNet, DenseNet
    # These usually have a 'classifier' which is a Sequential. 
    # The linear layer is typically the last element.
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            # DenseNet (sometimes simple linear)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return
        
        elif isinstance(model.classifier, nn.Sequential):
            # Inspect the last layer
            last_layer_idx = len(model.classifier) - 1
            last_layer = model.classifier[last_layer_idx]
            
            if isinstance(last_layer, nn.Linear):
                in_features = last_layer.in_features
                model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)
                return
            
            # Special Case: VGG/AlexNet often have Dropout -> Linear
            # We assume the user knows what they are doing if they pick VGG.
            # This heuristic covers MobileNetV2/EfficientNet well.

    # Case 3: Vision Transformer (ViT), Swin
    if hasattr(model, 'heads') and isinstance(model.heads, nn.Sequential):
        # ViT implementation in torchvision
        first_head = model.heads[0] # Usually the Linear layer
        if isinstance(first_head, nn.Linear):
            in_features = first_head.in_features
            model.heads = nn.Sequential(nn.Linear(in_features, num_classes))
            return
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
         # Swin Transformer
         in_features = model.head.in_features
         model.head = nn.Linear(in_features, num_classes)
         return

    # Fallback
    log.warning(
        f"Could not automatically replace head for '{model_name}'. "
        f"Model output will remain at default (usually 1000 classes). "
        "Please check model structure manually."
    )

def create_model(name: str, num_classes: int, task_type: str, **kwargs) -> nn.Module:
    """
    Adapter for Torchvision Models.
    """

    # kwargs usually contains 'weights'. 
    # Modern torchvision uses 'weights="DEFAULT"' instead of 'pretrained=True'
    weights = kwargs.pop("weights", None)
    pretrained = kwargs.pop("pretrained", False)
    
    # Handle legacy 'pretrained' arg if passed from config
    if pretrained and weights is None:
        # We try to guess the default weight enum name, but it's messy.
        # It's safer to just let the user specify "weights: ResNet50_Weights.DEFAULT" in config
        # OR just use "weights='DEFAULT'" which works for newer torchvision versions
        weights = "DEFAULT"

    log.info(f"Initializing Torchvision Model: {name} | Weights: {weights}")

    if not hasattr(models, name):
        raise ValueError(f"Model '{name}' not found in torchvision.models")

    # Instantiate
    model_fn = getattr(models, name)
    with hpc_proxy():
        try:
            model = model_fn(weights=weights, **kwargs)
        except TypeError:
            # Fallback for older torchvision versions
            model = model_fn(pretrained=pretrained, **kwargs)

    # Surgery
    _replace_head(model, num_classes, name)

    return model