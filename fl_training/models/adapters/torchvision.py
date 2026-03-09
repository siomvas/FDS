import logging
from typing import Callable
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


def _replace_detection_head(model: nn.Module, num_classes: int, model_name: str):
    """Replace detection head for torchvision two-stage detectors when possible."""
    roi_heads = getattr(model, "roi_heads", None)
    predictor = getattr(roi_heads, "box_predictor", None) if roi_heads is not None else None
    cls_score = getattr(predictor, "cls_score", None) if predictor is not None else None

    if isinstance(cls_score, nn.Linear):
        in_features = cls_score.in_features
        try:
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

    log.warning(
        f"Could not automatically replace detection head for '{model_name}'. "
        "Prefer passing num_classes in model args for this architecture."
    )


def _make_conv_with_new_out_channels(module: nn.Conv2d, num_classes: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=module.in_channels,
        out_channels=num_classes,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
    )


def _replace_last_conv(module: nn.Module, num_classes: int):
    if isinstance(module, nn.Conv2d):
        return _make_conv_with_new_out_channels(module, num_classes), True

    if isinstance(module, nn.Sequential):
        for idx in range(len(module) - 1, -1, -1):
            if isinstance(module[idx], nn.Conv2d):
                module[idx] = _make_conv_with_new_out_channels(module[idx], num_classes)
                return module, True
    return module, False


def _replace_segmentation_head(model: nn.Module, num_classes: int, model_name: str):
    replaced = False
    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        classifier, replaced = _replace_last_conv(classifier, num_classes)
        if replaced:
            model.classifier = classifier

    aux_classifier = getattr(model, "aux_classifier", None)
    if aux_classifier is not None:
        aux_classifier, aux_replaced = _replace_last_conv(aux_classifier, num_classes)
        if aux_replaced:
            model.aux_classifier = aux_classifier

    if replaced:
        return

    log.warning(
        f"Could not automatically replace segmentation head for '{model_name}'. "
        "Prefer passing num_classes in model args for this architecture."
    )


def _resolve_model_fn(name: str, task_type: str) -> Callable:
    if task_type == "classification":
        if hasattr(models, name):
            return getattr(models, name)
        raise ValueError(
            f"Model '{name}' not found in torchvision.models for classification task."
        )

    if task_type == "detection":
        if hasattr(models, "detection") and hasattr(models.detection, name):
            return getattr(models.detection, name)
        raise ValueError(
            f"Model '{name}' not found in torchvision.models.detection for detection task."
        )

    if task_type == "segmentation":
        if hasattr(models, "segmentation") and hasattr(models.segmentation, name):
            return getattr(models.segmentation, name)
        raise ValueError(
            f"Model '{name}' not found in torchvision.models.segmentation for segmentation task."
        )

    raise ValueError(f"Unsupported task type for torchvision adapter: {task_type}")


def _instantiate_model(model_fn: Callable, weights, pretrained: bool, kwargs):
    errors = []
    attempts = []
    if weights is not None:
        attempts.append(dict(kwargs, weights=weights))
    attempts.append(dict(kwargs))
    if pretrained:
        attempts.append(dict(kwargs, pretrained=True))

    for call_kwargs in attempts:
        try:
            return model_fn(**call_kwargs)
        except (TypeError, ValueError) as exc:
            errors.append(exc)

    raise RuntimeError(f"Failed to initialize torchvision model. Last error: {errors[-1]}")


def create_model(name: str, num_classes: int, task_type: str, **kwargs) -> nn.Module:
    """
    Adapter for Torchvision Models.
    """
    task_type = task_type.lower()
    if task_type not in {"classification", "segmentation", "detection"}:
        raise ValueError(f"Unsupported task type for torchvision adapter: {task_type}")

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

    log.info(f"Initializing Torchvision Model: {name} | Task: {task_type} | Weights: {weights}")

    # Instantiate
    model_fn = _resolve_model_fn(name, task_type)
    model_kwargs = dict(kwargs)
    if task_type in {"segmentation", "detection"}:
        model_kwargs.setdefault("num_classes", num_classes)

    with hpc_proxy():
        model = _instantiate_model(model_fn, weights=weights, pretrained=pretrained, kwargs=model_kwargs)

    # Surgery
    if task_type == "classification":
        if "num_classes" not in model_kwargs:
            _replace_head(model, num_classes, name)
    elif task_type == "segmentation":
        if "num_classes" not in model_kwargs:
            _replace_segmentation_head(model, num_classes, name)
    else:
        if "num_classes" not in model_kwargs:
            _replace_detection_head(model, num_classes, name)

    return model
