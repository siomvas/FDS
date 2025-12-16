"""
Model Factory for NVFlare FL Training

Provides unified interface to existing models for classification and segmentation tasks.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import sys
import torch
import torch.nn as nn

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.classification import (
    get_vgg11,
    get_resnet18_pretrained,
    get_resnet50_scratch,
    get_resnet50_pretrained,
    get_convnext_small_pretrained,
    get_vit_b16_pretrained
)
from models.segmentation import get_deeplabv3_resnet50


class ModelConfig:
    """Configuration for models"""

    # Available models per task type
    CLASSIFICATION_MODELS = [
        "vgg11",
        "resnet18",
        "resnet50",
        "resnet50_pretrained",
        "convnext_small",
        "vit_b16"
    ]

    SEGMENTATION_MODELS = [
        "deeplabv3_resnet50"
    ]

    # Default models per dataset
    DATASET_DEFAULTS = {
        "tiny_imagenet": {
            "model": "vgg11",
            "num_classes": 200,
            "image_size": 64
        },
        "food101": {
            "model": "resnet50_pretrained",
            "num_classes": 101,
            "image_size": 224
        },
        "ade20k": {
            "model": "deeplabv3_resnet50",
            "num_classes": 150,
            "image_size": 512
        }
    }


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to get a model for FL training.

    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (when available)
        **kwargs: Additional model-specific arguments

    Returns:
        PyTorch model instance

    Available Classification Models:
        - vgg11: VGG-11 from scratch
        - resnet18: ResNet-18 pretrained
        - resnet50: ResNet-50 from scratch
        - resnet50_pretrained: ResNet-50 pretrained
        - convnext_small: ConvNeXt-Small pretrained
        - vit_b16: Vision Transformer B/16 pretrained

    Available Segmentation Models:
        - deeplabv3_resnet50: DeepLabV3 with ResNet-50 backbone

    Examples:
        # Classification
        >>> model = get_model("vgg11", num_classes=200)
        >>> model = get_model("resnet50_pretrained", num_classes=101)

        # Segmentation
        >>> model = get_model("deeplabv3_resnet50", num_classes=150)
    """
    model_name = model_name.lower()

    # Classification models
    if model_name == "vgg11":
        model = get_vgg11(num_classes=num_classes)

    elif model_name == "resnet18":
        model = get_resnet18_pretrained(num_classes=num_classes)

    elif model_name == "resnet50":
        if pretrained:
            model = get_resnet50_pretrained(num_classes=num_classes)
        else:
            model = get_resnet50_scratch(num_classes=num_classes)

    elif model_name == "resnet50_pretrained":
        model = get_resnet50_pretrained(num_classes=num_classes)

    elif model_name == "convnext_small":
        model = get_convnext_small_pretrained(num_classes=num_classes)

    elif model_name == "vit_b16":
        model = get_vit_b16_pretrained(num_classes=num_classes)

    # Segmentation models
    elif model_name == "deeplabv3_resnet50":
        model = get_deeplabv3_resnet50(num_classes=num_classes)

    else:
        available_models = (
            ModelConfig.CLASSIFICATION_MODELS +
            ModelConfig.SEGMENTATION_MODELS
        )
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available_models}"
        )

    return model


def get_model_for_dataset(
    dataset_name: str,
    model_name: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Get a model configured for a specific dataset.

    Uses dataset defaults if model_name is not specified.

    Args:
        dataset_name: Name of dataset ('tiny_imagenet', 'food101', 'ade20k')
        model_name: Model architecture (uses default if None)
        **kwargs: Additional arguments for get_model

    Returns:
        Configured PyTorch model

    Examples:
        # Use default model for dataset
        >>> model = get_model_for_dataset("tiny_imagenet")  # Returns VGG-11

        # Override default
        >>> model = get_model_for_dataset("food101", model_name="resnet18")
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in ModelConfig.DATASET_DEFAULTS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(ModelConfig.DATASET_DEFAULTS.keys())}"
        )

    config = ModelConfig.DATASET_DEFAULTS[dataset_name]

    # Use default model if not specified
    if model_name is None:
        model_name = config["model"]

    # Get number of classes
    num_classes = kwargs.pop("num_classes", config["num_classes"])

    return get_model(
        model_name=model_name,
        num_classes=num_classes,
        **kwargs
    )


def get_task_type(model_name: str) -> str:
    """
    Determine task type from model name.

    Args:
        model_name: Model architecture name

    Returns:
        'classification' or 'segmentation'
    """
    model_name = model_name.lower()

    if model_name in ModelConfig.CLASSIFICATION_MODELS:
        return "classification"
    elif model_name in ModelConfig.SEGMENTATION_MODELS:
        return "segmentation"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params
    }


def freeze_backbone(model: nn.Module, model_name: str) -> nn.Module:
    """
    Freeze backbone layers for fine-tuning.

    Useful for transfer learning in FL scenarios.

    Args:
        model: PyTorch model
        model_name: Model architecture name

    Returns:
        Model with frozen backbone
    """
    model_name = model_name.lower()

    if "resnet" in model_name:
        # Freeze all layers except final FC layer
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    elif "vit" in model_name:
        # Freeze all except classifier head
        for name, param in model.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    elif "convnext" in model_name:
        # Freeze all except classifier
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    elif "deeplabv3" in model_name:
        # Freeze backbone, keep decoder trainable
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

    return model


def get_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer for model.

    Args:
        model: PyTorch model
        optimizer_type: Optimizer type ('adam', 'sgd', 'adamw')
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        PyTorch optimizer
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "sgd":
        momentum = kwargs.pop("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer_type}'. "
            f"Available: adam, sgd, adamw"
        )

    return optimizer


if __name__ == "__main__":
    # Example usage and testing
    print("Testing model_factory.py...")

    # Test classification models
    print("\nClassification Models:")
    for model_name in ["vgg11", "resnet18", "resnet50_pretrained"]:
        model = get_model(model_name, num_classes=200)
        params = count_parameters(model)
        print(f"  {model_name}: {params['trainable']:,} trainable params")

    # Test segmentation models
    print("\nSegmentation Models:")
    model = get_model("deeplabv3_resnet50", num_classes=150)
    params = count_parameters(model)
    print(f"  deeplabv3_resnet50: {params['trainable']:,} trainable params")

    # Test dataset-specific models
    print("\nDataset-specific models:")
    for dataset in ["tiny_imagenet", "food101", "ade20k"]:
        model = get_model_for_dataset(dataset)
        task_type = get_task_type(ModelConfig.DATASET_DEFAULTS[dataset]["model"])
        print(f"  {dataset}: {task_type}")

    print("\n✓ Model factory module ready!")
