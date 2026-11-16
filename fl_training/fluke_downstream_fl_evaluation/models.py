"""Small Tiny-ImageNet model components used by the Fluke adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


Normalization = Literal["group", "batch", "none"]
TinyImageNetModelName = Literal[
    "tiny_encoder_head",
    "fluke_resnet18gn",
    "torchvision_resnet18_imagenet_bn",
]


def _normalization(channels: int, kind: Normalization) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {kind}")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: Normalization):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _normalization(out_channels, normalization),
            # Opacus wraps activations for DPFedAVG and forbids an in-place
            # modification of its backward-hook views.
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TinyImageNetEncoder(nn.Module):
    """Three-block convolutional encoder with a fixed 128-dimensional embedding."""

    embedding_dim = 128

    def __init__(self, normalization: Normalization = "group"):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(3, 32, normalization),
            _ConvBlock(32, 64, normalization),
            _ConvBlock(64, self.embedding_dim, normalization),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Fluke's prototype helpers inspect a single sample directly.  Conv2d
        # expects a batch dimension, while ordinary client batches already have one.
        if inputs.ndim == 3:
            return self.features(inputs.unsqueeze(0))
        return self.features(inputs)


class TinyImageNetHead(nn.Module):
    """Classifier head with a stable path required by Fluke FedAwS."""

    def __init__(self, embedding_dim: int = 128, num_classes: int = 200):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


class TorchvisionResNet18ImageNetBN(nn.Module):
    """ImageNet-pretrained torchvision ResNet-18 with a Tiny-ImageNet head."""

    def __init__(self, output_size: int = 200, *, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.resnet(inputs)


try:  # This module is only imported by the optional Fluke adapter.
    from fluke.nets import EncoderHeadNet, ResNet18GN
except ImportError:  # pragma: no cover - optional dependency guard
    EncoderHeadNet = None  # type: ignore[assignment,misc]
    ResNet18GN = None  # type: ignore[assignment,misc]


if EncoderHeadNet is not None:

    class TinyImageNetFlukeNet(EncoderHeadNet):
        """Importable ``EncoderHeadNet`` constructor for native parallel Fluke runs."""

        def __init__(
            self, num_classes: int = 200, normalization: Normalization = "group"
        ):
            encoder = TinyImageNetEncoder(normalization=normalization)
            super().__init__(encoder, TinyImageNetHead(encoder.embedding_dim, num_classes))

else:

    class TinyImageNetFlukeNet(nn.Module):  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Fluke support requires `pip install -e '.[fluke]'`.")


@dataclass(frozen=True)
class TinyModelBundle:
    """Full model and optional components required by encoder/head-aware methods."""

    full: nn.Module
    encoder: nn.Module | None
    head: nn.Module | None
    embedding_dim: int | None
    name: TinyImageNetModelName

    def component(self, name: Literal["full", "encoder", "head"]) -> nn.Module:
        if name == "full":
            return self.full
        if name == "encoder" and self.encoder is not None:
            return self.encoder
        if name == "head" and self.head is not None:
            return self.head
        raise ValueError(
            f"Model {self.name!r} does not expose the {name!r} component required by this method."
        )


def build_tiny_imagenet_model_bundle(
    *,
    num_classes: int = 200,
    normalization: Normalization = "group",
    name: TinyImageNetModelName = "tiny_encoder_head",
    pretrained: bool = True,
) -> TinyModelBundle:
    """Build a Tiny-ImageNet model supported by the downstream adapter."""

    if name == "fluke_resnet18gn":
        if ResNet18GN is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("Fluke support requires `pip install -e '.[fluke]'`.")
        return TinyModelBundle(
            full=ResNet18GN(output_size=num_classes),
            encoder=None,
            head=None,
            embedding_dim=None,
            name=name,
        )
    if name == "torchvision_resnet18_imagenet_bn":
        return TinyModelBundle(
            full=TorchvisionResNet18ImageNetBN(
                output_size=num_classes,
                pretrained=pretrained,
            ),
            encoder=None,
            head=None,
            embedding_dim=None,
            name=name,
        )
    if name != "tiny_encoder_head":
        raise ValueError(f"Unsupported Tiny-ImageNet model name: {name}")
    if EncoderHeadNet is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Fluke support requires `pip install -e '.[fluke]'`.")
    full = TinyImageNetFlukeNet(num_classes=num_classes, normalization=normalization)
    return TinyModelBundle(
        full=full,
        encoder=full.encoder,
        head=full.head,
        embedding_dim=TinyImageNetEncoder.embedding_dim,
        name=name,
    )
