"""
Object Detection Models for Federated Learning

Provides factory functions for detection models with HPC proxy support for
pretrained weight downloads. Currently supports Faster R-CNN with ResNet-50-FPN backbone.
"""

import torch
import torch.nn as nn
import torchvision.models.detection as detection_models
from utils.proxy_config import hpc_proxy


def get_fasterrcnn_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create Faster R-CNN model with ResNet-50-FPN backbone.

    Faster R-CNN is a two-stage object detection model consisting of:
    1. Region Proposal Network (RPN): Generates object proposals
    2. Fast R-CNN detector: Classifies proposals and refines bounding boxes

    Args:
        num_classes (int): Number of object classes to detect
            - For COCO without background: 80
            - For COCO with background: 91
            - For custom datasets: number of classes (model adds background class internally)
        pretrained (bool): Whether to load pretrained weights from COCO dataset
            - True: Load COCO pretrained weights, then modify classifier head
            - False: Initialize from scratch

    Returns:
        nn.Module: Faster R-CNN model ready for training

    Architecture:
        - Backbone: ResNet-50 with Feature Pyramid Network (FPN)
        - RPN: Generates region proposals at multiple scales
        - ROI Head: Region of Interest pooling + classifier + box regressor
        - Pretrained on COCO dataset (if pretrained=True)

    HPC Proxy:
        Uses centralized proxy configuration for downloading pretrained weights in
        restricted network environments (HPC cluster).

    Examples:
        >>> # COCO detection (80 classes)
        >>> model = get_fasterrcnn_resnet50(num_classes=80, pretrained=True)
        >>> # Custom dataset (10 classes)
        >>> model = get_fasterrcnn_resnet50(num_classes=10, pretrained=False)

    Note:
        The model expects inputs as List[Tensor] where each Tensor is a CHW image.
        During training, targets must be a List[Dict] with keys:
        - boxes: FloatTensor[N, 4] in (x1, y1, x2, y2) format
        - labels: Int64Tensor[N] with class labels (1-indexed, 0 is background)

        During inference, returns List[Dict] with keys:
        - boxes: FloatTensor[N, 4] predicted boxes
        - labels: Int64Tensor[N] predicted class labels
        - scores: FloatTensor[N] detection confidence scores
    """
    # Use centralized proxy configuration for pretrained weight downloads
    with hpc_proxy():
        if pretrained:
            # Load pretrained weights from COCO
            model = detection_models.fasterrcnn_resnet50_fpn(
                weights='DEFAULT',
                num_classes=num_classes
            )
        else:
            # Initialize from scratch
            model = detection_models.fasterrcnn_resnet50_fpn(
                weights=None,
                num_classes=num_classes
            )

    return model
