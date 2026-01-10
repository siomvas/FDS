"""SegmentationLearner: Segmentation-specific learner for NVFlare 2.7 Client API.

Extends BaseLearner with semantic segmentation model creation, training loop, and evaluation.
Integrates with existing deeplabv3_resnet50 model for ADE20K dataset (150 semantic classes + 1 background).

Task-specific responsibilities:
- Create DeepLabV3 ResNet50 segmentation models
- Implement pixel-wise cross-entropy loss training loop
- Compute IoU and mIoU metrics for segmentation evaluation
"""

import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_learner import BaseLearner
from models.segmentation import deeplabv3_resnet50


class SegmentationLearner(BaseLearner):
    """Segmentation-specific learner extending BaseLearner.

    Implements semantic segmentation model creation, training, and evaluation for ADE20K dataset.
    Uses DeepLabV3 ResNet50 architecture with pixel-wise cross-entropy loss and IoU metrics.

    Args:
        data_path: Path to dataset splits (DatasetSplits JSON)
        model_name: Model architecture identifier (deeplabv3_resnet50)
        config: Training configuration dict with keys:
            - num_classes (int): Number of output classes (REQUIRED, ADE20K: 151)
            - lr: Learning rate
            - local_epochs: Epochs per round
            - batch_size: Training batch size
            - device: torch device
            - amp_enabled: Use automatic mixed precision
            - grad_tracking_enabled: Enable gradient diversity tracking
            - seed: Random seed
        client_id: Client identifier (auto-detected if None)

    Raises:
        ValueError: If num_classes not in config

    Examples:
        >>> config = {
        ...     "num_classes": 151,  # ADE20K: 150 semantic + 1 background
        ...     "lr": 0.001,
        ...     "local_epochs": 1,
        ...     "batch_size": 8,
        ...     "device": "cuda",
        ...     "splits_path": "outputs/splits.json",
        ...     "task_type": "segmentation"
        ... }
        >>> learner = SegmentationLearner(
        ...     data_path="outputs/splits.json",
        ...     model_name="deeplabv3_resnet50",
        ...     config=config
        ... )
        >>> learner.execute()
    """

    def __init__(
        self,
        data_path: str,
        model_name: str,
        config: Dict[str, Any],
        client_id: Optional[str] = None
    ):
        # Initialize base learner
        super().__init__(data_path, model_name, config, client_id)

        # Extract and validate num_classes
        if "num_classes" not in config:
            raise ValueError(
                "Missing required config key 'num_classes'. "
                "SegmentationLearner requires num_classes to be specified. "
                "Example: config = {'num_classes': 151, ...}"
            )

        self.num_classes = config["num_classes"]

    def create_model(self) -> nn.Module:
        """Create DeepLabV3 ResNet50 segmentation model.

        Returns:
            nn.Module: DeepLabV3 model ready for training, loaded to self.device
        """
        pass  # TODO: Implement in Task 2

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch with segmentation-specific logic.

        TODO: Implement in next plan (04-02-PLAN.md)

        Args:
            model: Segmentation model to train
            train_loader: Training data loader (from FDSDataFactory)
            optimizer: Optimizer instance (from BaseLearner)
            epoch: Current epoch number (0-indexed)

        Returns:
            Dict with keys:
                - "loss": Average pixel-wise cross-entropy loss
                - "iou": Mean Intersection over Union
        """
        pass  # TODO: Implement in Task 04-02

    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with segmentation-specific metrics.

        TODO: Implement in next plan (04-02-PLAN.md)

        Args:
            model: Segmentation model to evaluate
            val_loader: Validation data loader (may be None)

        Returns:
            Dict with keys:
                - "loss": Average validation loss
                - "iou": Mean Intersection over Union
                - "pixel_accuracy": Pixel-wise accuracy
        """
        pass  # TODO: Implement in Task 04-02
