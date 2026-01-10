"""DetectionLearner: Object detection learner for NVFlare 2.7 Client API.

Extends BaseLearner with object detection model creation, training loop, and evaluation.
Integrates with existing model_factory for model instantiation and handles HPC proxy
requirements for pretrained model downloads.

Task-specific responsibilities:
- Create detection models (Faster R-CNN) via model_factory
- Implement detection-specific training loop with region proposal losses
- Compute mean Average Precision (mAP) and detection metrics
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_learner import BaseLearner


class DetectionLearner(BaseLearner):
    """Object detection learner extending BaseLearner.

    Implements detection model creation, training, and evaluation using
    Faster R-CNN with region proposal network (RPN) and detection losses.
    Integrates with existing model_factory for model instantiation.

    Args:
        data_path: Path to dataset splits (DatasetSplits JSON)
        model_name: Model architecture identifier (fasterrcnn_resnet50, etc.)
        config: Training configuration dict with keys:
            - num_classes (int): Number of output classes (REQUIRED)
                For COCO: 80 (without background) or 91 (with background)
            - lr: Learning rate
            - local_epochs: Epochs per round
            - batch_size: Training batch size (typically smaller for detection, e.g., 2-4)
            - device: torch device
            - amp_enabled: Use automatic mixed precision
            - grad_tracking_enabled: Enable gradient diversity tracking
            - seed: Random seed
        client_id: Client identifier (auto-detected if None)

    Raises:
        ValueError: If num_classes not in config

    Examples:
        >>> config = {
        ...     "num_classes": 80,
        ...     "lr": 0.001,
        ...     "local_epochs": 1,
        ...     "batch_size": 4,
        ...     "device": "cuda",
        ...     "splits_path": "outputs/splits.json",
        ...     "task_type": "detection",
        ...     "amp_enabled": False
        ... }
        >>> learner = DetectionLearner(
        ...     data_path="outputs/splits.json",
        ...     model_name="fasterrcnn_resnet50",
        ...     config=config
        ... )
        >>> learner.execute()

    Note:
        Detection models output a list of dicts with keys:
        - boxes: FloatTensor[N, 4] in (x1, y1, x2, y2) format
        - labels: Int64Tensor[N] predicted class labels
        - scores: FloatTensor[N] detection confidence scores
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
                "DetectionLearner requires num_classes to be specified. "
                "For COCO detection: num_classes=80 (without background) or 91 (with background). "
                "Example: config['num_classes'] = 80"
            )

        self.num_classes = config["num_classes"]

    def create_model(self) -> nn.Module:
        """Create object detection model architecture.

        Uses existing model_factory.get_model() to instantiate detection models.
        Handles HPC proxy requirements for pretrained model downloads with try-finally
        cleanup pattern.

        Returns:
            nn.Module: Detection model ready for training, loaded to self.device

        Raises:
            ValueError: If model_name is invalid (raised by model_factory)

        Examples:
            >>> learner = DetectionLearner(...)
            >>> model = learner.create_model()
            >>> print(type(model).__name__)
            'FasterRCNN'
        """
        # TODO: Implement in Task 2
        pass

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch with detection-specific logic.

        Implements Faster R-CNN training with RPN and detection losses. Supports
        automatic mixed precision (AMP) if enabled via config. Logs progress
        every 10 batches.

        Args:
            model: Detection model to train
            train_loader: Training data loader (from FDSDataFactory)
            optimizer: Optimizer instance (from BaseLearner)
            epoch: Current epoch number (0-indexed)

        Returns:
            Dict with keys:
                - "loss_classifier": Classification loss
                - "loss_box_reg": Bounding box regression loss
                - "loss_objectness": RPN objectness loss
                - "loss_rpn_box_reg": RPN box regression loss
                - "total_loss": Sum of all losses

        Examples:
            >>> metrics = learner.train_epoch(model, train_loader, optimizer, epoch=0)
            >>> print(f"Total Loss: {metrics['total_loss']:.4f}")
        """
        # TODO: Implement in Plan 05-02 (detection training loop)
        pass

    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with detection-specific metrics.

        Computes mean Average Precision (mAP) and other detection metrics on validation set.
        Returns zero metrics if val_loader is None (e.g., when using global validation).
        Uses torch.no_grad() for efficient inference without gradient computation.

        Args:
            model: Detection model to evaluate
            val_loader: Validation data loader (may be None)

        Returns:
            Dict with keys:
                - "mAP": Mean Average Precision at IoU=0.5
                - "mAP_50_95": Mean Average Precision at IoU=0.5:0.95
            Returns {"mAP": 0.0, "mAP_50_95": 0.0} if val_loader is None

        Examples:
            >>> metrics = learner.evaluate(model, val_loader)
            >>> print(f"mAP: {metrics['mAP']:.4f}")
        """
        # TODO: Implement in Plan 05-02 (detection evaluation metrics)
        pass
