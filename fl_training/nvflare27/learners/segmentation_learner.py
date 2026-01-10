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
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

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

        Uses existing deeplabv3_resnet50 from models.segmentation with HPC proxy handling.
        Model outputs a dictionary with 'out' key containing logits per pixel.

        Returns:
            nn.Module: DeepLabV3 model ready for training, loaded to self.device

        Notes:
            - Uses HPC proxy for pretrained weight downloads
            - DeepLabV3 forward pass returns dict with 'out' key
            - Model expects input shape (N, 3, H, W)
            - Output shape (N, num_classes, H, W) - logits per pixel
            - For ADE20K: num_classes=151 (150 semantic classes + 1 background)

        Examples:
            >>> learner = SegmentationLearner(...)
            >>> model = learner.create_model()
            >>> outputs = model(images)
            >>> logits = outputs['out']  # Extract segmentation logits
            >>> print(logits.shape)  # (N, 151, H, W)
        """
        # Set HPC proxy for pretrained model downloads
        os.environ['https_proxy'] = 'http://proxy.example.org:3128'

        try:
            # Create DeepLabV3 model using existing model function
            model = deeplabv3_resnet50(pretrained=True, num_classes=self.num_classes)
        finally:
            # Always clear proxy after model creation
            os.environ.pop('https_proxy', None)

        # Move model to device
        model = model.to(self.device)

        return model

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch with segmentation-specific logic.

        Implements pixel-wise cross-entropy loss training with pixel accuracy metrics.
        Supports automatic mixed precision (AMP) if enabled via config. Loads masks
        from mask_path strings in batch dict. Logs progress every 10 batches.

        Args:
            model: Segmentation model to train (DeepLabV3)
            train_loader: Training data loader (from FDSDataFactory)
            optimizer: Optimizer instance (from BaseLearner)
            epoch: Current epoch number (0-indexed)

        Returns:
            Dict with keys:
                - "loss": Average pixel-wise cross-entropy loss
                - "pixel_accuracy": Correct pixels / total pixels

        Examples:
            >>> metrics = learner.train_epoch(model, train_loader, optimizer, epoch=0)
            >>> print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['pixel_accuracy']:.4f}")
        """
        # Set model to training mode
        model.train()

        # Initialize metrics tracking
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Extract images from batch dict
            images = batch["image"].to(self.device)
            mask_paths = batch["mask_path"]

            # Load masks from paths
            masks = []
            for path in mask_paths:
                mask = Image.open(path)
                mask = torch.from_numpy(np.array(mask)).long()
                masks.append(mask)
            masks = torch.stack(masks).to(self.device)  # Shape: (N, H, W)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.scaler is not None:
                # AMP enabled: use autocast for forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(images)['out']  # Extract logits from DeepLabV3 output dict

                    # Upsample if needed to match mask size
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(
                            outputs,
                            size=masks.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    loss = criterion(outputs, masks)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard training without AMP
                outputs = model(images)['out']  # Extract logits from DeepLabV3 output dict

                # Upsample if needed to match mask size
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            # Track metrics
            running_loss += loss.item() * images.size(0)

            # Compute pixel accuracy (argmax over channel dimension)
            preds = outputs.argmax(dim=1)  # Shape: (N, H, W)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                print(
                    f"[Client {self.client_id}] Epoch {epoch+1}, "
                    f"Batch {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Compute epoch metrics with zero-division handling
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        avg_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0

        return {"loss": avg_loss, "pixel_accuracy": pixel_accuracy}

    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with segmentation-specific metrics.

        Computes IoU (Intersection over Union), pixel accuracy, and loss on validation set.
        Returns zero metrics if val_loader is None (e.g., when using global validation).
        Uses torch.no_grad() for efficient inference without gradient computation.

        Args:
            model: Segmentation model to evaluate (DeepLabV3)
            val_loader: Validation data loader (may be None)

        Returns:
            Dict with keys:
                - "loss": Average validation loss
                - "pixel_accuracy": Pixel-wise accuracy
                - "mean_iou": Mean Intersection over Union across classes
            Returns {"loss": 0.0, "pixel_accuracy": 0.0, "mean_iou": 0.0} if val_loader is None

        Examples:
            >>> metrics = learner.evaluate(model, val_loader)
            >>> print(f"Val Loss: {metrics['loss']:.4f}, IoU: {metrics['mean_iou']:.4f}")
        """
        # Handle None val_loader (global validation scenario)
        if val_loader is None:
            return {"loss": 0.0, "pixel_accuracy": 0.0, "mean_iou": 0.0}

        # Set model to evaluation mode
        model.eval()

        # Initialize metrics
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        iou_per_class = defaultdict(lambda: {"intersection": 0, "union": 0})

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Evaluation loop without gradient computation
        with torch.no_grad():
            for batch in val_loader:
                # Extract images from batch dict
                images = batch["image"].to(self.device)
                mask_paths = batch["mask_path"]

                # Load masks from paths
                masks = []
                for path in mask_paths:
                    mask = Image.open(path)
                    mask = torch.from_numpy(np.array(mask)).long()
                    masks.append(mask)
                masks = torch.stack(masks).to(self.device)  # Shape: (N, H, W)

                # Forward pass (no AMP in evaluation)
                outputs = model(images)['out']  # Extract logits from DeepLabV3 output dict

                # Upsample if needed to match mask size
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                loss = criterion(outputs, masks)

                # Track metrics
                running_loss += loss.item() * images.size(0)

                # Compute predictions
                preds = outputs.argmax(dim=1)  # Shape: (N, H, W)

                # Pixel accuracy
                correct_pixels += (preds == masks).sum().item()
                total_pixels += masks.numel()

                # Compute IoU per class
                for cls in range(self.num_classes):
                    pred_mask = (preds == cls)
                    true_mask = (masks == cls)
                    intersection = (pred_mask & true_mask).sum().item()
                    union = (pred_mask | true_mask).sum().item()
                    iou_per_class[cls]["intersection"] += intersection
                    iou_per_class[cls]["union"] += union

        # Compute mean IoU
        ious = []
        for cls in range(self.num_classes):
            union = iou_per_class[cls]["union"]
            if union > 0:
                iou = iou_per_class[cls]["intersection"] / union
                ious.append(iou)
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0

        # Compute average metrics with zero-division handling
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        avg_loss = running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0

        return {"loss": avg_loss, "pixel_accuracy": pixel_accuracy, "mean_iou": mean_iou}
