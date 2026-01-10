"""ClassificationLearner: Classification-specific learner for NVFlare 2.7 Client API.

Extends BaseLearner with classification model creation, training loop, and evaluation.
Integrates with existing model_factory for model instantiation and handles HPC proxy
requirements for pretrained model downloads.

Task-specific responsibilities:
- Create classification models via model_factory
- Implement cross-entropy loss training loop
- Compute accuracy and loss metrics
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_learner import BaseLearner
from utils.proxy_config import hpc_proxy

# Add parent directories to path for model_factory import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from fl_training.nvflare_refactor.common.model_factory import get_model


class ClassificationLearner(BaseLearner):
    """Classification-specific learner extending BaseLearner.

    Implements classification model creation, training, and evaluation using
    cross-entropy loss and accuracy metrics. Integrates with existing model_factory
    for model instantiation.

    Args:
        data_path: Path to dataset splits (DatasetSplits JSON)
        model_name: Model architecture identifier (vgg11, resnet18, etc.)
        config: Training configuration dict with keys:
            - num_classes (int): Number of output classes (REQUIRED)
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
        ...     "num_classes": 200,
        ...     "lr": 0.001,
        ...     "local_epochs": 1,
        ...     "batch_size": 32,
        ...     "device": "cuda",
        ...     "splits_path": "outputs/splits.json",
        ...     "task_type": "classification"
        ... }
        >>> learner = ClassificationLearner(
        ...     data_path="outputs/splits.json",
        ...     model_name="vgg11",
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
                "ClassificationLearner requires num_classes to be specified. "
                "Example: config['num_classes'] = 200"
            )

        self.num_classes = config["num_classes"]

    def create_model(self) -> nn.Module:
        """Create classification model architecture.

        Uses existing model_factory.get_model() to instantiate classification models.
        Uses centralized proxy configuration for pretrained model downloads on HPC.

        Returns:
            nn.Module: Classification model ready for training, loaded to self.device

        Raises:
            ValueError: If model_name is invalid (raised by model_factory)

        Examples:
            >>> learner = ClassificationLearner(...)
            >>> model = learner.create_model()
            >>> print(type(model).__name__)
            'VGG'
        """
        # Use centralized proxy configuration for pretrained model downloads
        with hpc_proxy():
            model = get_model(self.model_name, self.num_classes)

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
        """Execute one training epoch with classification-specific logic.

        Implements cross-entropy loss training with accuracy metrics. Supports
        automatic mixed precision (AMP) if enabled via config. Logs progress
        every 10 batches.

        Args:
            model: Classification model to train
            train_loader: Training data loader (from FDSDataFactory)
            optimizer: Optimizer instance (from BaseLearner)
            epoch: Current epoch number (0-indexed)

        Returns:
            Dict with keys:
                - "loss": Average loss over epoch
                - "accuracy": Accuracy as fraction (0.0 to 1.0)

        Examples:
            >>> metrics = learner.train_epoch(model, train_loader, optimizer, epoch=0)
            >>> print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
        """
        # Set model to training mode
        model.train()

        # Initialize metrics tracking
        running_loss = 0.0
        correct = 0
        total = 0

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Extract inputs and labels from batch dict
            inputs = batch["image"]
            labels = batch["label"]

            # Move to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.scaler is not None:
                # AMP enabled: use autocast for forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard training without AMP
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                print(
                    f"[Client {self.client_id}] Epoch {epoch+1}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Compute epoch metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = running_loss / total if total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with classification-specific metrics.

        Computes accuracy and loss on validation set. Returns zero metrics if
        val_loader is None (e.g., when using global validation). Uses torch.no_grad()
        for efficient inference without gradient computation.

        Args:
            model: Classification model to evaluate
            val_loader: Validation data loader (may be None)

        Returns:
            Dict with keys:
                - "loss": Average validation loss
                - "accuracy": Accuracy as fraction (0.0 to 1.0)
            Returns {"loss": 0.0, "accuracy": 0.0} if val_loader is None

        Examples:
            >>> metrics = learner.evaluate(model, val_loader)
            >>> print(f"Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.4f}")
        """
        # Handle None val_loader (global validation scenario)
        if val_loader is None:
            return {"loss": 0.0, "accuracy": 0.0}

        # Set model to evaluation mode
        model.eval()

        # Initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Evaluation loop without gradient computation
        with torch.no_grad():
            for batch in val_loader:
                # Extract inputs and labels from batch dict
                inputs = batch["image"]
                labels = batch["label"]

                # Move to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass (no AMP in evaluation)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track metrics
                running_loss += loss.item() * inputs.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        # Compute average metrics with zero-division handling
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = running_loss / total if total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}
