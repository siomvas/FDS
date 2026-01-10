"""BaseLearner: Abstract base class for NVFlare 2.7 task-specific learners.

Provides shared infrastructure for Client API integration, gradient tracking,
TensorBoard logging, and model management. Task-specific learners (classification,
segmentation, detection) inherit from this class.

Architecture Decision: Separate Learners with shared BaseLearner (Phase 0 - Option B)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nvflare.client as flare
from nvflare.client.tracking import LogWriterName

# Existing FDS imports
from fl_training.nvflare_refactor.common.gradient_tracker import ClientGradientTracker


class BaseLearner(ABC):
    """Abstract base learner for NVFlare 2.7 Client API integration.

    Shared responsibilities:
    - NVFlare Client API lifecycle (init, receive, send)
    - Gradient tracking via ClientGradientTracker
    - TensorBoard logging integration
    - Model state management
    - Reproducible seeding
    - AMP support

    Task-specific learners must implement:
    - create_model() - Task-specific model architecture
    - train_epoch() - Task-specific training loop
    - evaluate() - Task-specific evaluation metrics

    Args:
        data_path: Path to dataset splits (DatasetSplits JSON)
        model_name: Model architecture identifier
        config: Training configuration dict with keys:
            - lr: Learning rate
            - local_epochs: Epochs per round
            - batch_size: Training batch size
            - device: torch device
            - amp_enabled: Use automatic mixed precision
            - grad_tracking_enabled: Enable gradient diversity tracking
            - seed: Random seed
        client_id: Client identifier (auto-detected if None)
    """

    def __init__(
        self,
        data_path: str,
        model_name: str,
        config: Dict[str, Any],
        client_id: Optional[str] = None
    ):
        # NVFlare Client API initialization
        flare.init()
        self.client_id = client_id or flare.get_site_name()
        self.job_id = flare.get_job_id()

        # Configuration
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # Training state
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.current_round = 0

        # AMP setup
        if config.get("amp_enabled", False):
            self.scaler = torch.cuda.amp.GradScaler()

        # Gradient tracking
        self.grad_tracker: Optional[ClientGradientTracker] = None
        if config.get("grad_tracking_enabled", False):
            self.grad_tracker = ClientGradientTracker(client_id=self.client_id)

        # TensorBoard logging
        flare.init_tracking(LogWriterName.TORCH_TB)

        # Seeding
        self._set_seed(config.get("seed", 42))

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create task-specific model architecture.

        Must implement model creation, load to device, handle HPC proxy if needed.
        Returns: PyTorch model ready for training
        """
        pass

    @abstractmethod
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch with task-specific logic.

        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer instance
            epoch: Current epoch number

        Returns:
            Dict of training metrics (e.g., {"loss": 0.5, "accuracy": 0.9})
        """
        pass

    @abstractmethod
    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with task-specific metrics.

        Args:
            model: Model to evaluate
            val_loader: Validation data loader

        Returns:
            Dict of validation metrics
        """
        pass

    def execute(self):
        """Main execution loop using NVFlare Client API.

        Implements the standard NVFlare training flow:
        1. Receive global model from server
        2. Load into local model
        3. Train for local_epochs
        4. Send updated model back to server
        """
        # Initialize model and optimizer
        self.model = self.create_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("lr", 0.001)
        )

        # Load data (to be implemented by subclass or use data_factory)
        train_loader, val_loader = self._load_data()

        # NVFlare training loop
        while flare.is_running():
            # Receive global model
            input_model = flare.receive()
            self.current_round = input_model.current_round if hasattr(input_model, 'current_round') else self.current_round + 1

            # Load global weights
            if input_model.params:
                self.model.load_state_dict(input_model.params)

            # Local training
            train_metrics = {}
            for epoch in range(self.config.get("local_epochs", 1)):
                epoch_metrics = self.train_epoch(
                    self.model,
                    train_loader,
                    self.optimizer,
                    epoch
                )
                train_metrics.update({f"train_{k}": v for k, v in epoch_metrics.items()})

                # Gradient tracking (if enabled)
                if self.grad_tracker:
                    self.grad_tracker.track_epoch(self.model, epoch)

            # Validation
            val_metrics = self.evaluate(self.model, val_loader)
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            if self.grad_tracker:
                grad_stats = self.grad_tracker.get_round_statistics()
                all_metrics.update({f"grad_{k}": v for k, v in grad_stats.items()})

            # Send updated model
            output_model = flare.FLModel(
                params=self.model.state_dict(),
                metrics=all_metrics,
                meta={"round": self.current_round, "client_id": self.client_id}
            )
            flare.send(output_model)

    def _load_data(self):
        """Load training and validation data.

        Placeholder - will integrate with data_factory in Plan 01-04.
        Subclasses can override for task-specific data loading.
        """
        raise NotImplementedError("Data loading will be integrated with data_factory in Plan 01-04")
