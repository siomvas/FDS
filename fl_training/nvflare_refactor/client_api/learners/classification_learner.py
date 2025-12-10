"""
Classification Learner for NVFlare Client API

Handles classification tasks for TinyImageNet and Food101 datasets.
Integrates with existing data_modules, models, and gradient tracking.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Import common modules
from fl_training.nvflare_refactor.common.data_factory import get_datamodule, get_dataset_info
from fl_training.nvflare_refactor.common.model_factory import get_model_for_dataset, get_optimizer
from fl_training.nvflare_refactor.common.gradient_tracker import ClientGradientTracker

# NVFlare imports
try:
    import nvflare.client as flare
    from nvflare.app_common.tracking.tracker_types import LogWriterName, TBWriter
except ImportError:
    print("Warning: NVFlare not installed. Install with: pip install nvflare")
    flare = None


class ClassificationLearner:
    """
    Client-side learner for image classification tasks.

    Supports:
    - TinyImageNet (200 classes)
    - Food101 (101 classes)
    - All splitting methods (IID, Dirichlet, VLM-guided, Embeddings)
    - Gradient diversity tracking
    - TensorBoard logging
    - Mixed precision training
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        split_file: Optional[str] = None,
        # Training hyperparameters
        batch_size: int = 64,
        num_epochs: int = 5,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam",
        weight_decay: float = 0.0,
        # Device and performance
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = False,
        num_workers: int = 4,
        # Gradient tracking
        track_gradients: bool = True,
        track_per_layer: bool = False,
        # Logging
        log_frequency: int = 10,
        seed: int = 42,
    ):
        """
        Args:
            dataset_name: 'tiny_imagenet' or 'food101'
            model_name: Model architecture (uses dataset default if None)
            data_dir: Path to dataset (uses default if None)
            split_file: Path to client split file
            batch_size: Training batch size
            num_epochs: Epochs per FL round
            learning_rate: Learning rate
            optimizer_type: 'adam', 'sgd', or 'adamw'
            weight_decay: L2 regularization
            device: 'cuda' or 'cpu'
            use_amp: Use automatic mixed precision
            num_workers: DataLoader workers
            track_gradients: Enable gradient metrics
            track_per_layer: Track per-layer gradient stats
            log_frequency: Log every N batches
            seed: Random seed
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.data_dir = data_dir
        self.split_file = split_file

        # Hyperparameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay

        # Device
        self.device = torch.device(device)
        self.use_amp = use_amp and (device == "cuda")

        # Data loading
        self.num_workers = num_workers

        # Gradient tracking
        self.track_gradients = track_gradients
        self.gradient_tracker = None
        if track_gradients:
            self.gradient_tracker = ClientGradientTracker(
                track_per_layer=track_per_layer,
                track_weight_divergence=True
            )

        # Logging
        self.log_frequency = log_frequency
        self.seed = seed

        # To be initialized
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        self.datamodule = None
        self.writer = None
        self.client_id = None
        self.global_step = 0

    def initialize(self):
        """Initialize NVFlare client and set up components."""
        # Initialize NVFlare
        if flare is not None:
            flare.init()
            self.client_id = flare.get_site_name()
            print(f"Initialized NVFlare client: {self.client_id}")

            # Get TensorBoard writer
            try:
                self.writer = flare.init_tracking(LogWriterName.TORCH_TB)
                print(f"TensorBoard logging enabled")
            except Exception as e:
                print(f"TensorBoard not available: {e}")
                self.writer = None
        else:
            self.client_id = "local_client_0"
            print("Running without NVFlare (local mode)")

        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Get dataset info
        dataset_info = get_dataset_info(self.dataset_name)
        num_classes = dataset_info["num_classes"]

        # Initialize model
        self.model = get_model_for_dataset(
            dataset_name=self.dataset_name,
            model_name=self.model_name
        ).to(self.device)

        print(f"Model: {self.model_name or 'default'} for {self.dataset_name}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Initialize optimizer
        self.optimizer = get_optimizer(
            self.model,
            optimizer_type=self.optimizer_type,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # AMP scaler
        if self.use_amp:
            self.scaler = GradScaler()
            print("Mixed precision training enabled")

        # Initialize data module
        # Note: client_idx will be extracted from client_id during setup
        self.setup_data()

        print(f"✓ Classification learner initialized")

    def setup_data(self):
        """Set up data module for this client."""
        # Extract client index from client_id
        # Assumes client_id format like "site-1", "site-2", etc.
        try:
            if self.client_id and "site-" in self.client_id:
                client_idx = int(self.client_id.split("-")[1]) - 1
            else:
                client_idx = 0
        except:
            client_idx = 0

        self.datamodule = get_datamodule(
            dataset_name=self.dataset_name,
            data_dir=self.data_dir,
            split_file=self.split_file,
            client_idx=client_idx,
            central=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed
        )

        # Setup data module
        self.datamodule.setup()

        print(f"Data loaded for client {client_idx}")
        print(f"  Train samples: {len(self.datamodule.train_dataloader.dataset)}")
        if hasattr(self.datamodule, 'val_dataloader') and self.datamodule.val_dataloader:
            print(f"  Val samples: {len(self.datamodule.val_dataloader.dataset)}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.datamodule.train_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Logging
            if batch_idx % self.log_frequency == 0:
                accuracy = 100.0 * correct / total
                print(f"  Epoch {epoch} [{batch_idx}/{len(self.datamodule.train_dataloader)}] "
                      f"Loss: {loss.item():.4f} Acc: {accuracy:.2f}%")

                if self.writer:
                    self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/batch_acc", accuracy, self.global_step)

            self.global_step += 1

        # Epoch metrics
        avg_loss = total_loss / len(self.datamodule.train_dataloader)
        accuracy = 100.0 * correct / total

        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Dictionary of validation metrics
        """
        if not hasattr(self.datamodule, 'val_dataloader') or self.datamodule.val_dataloader is None:
            return {"val_loss": 0.0, "val_accuracy": 0.0}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.datamodule.val_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.datamodule.val_dataloader)
        accuracy = 100.0 * correct / total

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }

    def train_round(self) -> Dict[str, Any]:
        """
        Perform one FL training round.

        Returns:
            Dictionary with model parameters and metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Training round started")
        print(f"={'=' * 60}")

        # Store initial weights for gradient tracking
        if self.gradient_tracker:
            self.gradient_tracker.store_initial_weights(self.model)

        # Train for multiple epochs
        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch(epoch)

            # Log epoch metrics
            if self.writer:
                self.writer.add_scalar("train/epoch_loss", train_metrics["train_loss"], epoch)
                self.writer.add_scalar("train/epoch_acc", train_metrics["train_accuracy"], epoch)

            print(f"  Epoch {epoch} - Loss: {train_metrics['train_loss']:.4f} "
                  f"Acc: {train_metrics['train_accuracy']:.2f}%")

        # Validation
        val_metrics = self.validate()
        print(f"  Validation - Loss: {val_metrics['val_loss']:.4f} "
              f"Acc: {val_metrics['val_accuracy']:.2f}%")

        if self.writer:
            self.writer.add_scalar("val/loss", val_metrics["val_loss"], self.global_step)
            self.writer.add_scalar("val/accuracy", val_metrics["val_accuracy"], self.global_step)

        # Compute gradient metrics
        gradient_metrics = {}
        if self.gradient_tracker:
            gradient_metrics = self.gradient_tracker.compute_all_metrics(self.model)
            print(f"  Gradient norm: {gradient_metrics.get('gradient_norm_mean', 0):.6f}")

            if self.writer:
                for key, value in gradient_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"gradients/{key}", value, self.global_step)

        # Combine all metrics
        all_metrics = {
            **train_metrics,
            **val_metrics,
            **gradient_metrics,
            "num_samples": len(self.datamodule.train_dataloader.dataset)
        }

        return {
            "params": self.model.state_dict(),
            "metrics": all_metrics
        }

    def run(self):
        """
        Main FL training loop using NVFlare Client API.
        """
        # Initialize
        self.initialize()

        round_num = 0
        while flare.is_running():
            # Receive global model from server
            input_model = flare.receive()

            if input_model is None:
                print("No model received, stopping...")
                break

            print(f"\n{'#' * 60}")
            print(f"# FL Round {round_num}")
            print(f"{'#' * 60}")

            # Load global model weights
            if input_model.params:
                self.model.load_state_dict(input_model.params)
                print(f"Loaded global model weights")

            # Train for one round
            result = self.train_round()

            # Create output model
            output_model = flare.FLModel(
                params=result["params"],
                metrics=result["metrics"],
                meta={"round": round_num}
            )

            # Send to server
            flare.send(output_model)
            print(f"Sent model update to server")

            round_num += 1

        print(f"\n{'#' * 60}")
        print(f"# Training completed after {round_num} rounds")
        print(f"{'#' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classification Learner for NVFlare")
    parser.add_argument("--dataset", type=str, default="tiny_imagenet",
                        help="Dataset name (tiny_imagenet, food101)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (uses dataset default if None)")
    parser.add_argument("--split_file", type=str, required=True,
                        help="Path to client split file")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true")

    args = parser.parse_args()

    # Create and run learner
    learner = ClassificationLearner(
        dataset_name=args.dataset,
        model_name=args.model,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        optimizer_type=args.optimizer,
        device=args.device,
        use_amp=args.use_amp
    )

    learner.run()
