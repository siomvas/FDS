"""
Base federated learning trainer with gradient diversity tracking.

Provides abstract base class for FL training with built-in support for:
- Gradient diversity metrics
- Training history logging
- Visualization utilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.gradient_metrics import GradientDiversityMetric, analyze_gradient_diversity


@dataclass
class FLConfig:
    """Configuration for federated learning training."""

    # FL settings
    n_clients: int
    n_rounds: int
    local_epochs: int
    clients_per_round: int  # Number of clients sampled per round

    # Optimization
    learning_rate: float
    batch_size: int
    optimizer: str = "sgd"  # "sgd", "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0

    # Gradient diversity tracking
    track_grad_diversity: bool = True
    grad_metric_type: str = "cosine"  # "cosine", "variance", "conflict", "l2"
    grad_normalization: str = "layer_wise"  # "none", "l2", "layer_wise"
    grad_layer_filter: Optional[List[str]] = None  # Filter specific layers

    # Logging
    eval_every: int = 1  # Evaluate every N rounds
    log_dir: str = "./logs"
    save_checkpoints: bool = True
    checkpoint_every: int = 10

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, filepath: str):
        """Save configuration to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, filepath: str) -> "FLConfig":
        """Load configuration from JSON."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class BaseClient:
    """
    Base class for federated learning client.

    Each client maintains local data and can compute gradients.
    """

    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: str = "cpu"
    ):
        """
        Args:
            client_id: Unique identifier for this client
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (can be None)
            device: Device to train on
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Model will be set by trainer
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None

    def set_model(self, model: nn.Module):
        """Set the model for this client."""
        self.model = model.to(self.device)

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set the optimizer for this client."""
        self.optimizer = optimizer

    def set_criterion(self, criterion: nn.Module):
        """Set the loss criterion for this client."""
        self.criterion = criterion

    def train_local(self, n_epochs: int) -> Dict[str, float]:
        """
        Perform local training for n_epochs.

        Args:
            n_epochs: Number of local epochs

        Returns:
            Dictionary with training metrics (loss, accuracy)
        """
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model, optimizer, or criterion not set")

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(n_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "n_samples": total_samples
        }

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get current model gradients.

        Returns:
            Dictionary mapping parameter names to gradient tensors
        """
        if self.model is None:
            raise RuntimeError("Model not set")

        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()

        return gradients

    def compute_gradients_on_batch(self, data: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute gradients on a single batch without updating parameters.

        Args:
            data: Input data
            target: Target labels

        Returns:
            Dictionary of gradients
        """
        if self.model is None or self.criterion is None:
            raise RuntimeError("Model or criterion not set")

        self.model.train()
        self.model.zero_grad()

        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()

        return self.get_gradients()

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or self.criterion is None:
            raise RuntimeError("Model or criterion not set")
        if self.val_loader is None:
            return {"loss": 0.0, "accuracy": 0.0, "n_samples": 0}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "n_samples": total_samples
        }


class BaseFLTrainer(ABC):
    """
    Abstract base class for federated learning trainers.

    Provides common FL training infrastructure with gradient diversity tracking.
    Subclasses should implement dataset-specific logic.
    """

    def __init__(
        self,
        config: FLConfig,
        global_model: nn.Module,
        clients: List[BaseClient],
        global_val_loader: Optional[DataLoader] = None,
        device: str = "cpu"
    ):
        """
        Args:
            config: FL configuration
            global_model: Initial global model
            clients: List of client instances
            global_val_loader: Optional global validation set
            device: Device for training
        """
        self.config = config
        self.global_model = global_model.to(device)
        self.clients = clients
        self.global_val_loader = global_val_loader
        self.device = device

        # Set seeds for reproducibility
        self._set_seeds(config.seed)

        # Initialize gradient diversity tracking
        if config.track_grad_diversity:
            self.grad_metric = GradientDiversityMetric(
                metric_type=config.grad_metric_type,
                normalization=config.grad_normalization,
                layer_filter=config.grad_layer_filter,
                device=device
            )
        else:
            self.grad_metric = None

        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "grad_diversity": [] if config.track_grad_diversity else None,
            "rounds": []
        }

        # Create log directory
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def create_criterion(self) -> nn.Module:
        """Create loss criterion. Subclasses must implement."""
        pass

    def sample_clients(self, round_num: int) -> List[BaseClient]:
        """
        Sample clients for this round.

        Args:
            round_num: Current round number

        Returns:
            List of sampled clients
        """
        np.random.seed(self.config.seed + round_num)
        sampled = np.random.choice(
            self.clients,
            size=min(self.config.clients_per_round, len(self.clients)),
            replace=False
        )
        return sampled.tolist()

    def broadcast_model(self, clients: List[BaseClient]):
        """Broadcast global model to selected clients."""
        for client in clients:
            client.model.load_state_dict(self.global_model.state_dict())

    def aggregate_gradients(self, client_gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client gradients (FedAvg).

        Args:
            client_gradients: List of gradient dictionaries from clients

        Returns:
            Aggregated gradients
        """
        aggregated = {}
        n_clients = len(client_gradients)

        for name in client_gradients[0].keys():
            # Simple average (can be weighted by dataset size)
            aggregated[name] = torch.stack([
                grads[name] for grads in client_gradients
            ]).mean(dim=0)

        return aggregated

    def apply_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Apply aggregated gradients to global model."""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in gradients:
                    param.grad = gradients[name]

        # Create optimizer and take step
        optimizer = self._create_global_optimizer()
        optimizer.step()
        optimizer.zero_grad()

    def _create_global_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for global model."""
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.global_model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.global_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def train_round(self, round_num: int) -> Dict[str, Any]:
        """
        Execute one round of federated learning.

        Args:
            round_num: Current round number

        Returns:
            Dictionary with round metrics
        """
        # 1. Sample clients
        selected_clients = self.sample_clients(round_num)

        # 2. Broadcast model
        self.broadcast_model(selected_clients)

        # 3. Local training and gradient collection
        client_gradients = []
        train_metrics = []

        criterion = self.create_criterion()
        for client in selected_clients:
            client.set_criterion(criterion)
            client.set_optimizer(self._create_client_optimizer(client.model))

            # Train locally
            metrics = client.train_local(self.config.local_epochs)
            train_metrics.append(metrics)

            # Collect gradients after training
            gradients = client.get_gradients()
            client_gradients.append(gradients)

        # 4. Compute gradient diversity BEFORE aggregation
        grad_diversity = None
        if self.config.track_grad_diversity and len(client_gradients) >= 2:
            grad_diversity = self.grad_metric.compute(client_gradients)

        # 5. Aggregate gradients
        aggregated = self.aggregate_gradients(client_gradients)

        # 6. Update global model
        self.apply_gradients(aggregated)

        # 7. Compute round statistics
        avg_train_loss = np.mean([m["loss"] for m in train_metrics])
        avg_train_acc = np.mean([m["accuracy"] for m in train_metrics])

        round_metrics = {
            "round": round_num,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "n_clients": len(selected_clients),
            "grad_diversity": grad_diversity
        }

        return round_metrics

    def _create_client_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer for client model."""
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def evaluate_global(self) -> Dict[str, float]:
        """Evaluate global model on validation set."""
        if self.global_val_loader is None:
            return {"loss": 0.0, "accuracy": 0.0}

        self.global_model.eval()
        criterion = self.create_criterion()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.global_val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples
        }

    def train(self):
        """Main training loop."""
        print(f"Starting FL training with {len(self.clients)} clients for {self.config.n_rounds} rounds")
        print(f"Gradient diversity tracking: {self.config.track_grad_diversity}")

        for round_num in range(self.config.n_rounds):
            # Training round
            round_metrics = self.train_round(round_num)

            # Evaluation
            if round_num % self.config.eval_every == 0:
                val_metrics = self.evaluate_global()
                round_metrics.update({
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"]
                })

            # Log metrics
            self._log_round(round_metrics)

            # Print progress
            if round_num % self.config.eval_every == 0:
                self._print_progress(round_metrics)

            # Save checkpoint
            if self.config.save_checkpoints and round_num % self.config.checkpoint_every == 0:
                self.save_checkpoint(round_num)

        print("Training complete!")
        self.save_results()

    def _log_round(self, metrics: Dict[str, Any]):
        """Log round metrics to history."""
        self.history["rounds"].append(metrics["round"])
        self.history["train_loss"].append(metrics["train_loss"])
        self.history["train_accuracy"].append(metrics["train_accuracy"])

        if "val_loss" in metrics:
            self.history["val_loss"].append(metrics["val_loss"])
            self.history["val_accuracy"].append(metrics["val_accuracy"])

        if self.config.track_grad_diversity and metrics["grad_diversity"] is not None:
            self.history["grad_diversity"].append(metrics["grad_diversity"])

    def _print_progress(self, metrics: Dict[str, Any]):
        """Print training progress."""
        msg = f"Round {metrics['round']:3d} | "
        msg += f"Train Loss: {metrics['train_loss']:.4f} | "
        msg += f"Train Acc: {metrics['train_accuracy']:.4f}"

        if "val_loss" in metrics:
            msg += f" | Val Loss: {metrics['val_loss']:.4f}"
            msg += f" | Val Acc: {metrics['val_accuracy']:.4f}"

        if metrics.get("grad_diversity") is not None:
            msg += f" | Grad Div: {metrics['grad_diversity']:.4f}"

        print(msg)

    def save_checkpoint(self, round_num: int):
        """Save model checkpoint."""
        checkpoint_path = self.log_dir / f"checkpoint_round_{round_num}.pt"
        torch.save({
            "round": round_num,
            "model_state_dict": self.global_model.state_dict(),
            "config": self.config.to_dict()
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_results(self):
        """Save training history and final model."""
        # Save history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Save final model
        model_path = self.log_dir / "final_model.pt"
        torch.save(self.global_model.state_dict(), model_path)

        # Save config
        config_path = self.log_dir / "config.json"
        self.config.save(str(config_path))

        print(f"Results saved to {self.log_dir}")

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves including gradient diversity."""
        n_plots = 2 if self.config.track_grad_diversity else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

        if n_plots == 1:
            axes = [axes]

        # Plot 1: Loss and accuracy
        ax1 = axes[0]
        ax1_twin = ax1.twinx()

        rounds = self.history["rounds"]
        ax1.plot(rounds, self.history["train_loss"], 'b-', label='Train Loss')
        if self.history["val_loss"]:
            ax1.plot(rounds, self.history["val_loss"], 'b--', label='Val Loss')

        ax1_twin.plot(rounds, self.history["train_accuracy"], 'r-', label='Train Acc')
        if self.history["val_accuracy"]:
            ax1_twin.plot(rounds, self.history["val_accuracy"], 'r--', label='Val Acc')

        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss', color='b')
        ax1_twin.set_ylabel('Accuracy', color='r')
        ax1.set_title('Training Progress')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gradient diversity
        if self.config.track_grad_diversity and self.history["grad_diversity"]:
            ax2 = axes[1]
            ax2.plot(rounds, self.history["grad_diversity"], 'g-', linewidth=2)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Gradient Diversity')
            ax2.set_title(f'Gradient Diversity Evolution ({self.config.grad_metric_type})')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.log_dir / "training_curves.png", dpi=150, bbox_inches='tight')

        plt.show()
