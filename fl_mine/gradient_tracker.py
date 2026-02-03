"""
Gradient Diversity Tracking for NVFlare FL Training

Integrates with existing gradient_metrics.py to provide:
- Client-side: Local gradient statistics (norms, variance)
- Server-side: Cross-client diversity metrics (cosine, variance, conflict)

Both tracked automatically and logged to TensorBoard.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from evaluation.gradient_metrics import GradientDiversityMetric


class ClientGradientTracker:
    """
    Tracks local gradient statistics on client side.

    Computes gradient norms, variance, and weight divergence after local training.
    These metrics are sent back to server along with model updates.
    """

    def __init__(
        self,
        track_per_layer: bool = False,
        track_weight_divergence: bool = True
    ):
        """
        Args:
            track_per_layer: If True, compute metrics per layer (more detailed but slower)
            track_weight_divergence: Track how much weights changed from initial
        """
        self.track_per_layer = track_per_layer
        self.track_weight_divergence = track_weight_divergence
        self.initial_weights = None
        self.epoch_history: List[Dict[str, float]] = []  # For track_epoch accumulation

    def store_initial_weights(self, model: nn.Module) -> None:
        """
        Store initial model weights before local training.

        Call this after receiving global model from server.

        Args:
            model: PyTorch model
        """
        self.initial_weights = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def compute_gradient_statistics(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute gradient statistics after local training.

        Args:
            model: PyTorch model with gradients

        Returns:
            Dictionary of gradient metrics
        """
        gradient_norms = []
        gradient_vars = []
        layer_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data

                # Gradient norm
                grad_norm = torch.norm(grad).item()
                gradient_norms.append(grad_norm)

                # Gradient variance
                grad_var = torch.var(grad).item()
                gradient_vars.append(grad_var)

                # Per-layer stats if requested
                if self.track_per_layer:
                    layer_stats[name] = {
                        "norm": grad_norm,
                        "var": grad_var,
                        "mean": torch.mean(torch.abs(grad)).item()
                    }

        metrics = {
            "gradient_norm_mean": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "gradient_norm_std": float(np.std(gradient_norms)) if gradient_norms else 0.0,
            "gradient_norm_max": float(np.max(gradient_norms)) if gradient_norms else 0.0,
            "gradient_var_mean": float(np.mean(gradient_vars)) if gradient_vars else 0.0,
        }

        # Add per-layer stats if requested
        if self.track_per_layer:
            metrics["per_layer_stats"] = layer_stats

        return metrics

    def compute_weight_divergence(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute how much weights diverged from initial values.

        Args:
            model: PyTorch model after local training

        Returns:
            Dictionary of divergence metrics
        """
        if self.initial_weights is None:
            return {"weight_divergence": 0.0}

        divergences = []
        relative_divergences = []

        current_weights = {
            name: param.data
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for name in self.initial_weights:
            if name in current_weights:
                initial = self.initial_weights[name]
                current = current_weights[name]

                # Absolute divergence
                diff = torch.norm(current - initial).item()
                divergences.append(diff)

                # Relative divergence
                initial_norm = torch.norm(initial).item()
                if initial_norm > 1e-8:
                    rel_diff = diff / initial_norm
                    relative_divergences.append(rel_diff)

        return {
            "weight_divergence_mean": float(np.mean(divergences)) if divergences else 0.0,
            "weight_divergence_max": float(np.max(divergences)) if divergences else 0.0,
            "relative_divergence_mean": float(np.mean(relative_divergences)) if relative_divergences else 0.0,
        }

    def compute_all_metrics(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute all client-side gradient metrics.

        Call this after local training, before sending update to server.

        Args:
            model: PyTorch model

        Returns:
            Complete dictionary of client gradient metrics
        """
        metrics = self.compute_gradient_statistics(model)

        if self.track_weight_divergence and self.initial_weights is not None:
            divergence_metrics = self.compute_weight_divergence(model)
            metrics.update(divergence_metrics)

        return metrics

    def track_epoch(self, model: nn.Module, epoch: int) -> None:
        """
        Track gradient statistics for a single epoch.

        Call this after each training epoch to accumulate metrics across the round.

        Args:
            model: PyTorch model with gradients computed
            epoch: Current epoch number (for logging purposes)
        """
        # Compute metrics for this epoch
        epoch_metrics = self.compute_all_metrics(model)

        # Store in history
        self.epoch_history.append(epoch_metrics)

    def get_round_statistics(self) -> Dict[str, float]:
        """
        Aggregate metrics across all epochs in the current round.

        Returns aggregated statistics (mean, max, min) and clears epoch history
        for the next round.

        Returns:
            Dictionary of aggregated gradient metrics
        """
        if not self.epoch_history:
            return {}

        # Collect all metric keys from first epoch (assumes consistent keys)
        all_keys = set()
        for epoch_metrics in self.epoch_history:
            all_keys.update(epoch_metrics.keys())

        # Remove per_layer_stats if present (not aggregatable)
        all_keys.discard("per_layer_stats")

        # Aggregate metrics
        aggregated = {}

        for key in all_keys:
            # Extract values for this key across all epochs
            values = [
                epoch_metrics[key]
                for epoch_metrics in self.epoch_history
                if key in epoch_metrics and not isinstance(epoch_metrics[key], dict)
            ]

            if values:
                # Compute statistics
                aggregated[f"{key}_avg"] = float(np.mean(values))
                aggregated[f"{key}_max"] = float(np.max(values))
                aggregated[f"{key}_min"] = float(np.min(values))

        # Clear history for next round
        self.epoch_history = []

        return aggregated


class ServerGradientTracker:
    """
    Tracks cross-client gradient diversity on server side.

    Computes diversity metrics across all client updates before aggregation.
    Uses existing GradientDiversityMetric from evaluation/gradient_metrics.py.
    """

    def __init__(
        self,
        metric_types: List[str] = None,
        normalization: str = "layer_wise",
        aggregation_method: str = "weighted_mean",
        layer_filter: Optional[List[str]] = None,
    ):
        """
        Args:
            metric_types: List of metrics to compute
                         Options: 'cosine', 'variance', 'conflict', 'l2', 'spectral'
                         Default: ['cosine', 'variance', 'conflict']
            normalization: Normalization strategy
                          Options: 'none', 'l2', 'layer_wise', 'std'
            aggregation_method: How to aggregate across layers
                               Options: 'mean', 'weighted_mean', 'max'
            layer_filter: List of layer name patterns to include (None = all layers)
        """
        if metric_types is None:
            metric_types = ['cosine', 'variance', 'conflict']

        self.metric_types = metric_types
        self.normalization = normalization
        self.aggregation_method = aggregation_method
        self.layer_filter = layer_filter

        # Create metric computers
        self.metrics = {
            metric_type: GradientDiversityMetric(
                metric_type=metric_type,
                normalization=normalization,
                aggregation_method=aggregation_method,
                layer_filter=layer_filter
            )
            for metric_type in metric_types
        }

    def extract_gradients_from_updates(
        self,
        global_model: nn.Module,
        client_models: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract gradient-like updates from client model parameters.

        Args:
            global_model: Global model before client updates
            client_models: List of client model state dicts

        Returns:
            List of gradient dictionaries (one per client)
        """
        global_params = {
            name: param.data
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }

        client_gradients = []

        for client_params in client_models:
            gradients = {}
            for name in global_params:
                if name in client_params:
                    # Gradient = Global - Client (for additive updates)
                    grad = global_params[name] - client_params[name]
                    gradients[name] = grad
            client_gradients.append(gradients)

        return client_gradients

    def compute_diversity_metrics(
        self,
        client_gradients: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute all diversity metrics across client gradients.

        Args:
            client_gradients: List of gradient dictionaries (one per client)

        Returns:
            Dictionary of diversity metrics
        """
        diversity_metrics = {}

        for metric_type, metric_computer in self.metrics.items():
            try:
                diversity_score = metric_computer.compute(client_gradients)
                diversity_metrics[f"{metric_type}_diversity"] = float(diversity_score)
            except Exception as e:
                print(f"Warning: Failed to compute {metric_type} diversity: {e}")
                diversity_metrics[f"{metric_type}_diversity"] = 0.0

        # Also compute number of clients
        diversity_metrics["num_clients_contributing"] = len(client_gradients)

        return diversity_metrics

    def compute_pairwise_similarities(
        self,
        client_gradients: List[Dict[str, torch.Tensor]]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarities between client gradients.

        Useful for analyzing client clustering and heterogeneity patterns.

        Args:
            client_gradients: List of gradient dictionaries

        Returns:
            NxN similarity matrix where N = number of clients
        """
        if "cosine" in self.metrics:
            metric = self.metrics["cosine"]
            similarity_matrix = metric.compute_pairwise_similarity(client_gradients)
            return similarity_matrix
        else:
            raise ValueError("Cosine metric not initialized")


def create_gradient_tracker(
    config: Dict[str, Any],
    side: str = "client"
) -> Any:
    """
    Factory function to create appropriate gradient tracker.

    Args:
        config: Configuration dictionary
        side: 'client' or 'server'

    Returns:
        ClientGradientTracker or ServerGradientTracker
    """
    if side == "client":
        return ClientGradientTracker(
            track_per_layer=config.get("track_per_layer", False),
            track_weight_divergence=config.get("track_weight_divergence", True)
        )
    elif side == "server":
        return ServerGradientTracker(
            metric_types=config.get("metric_types", ['cosine', 'variance', 'conflict']),
            normalization=config.get("normalization", "layer_wise"),
            aggregation_method=config.get("aggregation_method", "weighted_mean"),
            layer_filter=config.get("layer_filter", None)
        )
    else:
        raise ValueError(f"Unknown side '{side}'. Must be 'client' or 'server'")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing gradient_tracker.py...")

    # Create a dummy model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    # Client-side tracking
    print("\n Client-side tracking:")
    client_tracker = ClientGradientTracker()

    # Store initial weights
    client_tracker.store_initial_weights(model)

    # Simulate training (create fake gradients)
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.01

    # Compute metrics
    client_metrics = client_tracker.compute_all_metrics(model)
    for key, value in client_metrics.items():
        print(f"  {key}: {value:.6f}")

    # Server-side tracking
    print("\nServer-side tracking:")
    server_tracker = ServerGradientTracker()

    # Simulate multiple clients
    client_grads = []
    for i in range(3):
        grads = {
            name: torch.randn_like(param) * (0.01 * (i + 1))
            for name, param in model.named_parameters()
        }
        client_grads.append(grads)

    # Compute diversity
    server_metrics = server_tracker.compute_diversity_metrics(client_grads)
    for key, value in server_metrics.items():
        print(f"  {key}: {value:.6f}")

    print("\n✓ Gradient tracker module ready!")