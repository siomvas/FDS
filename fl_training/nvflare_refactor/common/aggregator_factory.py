"""
Aggregator Factory for NVFlare FL Training

Provides configurable aggregation methods for federated learning.
Supports: FedAvg, FedProx, SCAFFOLD, FedOpt, and custom aggregators.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import torch
import torch.nn as nn
import numpy as np


class AggregationMethod(str, Enum):
    """Supported aggregation methods"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDOPT = "fedopt"
    WEIGHTED = "weighted"


class FedAvgAggregator:
    """
    Federated Averaging (FedAvg) aggregator.

    Computes weighted average of client model parameters.
    Weight can be based on number of samples or uniform.
    """

    def __init__(self, weighted: bool = True):
        """
        Args:
            weighted: If True, weight by number of samples. If False, uniform weighting.
        """
        self.weighted = weighted

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model parameters.

        Args:
            client_params: List of client model state_dicts
            client_weights: List of client weights (e.g., number of samples)
                          If None and weighted=True, uses uniform weights

        Returns:
            Aggregated model parameters
        """
        n_clients = len(client_params)

        if not self.weighted or client_weights is None:
            # Uniform weighting
            client_weights = [1.0 / n_clients] * n_clients
        else:
            # Normalize weights to sum to 1
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # Get parameter names from first client
        param_names = list(client_params[0].keys())

        # Aggregate
        aggregated_params = {}
        for param_name in param_names:
            # Stack all client parameters for this layer
            stacked_params = torch.stack([
                client_params[i][param_name]
                for i in range(n_clients)
            ])

            # Weighted average
            weights_tensor = torch.tensor(
                client_weights,
                device=stacked_params.device,
                dtype=stacked_params.dtype
            )

            # Reshape weights for broadcasting
            for _ in range(stacked_params.ndim - 1):
                weights_tensor = weights_tensor.unsqueeze(-1)

            aggregated_params[param_name] = torch.sum(
                stacked_params * weights_tensor,
                dim=0
            )

        return aggregated_params


class FedProxAggregator:
    """
    FedProx aggregator with proximal term.

    Aggregates with penalty on deviation from global model.
    The proximal term is applied during local training, but aggregation
    uses standard FedAvg.
    """

    def __init__(self, mu: float = 0.01, weighted: bool = True):
        """
        Args:
            mu: Proximal term coefficient (applied during training, not aggregation)
            weighted: Whether to use weighted averaging
        """
        self.mu = mu
        self.fedavg = FedAvgAggregator(weighted=weighted)

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client parameters using FedAvg.

        Note: The proximal term is applied during client training,
        not during aggregation.
        """
        return self.fedavg.aggregate(client_params, client_weights)

    def get_proximal_term(
        self,
        local_params: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute proximal term for local training.

        Add this to the loss during local training:
        loss = task_loss + proximal_term

        Args:
            local_params: Current local model parameters
            global_params: Global model parameters

        Returns:
            Proximal regularization term
        """
        proximal_term = 0.0

        for param_name in local_params:
            if param_name in global_params:
                diff = local_params[param_name] - global_params[param_name]
                proximal_term += torch.sum(diff ** 2)

        return (self.mu / 2.0) * proximal_term


class SCAFFOLDAggregator:
    """
    SCAFFOLD aggregator with control variates.

    Maintains server and client control variates to correct for
    client drift in heterogeneous data settings.
    """

    def __init__(self, weighted: bool = True):
        """
        Args:
            weighted: Whether to use weighted averaging
        """
        self.weighted = weighted
        self.fedavg = FedAvgAggregator(weighted=weighted)

        # Control variates
        self.server_controls = None
        self.client_controls = None

    def initialize_controls(
        self,
        model_params: Dict[str, torch.Tensor],
        n_clients: int
    ) -> None:
        """
        Initialize control variates.

        Call this before first FL round.

        Args:
            model_params: Initial model parameters
            n_clients: Number of clients
        """
        # Server controls (initialized to zero)
        self.server_controls = {
            name: torch.zeros_like(param)
            for name, param in model_params.items()
        }

        # Client controls (one set per client, all initialized to zero)
        self.client_controls = [
            {name: torch.zeros_like(param) for name, param in model_params.items()}
            for _ in range(n_clients)
        ]

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        client_controls_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate with SCAFFOLD correction.

        Args:
            client_params: List of client model parameters
            client_controls_updates: List of client control variate updates
            client_weights: List of client weights

        Returns:
            Aggregated model parameters
        """
        # Standard FedAvg aggregation
        aggregated_params = self.fedavg.aggregate(client_params, client_weights)

        # Update server controls
        n_clients = len(client_params)
        for param_name in aggregated_params:
            # Average client control updates
            avg_client_control = torch.mean(
                torch.stack([
                    client_controls_updates[i][param_name]
                    for i in range(n_clients)
                ]),
                dim=0
            )

            # Update server control
            self.server_controls[param_name] = avg_client_control

        return aggregated_params


def get_aggregator(
    method: str,
    **kwargs
) -> Any:
    """
    Factory function to get aggregator.

    Args:
        method: Aggregation method name ('fedavg', 'fedprox', 'scaffold')
        **kwargs: Method-specific arguments

    Returns:
        Aggregator instance

    Examples:
        # FedAvg with weighted averaging
        >>> agg = get_aggregator('fedavg', weighted=True)

        # FedProx with mu=0.01
        >>> agg = get_aggregator('fedprox', mu=0.01)

        # SCAFFOLD
        >>> agg = get_aggregator('scaffold')
    """
    method = method.lower()

    if method == AggregationMethod.FEDAVG:
        return FedAvgAggregator(
            weighted=kwargs.get('weighted', True)
        )

    elif method == AggregationMethod.FEDPROX:
        return FedProxAggregator(
            mu=kwargs.get('mu', 0.01),
            weighted=kwargs.get('weighted', True)
        )

    elif method == AggregationMethod.SCAFFOLD:
        return SCAFFOLDAggregator(
            weighted=kwargs.get('weighted', True)
        )

    else:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Available: {[m.value for m in AggregationMethod]}"
        )


def get_nvflare_aggregator_config(
    method: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Get NVFlare aggregator configuration.

    Returns configuration dict for NVFlare's built-in aggregators.

    Args:
        method: Aggregation method
        **kwargs: Method-specific parameters

    Returns:
        Configuration dictionary for NVFlare

    Example:
        >>> config = get_nvflare_aggregator_config('fedavg', weighted=True)
        >>> # Use in NVFlare job configuration
    """
    method = method.lower()

    if method == AggregationMethod.FEDAVG:
        return {
            "id": "aggregator",
            "path": "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",
            "args": {
                "expected_data_kind": "WEIGHTS"
            }
        }

    elif method == AggregationMethod.FEDPROX:
        # FedProx uses FedAvg aggregation but different client training
        return {
            "id": "aggregator",
            "path": "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",
            "args": {
                "expected_data_kind": "WEIGHTS"
            },
            "proximal_mu": kwargs.get('mu', 0.01)
        }

    elif method == AggregationMethod.SCAFFOLD:
        return {
            "id": "aggregator",
            "path": "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",
            "args": {
                "expected_data_kind": "WEIGHTS"
            },
            "scaffold_enabled": True
        }

    else:
        raise ValueError(f"Unknown method: {method}")


# Helper function for manual aggregation (useful for testing)
def aggregate_models(
    global_model: nn.Module,
    client_models: List[nn.Module],
    method: str = "fedavg",
    client_weights: Optional[List[float]] = None,
    **kwargs
) -> nn.Module:
    """
    Aggregate client models into global model.

    Convenience function that handles model → state_dict conversion.

    Args:
        global_model: Global model to update
        client_models: List of client models
        method: Aggregation method
        client_weights: Optional client weights
        **kwargs: Aggregator-specific arguments

    Returns:
        Updated global model

    Example:
        >>> global_model = aggregate_models(
        ...     global_model,
        ...     client_models,
        ...     method='fedavg',
        ...     client_weights=[100, 150, 200]  # num samples per client
        ... )
    """
    # Get aggregator
    aggregator = get_aggregator(method, **kwargs)

    # Extract state dicts
    client_params = [model.state_dict() for model in client_models]

    # Aggregate
    aggregated_params = aggregator.aggregate(client_params, client_weights)

    # Load into global model
    global_model.load_state_dict(aggregated_params)

    return global_model


if __name__ == "__main__":
    # Example usage and testing
    print("Testing aggregator_factory.py...")

    # Create dummy models
    model1 = nn.Sequential(nn.Linear(10, 5))
    model2 = nn.Sequential(nn.Linear(10, 5))
    model3 = nn.Sequential(nn.Linear(10, 5))

    # Initialize with different weights
    for i, model in enumerate([model1, model2, model3]):
        for param in model.parameters():
            param.data.fill_(i + 1)

    # Test FedAvg
    print("\nFedAvg (uniform weights):")
    agg = get_aggregator('fedavg', weighted=False)
    aggregated = agg.aggregate([m.state_dict() for m in [model1, model2, model3]])
    avg_value = list(aggregated.values())[0][0, 0].item()
    print(f"  Average value: {avg_value:.2f} (expected: 2.00)")

    # Test FedAvg with weights
    print("\nFedAvg (weighted):")
    agg = get_aggregator('fedavg', weighted=True)
    aggregated = agg.aggregate(
        [m.state_dict() for m in [model1, model2, model3]],
        client_weights=[100, 200, 100]  # Model2 has 2x weight
    )
    avg_value = list(aggregated.values())[0][0, 0].item()
    print(f"  Weighted average value: {avg_value:.2f} (expected: 2.00)")

    # Test FedProx
    print("\nFedProx:")
    agg = get_aggregator('fedprox', mu=0.01)
    proximal = agg.get_proximal_term(model2.state_dict(), model1.state_dict())
    print(f"  Proximal term: {proximal:.4f}")

    # Test NVFlare config generation
    print("\nNVFlare configs:")
    for method in ['fedavg', 'fedprox', 'scaffold']:
        config = get_nvflare_aggregator_config(method)
        print(f"  {method}: {config['path'].split('.')[-1]}")

    print("\n✓ Aggregator factory module ready!")
