"""
Gradient diversity metrics for federated learning.

Measures gradient dissimilarity across clients during FL training to quantify
dynamic heterogeneity that complements static data distribution metrics.

Usage:
    from evaluation.gradient_metrics import GradientDiversityMetric

    # Initialize metric
    metric = GradientDiversityMetric(
        metric_type="cosine",
        normalization="layer_wise",
        aggregation="weighted_mean"
    )

    # Compute at each FL round
    client_gradients = [client.get_gradients() for client in clients]
    diversity = metric.compute(client_gradients)
"""

import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Literal, Union
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
import warnings


# ============================================================================
# Helper Functions
# ============================================================================

def flatten_gradients(
    gradients: Dict[str, Tensor],
    layer_filter: Optional[List[str]] = None,
    device: str = "cpu"
) -> Tensor:
    """
    Flatten gradient dictionary into a single vector.

    Args:
        gradients: Dictionary mapping parameter names to gradient tensors
        layer_filter: If provided, only include gradients from these layers
        device: Device to place flattened tensor on

    Returns:
        1D tensor containing all gradient values
    """
    grad_list = []

    for name, grad in gradients.items():
        # Apply layer filter if specified
        if layer_filter is not None and not any(layer in name for layer in layer_filter):
            continue

        if grad is not None:
            grad_list.append(grad.flatten().to(device))

    if len(grad_list) == 0:
        raise ValueError("No gradients found. Check layer_filter or gradient dict.")

    return torch.cat(grad_list)


def normalize_gradient(
    grad: Tensor,
    method: Literal["none", "l2", "std"] = "l2"
) -> Tensor:
    """
    Normalize a gradient tensor.

    Args:
        grad: Gradient tensor to normalize
        method: Normalization method
            - "none": No normalization
            - "l2": L2 normalization (unit vector)
            - "std": Standardization (zero mean, unit variance)

    Returns:
        Normalized gradient tensor
    """
    if method == "none":
        return grad
    elif method == "l2":
        norm = torch.norm(grad)
        if norm > 1e-10:
            return grad / norm
        return grad
    elif method == "std":
        mean = grad.mean()
        std = grad.std()
        if std > 1e-10:
            return (grad - mean) / std
        return grad - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_pairwise_cosine_similarity(
    gradients: List[Tensor],
    normalize: bool = True
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between gradients.

    Args:
        gradients: List of flattened gradient tensors
        normalize: Whether to L2-normalize before computing similarity

    Returns:
        C x C matrix where entry (i,j) is cosine similarity between
        gradients i and j
    """
    n_clients = len(gradients)

    # Stack gradients into matrix
    grad_matrix = torch.stack(gradients)  # (C, P)

    # L2 normalize if requested
    if normalize:
        grad_matrix = torch.nn.functional.normalize(grad_matrix, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(grad_matrix, grad_matrix.t())

    return similarity_matrix.cpu().numpy()


def compute_pairwise_l2_distance(gradients: List[Tensor]) -> np.ndarray:
    """
    Compute pairwise L2 distance matrix between gradients.

    Args:
        gradients: List of flattened gradient tensors

    Returns:
        C x C matrix of pairwise L2 distances
    """
    n_clients = len(gradients)
    grad_matrix = torch.stack(gradients).cpu().numpy()

    # Compute pairwise distances
    distances = pdist(grad_matrix, metric='euclidean')
    distance_matrix = squareform(distances)

    return distance_matrix


# ============================================================================
# Main Gradient Diversity Metric Class
# ============================================================================

class GradientDiversityMetric:
    """
    Compute gradient diversity metrics during federated learning training.

    Supports multiple diversity measures that quantify how dissimilar client
    gradients are at each FL round.

    Attributes:
        metric_type: Type of diversity metric to compute
        normalization: Gradient normalization strategy
        aggregation: How to aggregate across layers
        layer_filter: Optional list of layer names to include
        sample_params: If > 0, randomly sample this fraction of parameters
    """

    VALID_METRICS = ["cosine", "variance", "conflict", "l2", "spectral"]
    VALID_NORMALIZATIONS = ["none", "l2", "layer_wise", "std"]
    VALID_AGGREGATIONS = ["mean", "weighted_mean", "max"]

    def __init__(
        self,
        metric_type: Literal["cosine", "variance", "conflict", "l2", "spectral"] = "cosine",
        normalization: Literal["none", "l2", "layer_wise", "std"] = "layer_wise",
        aggregation: Literal["mean", "weighted_mean", "max"] = "weighted_mean",
        layer_filter: Optional[List[str]] = None,
        sample_params: float = 0.0,
        device: str = "cpu",
    ):
        """
        Initialize gradient diversity metric.

        Args:
            metric_type: Type of diversity metric
                - "cosine": 1 - mean pairwise cosine similarity
                - "variance": Variance of gradients across clients
                - "conflict": Fraction of client pairs with opposing gradients
                - "l2": Mean pairwise L2 distance
                - "spectral": Spectral diversity (condition number)
            normalization: How to normalize gradients
                - "none": No normalization
                - "l2": L2 normalize (unit vectors)
                - "layer_wise": L2 normalize within each layer
                - "std": Standardize (zero mean, unit std)
            aggregation: How to aggregate layer-wise metrics
                - "mean": Simple average
                - "weighted_mean": Weight by layer size
                - "max": Maximum across layers
            layer_filter: If provided, only compute on these layers
            sample_params: Fraction of parameters to sample (0-1). Use for efficiency.
            device: Device for computation
        """
        if metric_type not in self.VALID_METRICS:
            raise ValueError(f"metric_type must be one of {self.VALID_METRICS}")
        if normalization not in self.VALID_NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {self.VALID_NORMALIZATIONS}")
        if aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {self.VALID_AGGREGATIONS}")
        if not 0.0 <= sample_params <= 1.0:
            raise ValueError("sample_params must be between 0 and 1")

        self.metric_type = metric_type
        self.normalization = normalization
        self.aggregation = aggregation
        self.layer_filter = layer_filter
        self.sample_params = sample_params
        self.device = device

        # Cache for parameter sampling
        self._param_sample_indices: Optional[Dict[str, Tensor]] = None

    def compute(
        self,
        client_gradients: List[Dict[str, Tensor]],
        return_details: bool = False
    ) -> Union[float, Dict[str, any]]:
        """
        Compute gradient diversity for a set of client gradients.

        Args:
            client_gradients: List of gradient dictionaries, one per client
                Each dict maps parameter names to gradient tensors
            return_details: If True, return detailed breakdown

        Returns:
            Diversity score (float), or dict with details if return_details=True
        """
        if len(client_gradients) < 2:
            warnings.warn("Need at least 2 clients to compute gradient diversity")
            return 0.0 if not return_details else {"diversity": 0.0, "n_clients": len(client_gradients)}

        # Choose computation method based on normalization strategy
        if self.normalization == "layer_wise":
            diversity = self._compute_layer_wise(client_gradients)
        else:
            diversity = self._compute_global(client_gradients)

        if return_details:
            return {
                "diversity": diversity,
                "n_clients": len(client_gradients),
                "metric_type": self.metric_type,
                "normalization": self.normalization
            }

        return diversity

    def _compute_global(self, client_gradients: List[Dict[str, Tensor]]) -> float:
        """
        Compute diversity on globally flattened gradients.

        Used when normalization is "none", "l2", or "std".
        """
        # Flatten all client gradients
        flat_grads = []
        for grad_dict in client_gradients:
            flat = flatten_gradients(grad_dict, self.layer_filter, self.device)

            # Apply sampling if requested
            if self.sample_params > 0:
                if self._param_sample_indices is None:
                    n_params = len(flat)
                    n_sample = int(n_params * self.sample_params)
                    self._param_sample_indices = torch.randperm(n_params)[:n_sample]
                flat = flat[self._param_sample_indices]

            # Apply normalization
            flat = normalize_gradient(flat, method=self.normalization)
            flat_grads.append(flat)

        # Compute diversity metric
        return self._compute_diversity_from_tensors(flat_grads)

    def _compute_layer_wise(self, client_gradients: List[Dict[str, Tensor]]) -> float:
        """
        Compute diversity layer-by-layer, then aggregate.

        Used when normalization is "layer_wise".
        """
        # Get all layer names (from first client)
        layer_names = list(client_gradients[0].keys())
        if self.layer_filter:
            layer_names = [n for n in layer_names if any(f in n for f in self.layer_filter)]

        layer_diversities = []
        layer_weights = []

        for layer_name in layer_names:
            # Extract this layer's gradients from all clients
            layer_grads = []
            for grad_dict in client_gradients:
                if layer_name not in grad_dict or grad_dict[layer_name] is None:
                    continue

                grad = grad_dict[layer_name].flatten().to(self.device)

                # L2 normalize this layer
                grad = normalize_gradient(grad, method="l2")
                layer_grads.append(grad)

            if len(layer_grads) < 2:
                continue

            # Compute diversity for this layer
            layer_div = self._compute_diversity_from_tensors(layer_grads)
            layer_diversities.append(layer_div)

            # Weight by number of parameters
            layer_weights.append(len(layer_grads[0]))

        if len(layer_diversities) == 0:
            return 0.0

        # Aggregate across layers
        layer_diversities = np.array(layer_diversities)
        layer_weights = np.array(layer_weights)

        if self.aggregation == "mean":
            return float(np.mean(layer_diversities))
        elif self.aggregation == "weighted_mean":
            return float(np.average(layer_diversities, weights=layer_weights))
        elif self.aggregation == "max":
            return float(np.max(layer_diversities))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _compute_diversity_from_tensors(self, gradients: List[Tensor]) -> float:
        """
        Compute diversity metric from list of gradient tensors.

        Args:
            gradients: List of flattened, normalized gradient tensors

        Returns:
            Diversity score
        """
        if self.metric_type == "cosine":
            return self._compute_cosine_diversity(gradients)
        elif self.metric_type == "variance":
            return self._compute_variance_diversity(gradients)
        elif self.metric_type == "conflict":
            return self._compute_conflict_diversity(gradients)
        elif self.metric_type == "l2":
            return self._compute_l2_diversity(gradients)
        elif self.metric_type == "spectral":
            return self._compute_spectral_diversity(gradients)
        else:
            raise ValueError(f"Unknown metric type: {self.metric_type}")

    def _compute_cosine_diversity(self, gradients: List[Tensor]) -> float:
        """
        Compute 1 - mean pairwise cosine similarity.

        Returns value in [0, 2] where:
        - 0 = all gradients identical
        - 1 = orthogonal on average
        - 2 = all gradients opposite
        """
        sim_matrix = compute_pairwise_cosine_similarity(gradients, normalize=True)

        # Extract upper triangle (excluding diagonal)
        n = len(gradients)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_tri_indices]

        # Diversity = 1 - mean similarity
        mean_similarity = np.mean(pairwise_sims)
        diversity = 1.0 - mean_similarity

        return float(diversity)

    def _compute_variance_diversity(self, gradients: List[Tensor]) -> float:
        """
        Compute variance of gradients across clients.

        Returns the mean variance across all parameters.
        """
        grad_matrix = torch.stack(gradients)  # (C, P)
        variances = torch.var(grad_matrix, dim=0)  # (P,)
        mean_variance = torch.mean(variances)

        return float(mean_variance.item())

    def _compute_conflict_diversity(self, gradients: List[Tensor]) -> float:
        """
        Compute fraction of client pairs with conflicting gradient directions.

        Conflict occurs when cosine similarity < 0.
        Returns value in [0, 1].
        """
        sim_matrix = compute_pairwise_cosine_similarity(gradients, normalize=True)

        # Count negative similarities
        n = len(gradients)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_tri_indices]

        conflict_rate = np.mean(pairwise_sims < 0)

        return float(conflict_rate)

    def _compute_l2_diversity(self, gradients: List[Tensor]) -> float:
        """
        Compute mean pairwise L2 distance between gradients.
        """
        dist_matrix = compute_pairwise_l2_distance(gradients)

        # Extract upper triangle
        n = len(gradients)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_dists = dist_matrix[upper_tri_indices]

        mean_distance = np.mean(pairwise_dists)

        return float(mean_distance)

    def _compute_spectral_diversity(self, gradients: List[Tensor]) -> float:
        """
        Compute spectral diversity (condition number of gradient matrix).

        Returns (σ_max - σ_min) / σ_max where σ are singular values.
        Value in [0, 1] where 1 = maximum diversity.
        """
        grad_matrix = torch.stack(gradients).cpu().numpy()  # (C, P)

        # Compute SVD
        try:
            _, singular_values, _ = np.linalg.svd(grad_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            warnings.warn("SVD failed, returning 0 diversity")
            return 0.0

        if len(singular_values) < 2 or singular_values[0] < 1e-10:
            return 0.0

        # Compute diversity as (max - min) / max
        diversity = (singular_values[0] - singular_values[-1]) / singular_values[0]

        return float(diversity)

    def get_pairwise_similarity_matrix(
        self,
        client_gradients: List[Dict[str, Tensor]]
    ) -> np.ndarray:
        """
        Get pairwise similarity matrix for analysis.

        Args:
            client_gradients: List of gradient dictionaries

        Returns:
            C x C matrix of cosine similarities
        """
        if self.normalization == "layer_wise":
            # Average similarity across layers
            layer_names = list(client_gradients[0].keys())
            if self.layer_filter:
                layer_names = [n for n in layer_names if any(f in n for f in self.layer_filter)]

            layer_sim_matrices = []
            for layer_name in layer_names:
                layer_grads = []
                for grad_dict in client_gradients:
                    if layer_name in grad_dict and grad_dict[layer_name] is not None:
                        grad = grad_dict[layer_name].flatten().to(self.device)
                        grad = normalize_gradient(grad, method="l2")
                        layer_grads.append(grad)

                if len(layer_grads) >= 2:
                    sim_matrix = compute_pairwise_cosine_similarity(layer_grads, normalize=True)
                    layer_sim_matrices.append(sim_matrix)

            # Average across layers
            return np.mean(layer_sim_matrices, axis=0)

        else:
            # Global similarity
            flat_grads = []
            for grad_dict in client_gradients:
                flat = flatten_gradients(grad_dict, self.layer_filter, self.device)
                flat = normalize_gradient(flat, method=self.normalization)
                flat_grads.append(flat)

            return compute_pairwise_cosine_similarity(flat_grads, normalize=True)

    def compute_all_metrics(
        self,
        client_gradients: List[Dict[str, Tensor]]
    ) -> Dict[str, float]:
        """
        Compute all diversity metrics at once for comparison.

        Args:
            client_gradients: List of gradient dictionaries

        Returns:
            Dictionary mapping metric names to diversity scores
        """
        results = {}

        for metric_type in self.VALID_METRICS:
            temp_metric = GradientDiversityMetric(
                metric_type=metric_type,
                normalization=self.normalization,
                aggregation=self.aggregation,
                layer_filter=self.layer_filter,
                device=self.device
            )
            results[metric_type] = temp_metric.compute(client_gradients)

        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def compute_gradient_diversity(
    client_gradients: List[Dict[str, Tensor]],
    metric_type: str = "cosine",
    **kwargs
) -> float:
    """
    Convenience function to compute gradient diversity.

    Args:
        client_gradients: List of gradient dictionaries
        metric_type: Type of diversity metric
        **kwargs: Additional arguments for GradientDiversityMetric

    Returns:
        Diversity score
    """
    metric = GradientDiversityMetric(metric_type=metric_type, **kwargs)
    return metric.compute(client_gradients)


def analyze_gradient_diversity(
    client_gradients: List[Dict[str, Tensor]],
    normalize: bool = True
) -> Dict[str, any]:
    """
    Comprehensive gradient diversity analysis.

    Args:
        client_gradients: List of gradient dictionaries
        normalize: Whether to use layer-wise normalization

    Returns:
        Dictionary containing:
        - All diversity metrics
        - Pairwise similarity matrix
        - Summary statistics
    """
    normalization = "layer_wise" if normalize else "none"

    metric = GradientDiversityMetric(
        metric_type="cosine",
        normalization=normalization
    )

    # Compute all metrics
    all_metrics = metric.compute_all_metrics(client_gradients)

    # Get similarity matrix
    sim_matrix = metric.get_pairwise_similarity_matrix(client_gradients)

    # Compute summary stats
    upper_tri_indices = np.triu_indices(len(client_gradients), k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]

    return {
        "metrics": all_metrics,
        "similarity_matrix": sim_matrix,
        "pairwise_similarities": pairwise_sims,
        "mean_similarity": float(np.mean(pairwise_sims)),
        "std_similarity": float(np.std(pairwise_sims)),
        "min_similarity": float(np.min(pairwise_sims)),
        "max_similarity": float(np.max(pairwise_sims)),
        "n_clients": len(client_gradients)
    }
