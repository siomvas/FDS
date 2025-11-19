"""
Pooling strategies for aggregating token embeddings.

Provides various methods to aggregate token-level embeddings into
sample-level representations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


class PoolingStrategy(ABC):
    """Abstract base class for pooling strategies."""

    @abstractmethod
    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Pool token embeddings to sample-level embeddings.

        Args:
            embeddings: Token embeddings of shape:
                - (n_samples, embed_dim) for single token per sample
                - (n_samples, n_tokens, embed_dim) for multiple tokens

        Returns:
            Pooled embeddings of shape (n_samples, embed_dim)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CLSPooling(PoolingStrategy):
    """Extract CLS token only (first token)."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Extract CLS token (first token) from token embeddings."""
        if embeddings.ndim == 2:
            # Already single token per sample
            return embeddings
        elif embeddings.ndim == 3:
            # Extract first token (CLS)
            return embeddings[:, 0, :]
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")

    def __repr__(self) -> str:
        return "CLSPooling()"


class MeanPooling(PoolingStrategy):
    """Average all token embeddings."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Average over token dimension."""
        if embeddings.ndim == 2:
            # Already pooled
            return embeddings
        elif embeddings.ndim == 3:
            # Mean over tokens
            return np.mean(embeddings, axis=1)
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")

    def __repr__(self) -> str:
        return "MeanPooling()"


class MaxPooling(PoolingStrategy):
    """Max pooling over token embeddings."""

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Max pooling over token dimension."""
        if embeddings.ndim == 2:
            return embeddings
        elif embeddings.ndim == 3:
            return np.max(embeddings, axis=1)
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")

    def __repr__(self) -> str:
        return "MaxPooling()"


class GeMLPooling(PoolingStrategy):
    """
    Generalized Mean (GeM) Pooling.

    GeM pooling: (mean(x^p))^(1/p)
    - p=1: arithmetic mean
    - p=2: quadratic mean (RMS)
    - p→∞: max pooling
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        """
        Args:
            p: GeM pooling parameter (typically 3.0-4.0)
            eps: Small constant for numerical stability
        """
        self.p = p
        self.eps = eps

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply GeM pooling."""
        if embeddings.ndim == 2:
            return embeddings
        elif embeddings.ndim == 3:
            # GeM formula: (mean(x^p))^(1/p)
            # Add eps for numerical stability
            powered = np.power(np.abs(embeddings) + self.eps, self.p)
            mean_powered = np.mean(powered, axis=1)
            return np.sign(embeddings[:, 0, :]) * np.power(mean_powered, 1.0 / self.p)
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")

    def __repr__(self) -> str:
        return f"GeMLPooling(p={self.p})"


class LastKLayerPooling(PoolingStrategy):
    """
    Aggregate embeddings from last K layers.

    Can use mean, concatenation, or weighted average.
    """

    def __init__(self, k: int = 4, aggregation: str = "mean"):
        """
        Args:
            k: Number of last layers to aggregate
            aggregation: How to aggregate ("mean", "concat", "weighted")
        """
        self.k = k
        self.aggregation = aggregation

        if aggregation not in ["mean", "concat", "weighted"]:
            raise ValueError(f"Unknown aggregation: {aggregation}. Choose from mean/concat/weighted")

    def pool(self, layer_embeddings: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Pool embeddings from multiple layers.

        Args:
            layer_embeddings: Dict mapping layer index to embeddings (n_samples, embed_dim)

        Returns:
            Aggregated embeddings
        """
        if not isinstance(layer_embeddings, dict):
            raise TypeError("LastKLayerPooling expects dict of layer embeddings")

        # Get last K layers
        sorted_layers = sorted(layer_embeddings.keys())
        last_k_layers = sorted_layers[-self.k :]

        # Stack embeddings from last K layers
        stacked = np.stack([layer_embeddings[layer] for layer in last_k_layers], axis=0)
        # Shape: (k, n_samples, embed_dim)

        if self.aggregation == "mean":
            # Average across layers
            return np.mean(stacked, axis=0)

        elif self.aggregation == "concat":
            # Concatenate layers
            # Shape: (n_samples, k * embed_dim)
            return np.concatenate([layer_embeddings[layer] for layer in last_k_layers], axis=1)

        elif self.aggregation == "weighted":
            # Weighted average (more weight to later layers)
            weights = np.arange(1, self.k + 1, dtype=float)
            weights = weights / weights.sum()
            weighted = np.sum(stacked * weights[:, None, None], axis=0)
            return weighted

    def __repr__(self) -> str:
        return f"LastKLayerPooling(k={self.k}, aggregation={self.aggregation})"


class SpectralSharpeningPooling(PoolingStrategy):
    """
    Spectral sharpening via Laplacian eigenvectors.

    Projects embeddings onto top-K eigenvectors of the graph Laplacian
    to sharpen the representation by emphasizing dominant structures.

    Based on spectral clustering theory and graph signal processing.
    """

    def __init__(
        self,
        k: int = 50,
        sigma: Optional[float] = None,
        use_normalized_laplacian: bool = True,
    ):
        """
        Args:
            k: Number of top eigenvectors to project onto
            sigma: RBF kernel bandwidth (if None, uses median heuristic)
            use_normalized_laplacian: Use normalized Laplacian (recommended)
        """
        self.k = k
        self.sigma = sigma
        self.use_normalized_laplacian = use_normalized_laplacian

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply spectral sharpening.

        Args:
            embeddings: Shape (n_samples, embed_dim) or (n_samples, n_tokens, embed_dim)

        Returns:
            Sharpened embeddings (n_samples, k) or original shape if too few samples
        """
        # First pool to sample level if needed
        if embeddings.ndim == 3:
            embeddings = np.mean(embeddings, axis=1)

        n_samples, embed_dim = embeddings.shape

        # Need at least k samples
        if n_samples < self.k:
            print(
                f"Warning: Only {n_samples} samples, less than k={self.k}. "
                "Returning original embeddings."
            )
            return embeddings

        # 1. Compute pairwise similarities using RBF kernel
        if self.sigma is None:
            # Median heuristic: sigma = median of pairwise distances
            distances = cdist(embeddings, embeddings, metric="euclidean")
            self.sigma = np.median(distances[distances > 0])

        # RBF kernel: exp(-||x_i - x_j||^2 / (2 * sigma^2))
        distances_sq = cdist(embeddings, embeddings, metric="sqeuclidean")
        similarity = np.exp(-distances_sq / (2 * self.sigma**2))

        # 2. Build graph Laplacian
        # Degree matrix
        degree = np.sum(similarity, axis=1)

        if self.use_normalized_laplacian:
            # Normalized Laplacian: L = I - D^(-1/2) * W * D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-10))
            laplacian = np.eye(n_samples) - D_inv_sqrt @ similarity @ D_inv_sqrt
        else:
            # Unnormalized Laplacian: L = D - W
            D = np.diag(degree)
            laplacian = D - similarity

        # 3. Compute top K eigenvectors
        # For Laplacian, smallest eigenvalues correspond to dominant structures
        # Convert to sparse for efficiency with large matrices
        laplacian_sparse = csr_matrix(laplacian)

        try:
            # Compute k smallest eigenvalues and eigenvectors
            eigenvalues, eigenvectors = eigsh(laplacian_sparse, k=self.k, which="SM")

            # 4. Project embeddings onto eigenvectors
            # This gives us k-dimensional "sharpened" representation
            sharpened = eigenvectors  # Shape: (n_samples, k)

            return sharpened

        except Exception as e:
            print(f"Warning: Spectral decomposition failed: {e}. Returning original embeddings.")
            return embeddings

    def __repr__(self) -> str:
        return f"SpectralSharpeningPooling(k={self.k}, sigma={self.sigma})"


# Factory function for easy instantiation
def get_pooling_strategy(name: str, **kwargs) -> PoolingStrategy:
    """
    Factory function to get pooling strategy by name.

    Args:
        name: Strategy name (cls/mean/max/gem/lastk/spectral)
        **kwargs: Strategy-specific parameters

    Returns:
        PoolingStrategy instance

    Example:
        >>> pooling = get_pooling_strategy("gem", p=3.0)
        >>> pooling = get_pooling_strategy("lastk", k=4, aggregation="mean")
    """
    strategies = {
        "cls": CLSPooling,
        "mean": MeanPooling,
        "max": MaxPooling,
        "gem": GeMLPooling,
        "lastk": LastKLayerPooling,
        "spectral": SpectralSharpeningPooling,
    }

    name_lower = name.lower()
    if name_lower not in strategies:
        raise ValueError(
            f"Unknown pooling strategy: {name}. Choose from {list(strategies.keys())}"
        )

    return strategies[name_lower](**kwargs)
