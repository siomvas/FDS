"""Pooling strategies for aggregating token embeddings.

This module provides various pooling strategies to aggregate token-level
embeddings (from Vision Transformers) into fixed-size sample-level
representations suitable for clustering and downstream tasks.

Available Strategies
--------------------
MeanPooling
    Average all token embeddings (spatial mean pooling).
MaxPooling
    Element-wise maximum across all tokens.
GeMLPooling
    Generalized Mean pooling with configurable power parameter.
LastKLayerPooling
    Aggregate embeddings from multiple transformer layers.
SpectralSharpeningPooling
    Project embeddings onto graph Laplacian eigenvectors for sharpening.

Note
----
CLSPooling has been removed as it is vacuous. To use CLS token, simply specify
`return_which_tokens="cls"` when calling the extractor's `extract_embeddings()` method.

Usage
-----
Use the factory function for easy instantiation:

    >>> from partitioning.embedding_utils.pooling_strategies import get_pooling_strategy
    >>> pooling = get_pooling_strategy("gem", p=3.0)
    >>> pooled = pooling.pool(token_embeddings)

Or instantiate directly:

    >>> from partitioning.embedding_utils.pooling_strategies import GeMLPooling
    >>> pooling = GeMLPooling(p=3.0)
    >>> pooled = pooling.pool(token_embeddings)

See Also
--------
EmbeddingSplitter : Uses pooling strategies for embedding-based partitioning.
EmbeddingConfig : Configuration for selecting pooling strategies.
"""

from abc import ABC, abstractmethod
import logging
import warnings
from typing import Dict, Optional
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class PoolingStrategy(ABC):
    """Abstract base class for pooling strategies.

    All pooling strategies must implement the :meth:`pool` method that
    transforms token-level embeddings into sample-level embeddings.
    """

    @abstractmethod
    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Pool token embeddings to sample-level embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Token embeddings with shape:

            - (n_samples, embed_dim) for single token per sample
            - (n_samples, n_tokens, embed_dim) for multiple tokens per sample

        Returns
        -------
        np.ndarray
            Pooled embeddings of shape (n_samples, embed_dim).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MeanPooling(PoolingStrategy):
    """Average all token embeddings.

    Computes element-wise mean across all spatial tokens, providing a
    representation that considers all image regions equally.

    Notes
    -----
    - Good for models without meaningful CLS token
    - Captures global context from all spatial locations
    - More robust to local variations than CLS pooling
    """

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Average over token dimension."""
        if embeddings.ndim == 2:
            raise ValueError("MeanPooling expects 3D embeddings with multiple tokens per sample.")
        elif embeddings.ndim == 3:
            # Mean over tokens
            return np.mean(embeddings, axis=1)
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")

    def __repr__(self) -> str:
        return "MeanPooling()"


class MaxPooling(PoolingStrategy):
    """Max pooling over token embeddings.

    Computes element-wise maximum across all spatial tokens, retaining
    only the strongest activations from any spatial location.

    Notes
    -----
    - Emphasizes the most salient features
    - More sensitive to discriminative local features
    - Can be less robust to noise than mean pooling
    """

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Max pooling over token dimension."""
        if embeddings.ndim == 2:
            raise ValueError("MaxPooling expects 3D embeddings with multiple tokens per sample.")
        elif embeddings.ndim == 3:
            return np.max(embeddings, axis=1)
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")

    def __repr__(self) -> str:
        return "MaxPooling()"


class GeMLPooling(PoolingStrategy):
    """Generalized Mean (GeM) Pooling.

    Computes generalized mean pooling: ``(mean(|x|^p))^(1/p)``.
    Interpolates between different pooling behaviors based on p:

    - p=1: Arithmetic mean (equivalent to MeanPooling)
    - p=2: Quadratic mean (RMS)
    - p>1: Emphasizes larger activations more than mean pooling
    - p->inf: Approaches max pooling

    Parameters
    ----------
    p : float, optional
        GeM pooling exponent. Typical values: 3.0-4.0. Default: 3.0.
    eps : float, optional
        Small constant for numerical stability. Default: 1e-6.

    Notes
    -----
    - Recommended for image retrieval and fine-grained recognition
    - Often outperforms simple mean pooling for similarity tasks
    - Handles signed values using sign-preserving power computation

    References
    ----------
    Radenovic et al., "Fine-tuning CNN Image Retrieval with No Human
    Annotation" (2018)
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        self.p = p
        self.eps = eps

    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply GeM pooling with proper handling of signed values."""
        if embeddings.ndim == 2:
            raise ValueError("GeMLPooling expects 3D embeddings with multiple tokens per sample.")
        elif embeddings.ndim == 3:
            # GeM formula: (mean(x^p))^(1/p)
            # For signed values, use: sign(x) * |x|^p to preserve signs
            # This ensures that when p=1, we get mean(x) exactly

            # Compute signed power: sign(x) * (|x| + eps)^p
            signs = np.sign(embeddings)
            abs_embeddings = np.abs(embeddings) + self.eps
            signed_powered = signs * np.power(abs_embeddings, self.p)

            # Mean over token dimension
            mean_powered = np.mean(signed_powered, axis=1)

            # Inverse signed power: sign(mean) * |mean|^(1/p)
            result_signs = np.sign(mean_powered)
            result = result_signs * np.power(np.abs(mean_powered), 1.0 / self.p)

            return result
        else:
            raise ValueError(f"Expected 2D or 3D embeddings, got shape {embeddings.shape}")

    def __repr__(self) -> str:
        return f"GeMLPooling(p={self.p})"


class LastKLayerPooling(PoolingStrategy):
    """Aggregate embeddings from last K transformer layers.

    Combines information from multiple layers of a transformer, which can
    capture different levels of semantic abstraction.

    Parameters
    ----------
    layer_aggregation : str, optional
        Aggregation method across the last K layers. Options:

        - "mean": Average across layers (preserves dimension)
        - "concat": Concatenate layers (output dim = k * embed_dim)
        - "weighted": Weighted average (later layers weighted higher)

        Default: "mean".
    token_aggregation: str, optional
        Aggregation method across tokens after layer aggregation. Options:

        - "mean": Average across tokens
        - "max": Max across tokens
        - "gem": Generalized mean pooling (use p=... kwarg)
        - "cls": Use CLS token (if available)

        Default: "mean".

    Notes
    -----
    - Requires transformer models that output hidden states
    - "concat" increases embedding dimension by factor k
    - "weighted" assigns linearly increasing weights to later layers
    - Earlier layers capture low-level features, later layers capture semantics
    - After layer aggregation, further token aggregation is applied as defined by `token_aggregation`

    Raises
    ------
    ValueError
        If aggregation method is not recognized.
    TypeError
        If input is not a dictionary of layer embeddings.
    """

    def __init__(self, layer_aggregation: str = "mean", token_aggregation: str = "mean", **kwargs):
        self.layer_aggregation = layer_aggregation
        self.token_aggregation = token_aggregation
        self.kwargs = kwargs

    def pool_layers(self, layer_embeddings: np.ndarray) -> np.ndarray:
        """
        Pool embeddings from multiple layers.

        Args:
            layer_embeddings: Dict mapping layer index to embeddings (n_samples, embed_dim)

        Returns:
            Aggregated embeddings
        """
        if layer_embeddings.ndim != 4:
            raise ValueError(
                f"Expected layer_embeddings to be a [n_samples, n_layers, n_tokens, embed_dim] array."
                f"Got shape: {layer_embeddings.shape}"
            )

        n_samples, n_layers, n_tokens, embed_dim = layer_embeddings.shape

        logger.debug(
            "Pooling %d samples from %d layers with %d tokens each (embed_dim=%d).",
            n_samples,
            n_layers,
            n_tokens,
            embed_dim,
        )

        if self.layer_aggregation == "mean":

            return np.mean(layer_embeddings, axis=1)

        elif self.layer_aggregation == "concat":
            # Shape: (n_samples, n_tokens, n_layers * embed_dim)
            transposed = np.transpose(layer_embeddings, (0, 2, 1, 3))
            return transposed.reshape(n_samples, n_tokens, n_layers * embed_dim)

        elif self.layer_aggregation == "weighted":
            # Weighted average (more weight to later layers)
            weights = np.arange(1, n_layers+1, dtype=float)
            weights = weights / weights.sum()
            weighted = np.tensordot(layer_embeddings, weights, axes=([1], [0]))
            return weighted
        
        else:
            raise ValueError(f"Unknown aggregation: {self.layer_aggregation}. Choose from mean/concat/weighted")
    
    def pool_tokens(self, embeddings: np.ndarray) -> np.ndarray:

        if self.token_aggregation == "mean":
            return MeanPooling().pool(embeddings)
        elif self.token_aggregation == "max":
            return MaxPooling().pool(embeddings)
        elif self.token_aggregation == "gem":
            p = self.kwargs.get("p", 3.0)
            return GeMLPooling(p=p).pool(embeddings)
        elif self.token_aggregation == "cls":
            if embeddings.ndim != 3:
                raise ValueError("CLS token aggregation expects 3D embeddings with multiple tokens per sample.")
            return embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown token aggregation: {self.token_aggregation}. Choose from mean/max/gem/cls")
    
    def pool(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Pool embeddings from last K layers and then across tokens.

        Args:
            embeddings: Shape (n_samples, n_layers, n_tokens, embed_dim)

        Returns:
            Pooled embeddings (n_samples, embed_dim) or (n_samples, k*embed_dim) if concat
        """
        # First pool across layers
        layer_pooled = self.pool_layers(embeddings)

        # Then pool across tokens
        token_pooled = self.pool_tokens(layer_pooled)

        return token_pooled

    def __repr__(self) -> str:
        return f"LastKLayerPooling(layer_aggregation={self.layer_aggregation}, token_aggregation={self.token_aggregation})"


class SpectralSharpeningPooling(PoolingStrategy):
    """Spectral sharpening via Laplacian eigenvectors.

    Deprecated
    ----------
    This pooling strategy is deprecated. It is computationally expensive and its
    dataset-level graph construction makes it a poor fit for the current embedding
    partitioning pipeline.

    Projects embeddings onto the top-K eigenvectors of the sample similarity
    graph's Laplacian. This "sharpens" the representation by emphasizing the
    dominant cluster structure in the data.

    The process:
    1. Build a similarity graph using RBF kernel
    2. Compute the graph Laplacian
    3. Extract top-K eigenvectors (smallest eigenvalues)
    4. Use eigenvector coordinates as new representation

    Parameters
    ----------
    k : int, optional
        Number of top eigenvectors to project onto. This determines the
        output dimension. Default: 50.
    sigma : float, optional
        RBF kernel bandwidth. If None, uses median heuristic (median of
        non-zero pairwise distances). Default: None.
    use_normalized_laplacian : bool, optional
        Whether to use normalized Laplacian (recommended for better
        numerical stability). Default: True.

    Attributes
    ----------
    sigma : float
        The kernel bandwidth used (computed or provided).

    Notes
    -----
    - Output dimension is k (not embed_dim)
    - Requires at least k samples to compute
    - Computationally expensive for large datasets
    - Most effective for data with clear cluster structure

    References
    ----------
    Based on spectral clustering theory and graph signal processing.
    See: von Luxburg, "A Tutorial on Spectral Clustering" (2007)
    """

    def __init__(
        self,
        k: int = 50,
        sigma: Optional[float] = None,
        use_normalized_laplacian: bool = True,
    ):
        warnings.warn(
            "SpectralSharpeningPooling is deprecated and may be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            logger.warning(
                "Only %d samples, less than spectral k=%d; returning original embeddings.",
                n_samples,
                self.k,
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

        except (ArpackNoConvergence, RuntimeError, ValueError, FloatingPointError) as e:
            logger.warning("Spectral decomposition failed: %s. Returning original embeddings.", e)
            return embeddings

    def __repr__(self) -> str:
        return f"SpectralSharpeningPooling(k={self.k}, sigma={self.sigma})"


# Factory function for easy instantiation
def get_pooling_strategy(name: str, **kwargs) -> PoolingStrategy:
    """Factory function to get pooling strategy by name.

    Parameters
    ----------
    name : str
        Strategy name. One of:

        - "mean": Mean pooling
        - "max": Max pooling
        - "gem": Generalized mean pooling (use p=... kwarg)
        - "lastk": Last K layers (use k=..., aggregation=... kwargs)
        - "spectral": Spectral sharpening (use k=..., sigma=... kwargs)

    **kwargs
        Strategy-specific parameters passed to the constructor.

    Returns
    -------
    PoolingStrategy
        Initialized pooling strategy instance.

    Raises
    ------
    ValueError
        If the strategy name is not recognized.

    Notes
    -----
    For CLS token pooling, do not use this factory. Instead, call the
    extractor's `extract_embeddings()` with `return_which_tokens="cls"`.

    Examples
    --------
    >>> pooling = get_pooling_strategy("gem", p=3.0)
    >>> pooling = get_pooling_strategy("lastk", k=4, aggregation="mean")
    >>> pooling = get_pooling_strategy("spectral", k=50)
    """
    strategies = {
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
