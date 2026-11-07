"""Clustering methods for embedding-based partitioning.

This module provides various clustering algorithms to group image embeddings
into semantically meaningful clusters for federated learning data partitioning.

Available Methods
-----------------
HDBSCANMethod
    Density-based clustering with automatic cluster detection.
KMeansMethod
    Simple k-means clustering with fixed number of clusters.
KNNGraphMethod
    k-NN graph-based clustering using connected components.
SpectralMethod
    Spectral clustering using graph Laplacian eigenvectors.

Usage
-----
Use the factory function for easy instantiation:

    >>> from partitioning.embedding_utils.clustering_methods import get_clustering_method
    >>> clustering = get_clustering_method("hdbscan", min_cluster_size=100, metric="cosine")
    >>> labels = clustering.cluster(embeddings)
    >>> info = clustering.get_cluster_info()

Or instantiate directly:

    >>> from partitioning.embedding_utils.clustering_methods import HDBSCANMethod
    >>> clustering = HDBSCANMethod(min_cluster_size=100, min_samples=5)
    >>> labels = clustering.cluster(embeddings)

See Also
--------
EmbeddingSplitter : Uses clustering methods for embedding-based partitioning.
EmbeddingConfig : Configuration for selecting clustering methods.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from importlib import import_module
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.sparse.csgraph import connected_components


DEFAULT_CLUSTERING_THREAD_LIMIT = 8
CLUSTERING_THREAD_LIMIT_ENV = "FDS_CLUSTERING_MAX_THREADS"


def _require_sklearn_attr(module_name: str, attr_name: str):
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "Clustering features require the optional dependency 'scikit-learn'."
        ) from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"Expected '{attr_name}' in optional dependency module '{module_name}'."
        ) from exc


def _get_clustering_thread_limit() -> Optional[int]:
    raw_limit = os.getenv(CLUSTERING_THREAD_LIMIT_ENV)
    if raw_limit in (None, ""):
        return DEFAULT_CLUSTERING_THREAD_LIMIT

    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        return DEFAULT_CLUSTERING_THREAD_LIMIT

    return parsed_limit if parsed_limit > 0 else None


def _threadpool_limit_context(limit: Optional[int] = None):
    effective_limit = _get_clustering_thread_limit() if limit is None else limit
    if effective_limit is None:
        return nullcontext()

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()

    return threadpool_limits(limits=effective_limit)


class ClusteringMethod(ABC):
    """Abstract base class for clustering methods.

    All clustering methods must implement :meth:`cluster` to assign samples
    to groups and :meth:`get_cluster_info` to report clustering statistics.
    """

    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings into groups.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to cluster, shape (n_samples, embed_dim).

        Returns
        -------
        np.ndarray
            Cluster labels, shape (n_samples,). Labels are integers >= 0,
            or -1 for noise/outliers (HDBSCAN only).
        """
        pass

    @abstractmethod
    def get_cluster_info(self) -> Dict:
        """Get information about the clustering result.

        Returns
        -------
        dict
            Dictionary with clustering metadata including:

            - "method": Name of the clustering method
            - "num_clusters": Number of clusters found
            - "cluster_sizes": Dict mapping cluster ID to sample count
            - Additional method-specific information
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class HDBSCANMethod(ClusteringMethod):
    """HDBSCAN (Hierarchical Density-Based Spatial Clustering).

    Automatically determines the number of clusters based on local density.
    Unlike k-means, does not require specifying the number of clusters.
    Can identify noise/outliers (labeled as -1).

    Parameters
    ----------
    min_cluster_size : int, optional
        Minimum number of samples in a cluster. Smaller values create more
        clusters. Default: 100.
    min_samples : int, optional
        Minimum samples in a neighborhood for core points. If None, defaults
        to min_cluster_size. Smaller values are less conservative.
    metric : str, optional
        Distance metric. Default: "euclidean".
    cluster_selection_epsilon : float, optional
        Distance threshold for cluster merging. Larger values merge more
        clusters. Default: 0.0 (no merging based on distance).
    cluster_selection_method : str, optional
        Method for extracting clusters from hierarchy:

        - "eom": Excess of Mass (default, better for varying densities)
        - "leaf": Leaf nodes (produces more fine-grained clusters)

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels after clustering (-1 indicates noise).
    probabilities_ : np.ndarray
        Cluster membership probabilities.
    clusterer : hdbscan.HDBSCAN
        The underlying HDBSCAN clusterer.

    Notes
    -----
    - Recommended default for embedding-based partitioning
    - Robust to varying cluster densities and shapes
    - Noise samples require special handling (see EmbeddingSplitter.handle_noise)
    """

    def __init__(
        self,
        min_cluster_size: int = 100,
        min_samples: Optional[int] = None,
        hdbscan_metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        hdbscan_method: str = "eom",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = hdbscan_metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = hdbscan_method

        self.clusterer = None
        self.labels_ = None
        self.probabilities_ = None

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply HDBSCAN clustering."""
        hdbscan_cls = _require_sklearn_attr("sklearn.cluster", "HDBSCAN")
        self.clusterer = hdbscan_cls(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            # n_jobs=4,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
        )

        with _threadpool_limit_context():
            self.labels_ = self.clusterer.fit_predict(embeddings)
        self.probabilities_ = self.clusterer.probabilities_

        return self.labels_

    def get_cluster_info(self) -> Dict:
        """Get HDBSCAN clustering information."""
        if self.labels_ is None:
            return {}

        unique_labels = set(self.labels_)
        num_clusters = len(unique_labels - {-1})  # Exclude noise label
        noise_count = np.sum(self.labels_ == -1)

        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(self.labels_ == label))

        return {
            "method": "HDBSCAN",
            "num_clusters": num_clusters,
            "noise_count": noise_count,
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
        }

    def __repr__(self) -> str:
        return (
            f"HDBSCANMethod(min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}),"
            f"metric={self.metric}),"
            f"cluster_selection_method={self.cluster_selection_method})"
        )


class KMeansMethod(ClusteringMethod):
    """K-Means clustering.

    Classic centroid-based clustering. Simple and fast, but requires
    specifying the number of clusters in advance.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form. Required parameter.
    random_state : int, optional
        Random seed for reproducibility. Default: 42.
    n_init : int, optional
        Number of different centroid initializations to try. The best
        result is kept. Default: 10.
    max_iter : int, optional
        Maximum number of iterations per run. Default: 300.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels after clustering.
    inertia_ : float
        Sum of squared distances to cluster centers.
    clusterer : sklearn.cluster.KMeans
        The underlying k-means clusterer.

    Notes
    -----
    - Fastest clustering method
    - Assumes spherical, equally-sized clusters
    - Good baseline for comparison with other methods
    - No noise detection (all samples assigned to clusters)
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self.clusterer = None
        self.labels_ = None
        self.inertia_ = None

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply K-means clustering."""
        kmeans_cls = _require_sklearn_attr("sklearn.cluster", "KMeans")
        self.clusterer = kmeans_cls(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )

        with _threadpool_limit_context():
            self.labels_ = self.clusterer.fit_predict(embeddings)
        self.inertia_ = self.clusterer.inertia_

        return self.labels_

    def get_cluster_info(self) -> Dict:
        """Get K-means clustering information."""
        if self.labels_ is None:
            return {}

        cluster_sizes = {}
        for label in range(self.n_clusters):
            cluster_sizes[label] = int(np.sum(self.labels_ == label))

        return {
            "method": "KMeans",
            "num_clusters": self.n_clusters,
            "cluster_sizes": cluster_sizes,
            "inertia": float(self.inertia_) if self.inertia_ is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"KMeansMethod(n_clusters={self.n_clusters}),"
            f"random_state={self.random_state})"
            f"n_init={self.n_init})"
        )


class KNNGraphMethod(ClusteringMethod):
    """K-Nearest Neighbors graph clustering.

    Builds a k-NN graph and extracts connected components as clusters.
    Simple and parameter-free (aside from k), but number of clusters
    depends on the graph connectivity.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of nearest neighbors for each sample. Larger values
        create more connected graphs (fewer clusters). Default: 15.
    metric : str, optional
        Distance metric for k-NN computation. Default: "cosine".
    mode : str, optional
        Type of graph weights:

        - "connectivity": Binary edges (0 or 1)
        - "distance": Edge weights = distances

        Default: "connectivity".

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels after clustering.
    graph_ : sparse matrix
        The k-NN graph (symmetric).

    Notes
    -----
    - Number of clusters depends on graph structure
    - Sensitive to n_neighbors parameter
    - No noise detection (all samples in some cluster)
    - Graph is made symmetric (undirected)
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "cosine",
        mode: str = "connectivity",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.mode = mode

        self.labels_ = None
        self.graph_ = None

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply k-NN graph clustering."""
        # Build k-NN graph
        kneighbors_graph = _require_sklearn_attr("sklearn.neighbors", "kneighbors_graph")
        self.graph_ = kneighbors_graph(
            embeddings,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            mode=self.mode,
            include_self=False,
        )

        # Make graph symmetric (undirected)
        self.graph_ = self.graph_ + self.graph_.T

        # Find connected components
        n_components, self.labels_ = connected_components(
            csgraph=self.graph_,
            directed=False,
            return_labels=True,
        )

        return self.labels_

    def get_cluster_info(self) -> Dict:
        """Get k-NN graph clustering information."""
        if self.labels_ is None:
            return {}

        unique_labels = set(self.labels_)
        num_clusters = len(unique_labels)

        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(self.labels_ == label))

        return {
            "method": "KNNGraph",
            "num_clusters": num_clusters,
            "cluster_sizes": cluster_sizes,
            "n_neighbors": self.n_neighbors,
        }

    def __repr__(self) -> str:
        return f"KNNGraphMethod(n_neighbors={self.n_neighbors})"


class SpectralMethod(ClusteringMethod):
    """Spectral clustering using graph Laplacian eigenvectors.

    Projects data onto the top eigenvectors of a similarity graph's
    Laplacian, then applies k-means in this spectral embedding space.
    Effective for non-convex cluster shapes.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form. Required parameter.
    affinity : str, optional
        How to construct the affinity matrix:

        - "rbf": RBF (Gaussian) kernel
        - "nearest_neighbors": k-NN graph

        Default: "rbf".
    gamma : float, optional
        Kernel coefficient for RBF. If None, uses 1/n_features.
    n_neighbors : int, optional
        Number of neighbors for "nearest_neighbors" affinity. Default: 10.
    random_state : int, optional
        Random seed for reproducibility. Default: 42.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels after clustering.
    clusterer : sklearn.cluster.SpectralClustering
        The underlying spectral clusterer.

    Notes
    -----
    - Good for complex, non-convex cluster shapes
    - More expensive than k-means (O(n^3) for eigendecomposition)
    - No noise detection (all samples assigned to clusters)
    - Consider using "nearest_neighbors" for sparse similarity graphs
    """

    def __init__(
        self,
        n_clusters: int,
        affinity: str = "rbf",
        gamma: Optional[float] = None,
        n_neighbors: int = 10,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.clusterer = None
        self.labels_ = None

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply spectral clustering."""
        # Set gamma if not provided
        gamma = self.gamma if self.gamma is not None else 1.0 / embeddings.shape[1]

        spectral_cls = _require_sklearn_attr("sklearn.cluster", "SpectralClustering")
        self.clusterer = spectral_cls(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=gamma,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            # n_jobs=4,
        )

        with _threadpool_limit_context():
            self.labels_ = self.clusterer.fit_predict(embeddings)

        return self.labels_

    def get_cluster_info(self) -> Dict:
        """Get spectral clustering information."""
        if self.labels_ is None:
            return {}

        cluster_sizes = {}
        for label in range(self.n_clusters):
            cluster_sizes[label] = int(np.sum(self.labels_ == label))

        return {
            "method": "Spectral",
            "num_clusters": self.n_clusters,
            "cluster_sizes": cluster_sizes,
            "affinity": self.affinity,
        }

    def __repr__(self) -> str:
        return f"SpectralMethod(n_clusters={self.n_clusters}, affinity={self.affinity})"


# Factory function for easy instantiation
def get_clustering_method(name: str, **kwargs) -> ClusteringMethod:
    """Factory function to get clustering method by name.

    Parameters
    ----------
    name : str
        Method name. One of:

        - "hdbscan": HDBSCAN (use min_cluster_size=..., min_samples=...)
        - "kmeans": K-Means (use n_clusters=...)
        - "knn": k-NN Graph (use n_neighbors=...)
        - "spectral": Spectral (use n_clusters=..., affinity=...)

    **kwargs
        Method-specific parameters passed to the constructor.

    Returns
    -------
    ClusteringMethod
        Initialized clustering method instance.

    Raises
    ------
    ValueError
        If the method name is not recognized.

    Examples
    --------
    >>> clustering = get_clustering_method("hdbscan", min_cluster_size=100)
    >>> clustering = get_clustering_method("kmeans", n_clusters=10)
    >>> clustering = get_clustering_method("knn", n_neighbors=15)
    >>> clustering = get_clustering_method("spectral", n_clusters=10, affinity="rbf")
    """
    methods = {
        "hdbscan": HDBSCANMethod,
        "kmeans": KMeansMethod,
        "knn": KNNGraphMethod,
        "spectral": SpectralMethod,
    }

    name_lower = name.lower()
    if name_lower not in methods:
        raise ValueError(
            f"Unknown clustering method: {name}. Choose from {list(methods.keys())}"
        )

    return methods[name_lower](**kwargs)
