"""
Clustering methods for embedding-based partitioning.

Provides various clustering algorithms to group embeddings into
semantically meaningful clusters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
import hdbscan


class ClusteringMethod(ABC):
    """Abstract base class for clustering methods."""

    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings into groups.

        Args:
            embeddings: Embeddings to cluster, shape (n_samples, embed_dim)

        Returns:
            Cluster labels, shape (n_samples,)
            Labels are integers >= 0, or -1 for noise/outliers
        """
        pass

    @abstractmethod
    def get_cluster_info(self) -> Dict:
        """
        Get information about the clustering result.

        Returns:
            Dictionary with clustering metadata (num_clusters, noise_count, etc.)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class HDBSCANMethod(ClusteringMethod):
    """
    HDBSCAN (Hierarchical Density-Based Spatial Clustering).

    Automatically determines the number of clusters based on density.
    Can handle noise and outliers (labeled as -1).
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
    ):
        """
        Args:
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples in neighborhood (defaults to min_cluster_size)
            metric: Distance metric to use
            cluster_selection_epsilon: Distance threshold for cluster selection
            cluster_selection_method: Method for selecting clusters ('eom' or 'leaf')
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method

        self.clusterer = None
        self.labels_ = None
        self.probabilities_ = None

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply HDBSCAN clustering."""
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
        )

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
            f"min_samples={self.min_samples})"
        )


class KMeansMethod(ClusteringMethod):
    """
    K-Means clustering.

    Simple baseline that requires specifying the number of clusters.
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        """
        Args:
            n_clusters: Number of clusters to form
            random_state: Random seed for reproducibility
            n_init: Number of initializations
            max_iter: Maximum iterations
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self.clusterer = None
        self.labels_ = None
        self.inertia_ = None

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply K-means clustering."""
        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )

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
        return f"KMeansMethod(n_clusters={self.n_clusters})"


class KNNGraphMethod(ClusteringMethod):
    """
    K-Nearest Neighbors graph clustering.

    Builds a k-NN graph and extracts connected components as clusters.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "cosine",
        mode: str = "connectivity",
    ):
        """
        Args:
            n_neighbors: Number of nearest neighbors
            metric: Distance metric for k-NN graph
            mode: Type of graph ('connectivity' or 'distance')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.mode = mode

        self.labels_ = None
        self.graph_ = None

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply k-NN graph clustering."""
        # Build k-NN graph
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
    """
    Spectral clustering using graph Laplacian eigenvectors.

    Projects data onto top eigenvectors of the similarity graph,
    then applies k-means in the spectral space.
    """

    def __init__(
        self,
        n_clusters: int,
        affinity: str = "rbf",
        gamma: Optional[float] = None,
        n_neighbors: int = 10,
        random_state: int = 42,
    ):
        """
        Args:
            n_clusters: Number of clusters to form
            affinity: How to construct affinity matrix ('rbf', 'nearest_neighbors')
            gamma: Kernel coefficient for RBF (if None, uses 1/n_features)
            n_neighbors: Number of neighbors for nearest_neighbors affinity
            random_state: Random seed
        """
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

        self.clusterer = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=gamma,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )

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
    """
    Factory function to get clustering method by name.

    Args:
        name: Method name (hdbscan/kmeans/knn/spectral)
        **kwargs: Method-specific parameters

    Returns:
        ClusteringMethod instance

    Example:
        >>> clustering = get_clustering_method("hdbscan", min_cluster_size=15)
        >>> clustering = get_clustering_method("kmeans", n_clusters=10)
        >>> clustering = get_clustering_method("knn", n_neighbors=15)
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
