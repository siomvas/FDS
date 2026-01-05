"""Embedding-based data splitting using foundation models and clustering.

This module implements semantic data partitioning by:
1. Extracting image embeddings using foundation models (via ModelRegistry)
2. Applying optional pooling strategies to aggregate spatial features
3. Clustering embeddings to group semantically similar samples
4. Directly mapping clusters to clients (1 cluster = 1 client)

This creates natural, semantically-meaningful heterogeneity where each client
receives data from a distinct cluster, mimicking real-world scenarios where
different clients have different types of data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from .base_splitter import BaseDatasetSplitter, DatasetSplits, ClientSplit
from data_loading.dataset import BaseDataset
from embedding.embedding_extractors import ModelRegistry, EmbeddingExtractor
from embedding.cache_manager import CacheManager
from .embedding_utils.pooling_strategies import get_pooling_strategy, PoolingStrategy
from .embedding_utils.clustering_methods import get_clustering_method, ClusteringMethod


# Mapping from pooling strategy to extractor configuration
POOLING_TO_EXTRACTION = {
    "cls": {"return_which_tokens": "cls", "needs_pooling": False},
    "mean": {"return_which_tokens": "all_last", "needs_pooling": True},
    "max": {"return_which_tokens": "all_last", "needs_pooling": True},
    "gem": {"return_which_tokens": "all_last", "needs_pooling": True},
    "lastk": {"return_which_tokens": "k_last_cls", "needs_pooling": True},
    "spectral": {"return_which_tokens": "all_last", "needs_pooling": True},
}


class EmbeddingSplitter(BaseDatasetSplitter):
    """Splits datasets using embedding-based clustering for federated learning.

    This splitter creates semantically-meaningful heterogeneous partitions by:

    1. Extracting image embeddings using foundation models (via ModelRegistry)
    2. Optionally applying pooling strategies to aggregate features
    3. Clustering embeddings to group semantically similar samples
    4. Directly mapping clusters to clients (n_clients = number of clusters)

    The resulting partitions create natural heterogeneity where each client
    receives data from a distinct semantic cluster.

    Parameters
    ----------
    train_dataset : BaseDataset
        Training dataset containing image paths and labels.
    val_dataset : BaseDataset or None
        Validation dataset. If None, no validation split is created.
    extractor_name : str
        Model identifier for ModelRegistry (e.g., "dinov2-base",
        "clip-large-patch14", "inception_v3", "mae-base").
    clustering_method : str
        Clustering algorithm to use:

        - "hdbscan": Density-based, auto-determines number of clusters
        - "kmeans": Fixed number of clusters (requires n_clusters param)
        - "knn": k-NN graph connected components
        - "spectral": Fixed number of clusters (requires n_clusters param)

    clustering_params : dict, optional
        Method-specific clustering parameters:

        - HDBSCAN: min_cluster_size, min_samples, metric
        - KMeans: n_clusters (required), random_state, n_init
        - KNN: n_neighbors, metric
        - Spectral: n_clusters (required), affinity, gamma

    pooling_strategy : str, optional
        How to aggregate token embeddings:

        - "cls": CLS token only (no post-processing)
        - "mean": Average all tokens
        - "max": Max pooling across tokens
        - "gem": Generalized mean pooling
        - "lastk": Aggregate last K layers
        - "spectral": Spectral sharpening

        Default: "cls".

    pooling_params : dict, optional
        Strategy-specific pooling parameters:

        - gem: p (power parameter, default 3.0)
        - lastk: k (number of layers, default 4), aggregation ("mean"/"concat"/"weighted")
        - spectral: k (eigenvectors, default 50), sigma (RBF bandwidth)

    valset_type : str, optional
        How to handle validation data:

        - "global": All clients share the same validation set
        - "local": Split validation proportionally across clients
        - "none": No validation set for clients

        Default: "global".

    seed : int, optional
        Random seed for reproducibility.
    cache_dir : str, optional
        Directory for caching embeddings. Default: "./.embedding_cache".
    dataset_name : str, optional
        Dataset name for efficient cache key computation.
    base_path : str, optional
        Base path for path normalization in caching.
    device : str, optional
        Device for computation ("cuda", "cpu", etc.). Auto-detects if None.
    batch_size : int, optional
        Batch size for embedding extraction. Default: 32.

    Attributes
    ----------
    n_clients : int
        Number of clients (determined after clustering).
    cluster_labels_ : np.ndarray or None
        Cluster assignments after splitting.
    cluster_info_ : dict or None
        Clustering metadata after splitting.

    Examples
    --------
    Basic usage with HDBSCAN (auto-determines number of clients):

    >>> from partitioning.embedding_splitter import EmbeddingSplitter
    >>> from data_loading.dataset import BaseDataset
    >>>
    >>> train_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "train")
    >>> val_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "val")
    >>>
    >>> splitter = EmbeddingSplitter(
    ...     train_dataset=train_dataset,
    ...     val_dataset=val_dataset,
    ...     extractor_name="dinov2-base",
    ...     clustering_method="hdbscan",
    ...     clustering_params={"min_cluster_size": 15},
    ...     pooling_strategy="cls",
    ...     valset_type="global",
    ...     seed=42,
    ... )
    >>> splits = splitter.get_client_splits()
    >>> print(f"Found {splitter.n_clients} clusters/clients")
    >>> splitter.store_client_splits("embedding_splits.json")

    Using KMeans with fixed number of clients:

    >>> splitter = EmbeddingSplitter(
    ...     train_dataset=train_dataset,
    ...     val_dataset=val_dataset,
    ...     extractor_name="clip-base-patch16",
    ...     clustering_method="kmeans",
    ...     clustering_params={"n_clusters": 10},
    ...     pooling_strategy="mean",
    ...     dataset_name="tiny_imagenet",
    ...     base_path="/path/to/data",
    ... )
    >>> splits = splitter.get_client_splits()
    >>> assert splitter.n_clients == 10

    See Also
    --------
    DirichletSplitter : Label-based non-IID splitting.
    MultivariateDirichletSplitter : VLM-based semantic splitting.

    Notes
    -----
    - Embeddings are automatically cached for re-use across runs.
    - For HDBSCAN, noise samples (label=-1) are assigned to nearest cluster.
    - Number of clients equals number of clusters (no assignment strategies).
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset],
        extractor_name: str,
        clustering_method: str,
        clustering_params: Optional[Dict] = None,
        pooling_strategy: str = "cls",
        pooling_params: Optional[Dict] = None,
        valset_type: str = "global",
        seed: Optional[int] = None,
        cache_dir: str = "./.embedding_cache",
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        use_cache: bool = True,
    ):
        """Initialize the embedding splitter."""
        # Validate pooling strategy
        valid_pooling = ["cls", "mean", "max", "gem", "lastk", "spectral"]
        if pooling_strategy not in valid_pooling:
            raise ValueError(
                f"Invalid pooling: {pooling_strategy}. Choose from {valid_pooling}"
            )

        # Validate clustering method
        valid_clustering = ["hdbscan", "kmeans", "knn", "spectral"]
        if clustering_method.lower() not in valid_clustering:
            raise ValueError(
                f"Invalid clustering: {clustering_method}. Choose from {valid_clustering}"
            )

        # n_clients determined AFTER clustering
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=1,  # Placeholder, updated after clustering
            heterogeneity=0.0,  # Not used in this splitter
            valset_type=valset_type,
            seed=seed,
        )

        self.extractor_name = extractor_name
        self.clustering_method = clustering_method.lower()
        self.clustering_params = clustering_params or {}
        self.pooling_strategy = pooling_strategy
        self.pooling_params = pooling_params or {}
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.device = device
        self.batch_size = batch_size
        self.use_cache = use_cache

        # Results stored after computation
        self.cluster_labels_: Optional[np.ndarray] = None
        self.cluster_info_: Optional[Dict] = None

    def _get_extractor(self) -> EmbeddingExtractor:
        """Create extractor using ModelRegistry.

        Returns
        -------
        EmbeddingExtractor
            Initialized extractor with cache manager.
        """
        cache_manager = CacheManager(cache_dir=self.cache_dir, enabled=True)
        return ModelRegistry.create(
            self.extractor_name,
            device=self.device,
            batch_size=self.batch_size,
            cache_manager=cache_manager,
        )

    def _get_embeddings(self) -> np.ndarray:
        """Extract and optionally pool embeddings.

        Returns
        -------
        np.ndarray
            Pooled embeddings of shape (n_samples, embed_dim).
        """
        extractor = self._get_extractor()
        image_paths = [
            str(self.train_dataset[i][0]) for i in range(len(self.train_dataset))
        ]

        # Get extraction config for this pooling strategy
        extraction_config = POOLING_TO_EXTRACTION[self.pooling_strategy]

        print(f"Extracting embeddings using {self.extractor_name}...")
        print(f"  Pooling strategy: {self.pooling_strategy}")
        print(f"  Token selection: {extraction_config['return_which_tokens']}")

        # Extract embeddings (caching controlled by use_cache parameter)
        print(f"  Caching: {'enabled' if self.use_cache else 'disabled'}")
        raw_embeddings = extractor.extract_embeddings(
            images=image_paths,
            return_which_tokens=extraction_config["return_which_tokens"],
            k_last=self.pooling_params.get("k", 4),  # For k_last extraction
            use_cache=self.use_cache,
            dataset_name=self.dataset_name,
            base_path=self.base_path,
        )

        # Apply pooling if needed
        if extraction_config["needs_pooling"]:
            print(f"  Applying {self.pooling_strategy} pooling...")
            pooling = get_pooling_strategy(
                self.pooling_strategy, **self.pooling_params
            )
            embeddings = pooling.pool(raw_embeddings)
        else:
            embeddings = raw_embeddings

        print(f"✓ Final embeddings shape: {embeddings.shape}")
        return embeddings

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply clustering method to embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to cluster, shape (n_samples, embed_dim).

        Returns
        -------
        np.ndarray
            Cluster labels, shape (n_samples,). Labels are integers >= 0,
            or -1 for noise (HDBSCAN only).
        """
        clustering = get_clustering_method(
            self.clustering_method, **self.clustering_params
        )

        print(f"Clustering with {self.clustering_method}...")
        cluster_labels = clustering.cluster(embeddings)
        self.cluster_info_ = clustering.get_cluster_info()

        num_clusters = self.cluster_info_.get("num_clusters", 0)
        print(f"✓ Found {num_clusters} clusters")

        if "noise_count" in self.cluster_info_:
            print(f"  Noise samples: {self.cluster_info_['noise_count']}")

        return cluster_labels

    def _assign_noise_to_nearest(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Assign noise points (label=-1) to their nearest cluster.

        Parameters
        ----------
        embeddings : np.ndarray
            Sample embeddings, shape (n_samples, embed_dim).
        labels : np.ndarray
            Cluster labels, shape (n_samples,). May contain -1 for noise.

        Returns
        -------
        np.ndarray
            Updated labels with noise points assigned to nearest clusters.
        """
        noise_mask = labels == -1
        if not noise_mask.any():
            return labels

        print(f"  Assigning {noise_mask.sum()} noise points to nearest clusters...")

        # Get cluster centroids
        unique_labels = [l for l in set(labels) if l != -1]
        centroids = {
            label: embeddings[labels == label].mean(axis=0) for label in unique_labels
        }

        # Assign each noise point to nearest centroid
        labels = labels.copy()
        for idx in np.where(noise_mask)[0]:
            distances = {
                label: np.linalg.norm(embeddings[idx] - centroid)
                for label, centroid in centroids.items()
            }
            labels[idx] = min(distances, key=distances.get)

        return labels

    def _compute_client_splits(self) -> DatasetSplits:
        """Compute client splits using embedding clustering.

        This is the main method that orchestrates the embedding-based
        partitioning pipeline:

        1. Extract embeddings from training images
        2. Cluster embeddings into groups
        3. Handle noise points (assign to nearest cluster)
        4. Set n_clients = number of clusters
        5. Directly map clusters to clients (cluster_i -> client_i)
        6. Build the final splits structure with train/val data

        Returns
        -------
        DatasetSplits
            Dictionary with structure::

                {
                    "clients": {
                        "client_0": {"train": [...], "val": [...]},
                        ...
                    },
                    "global_val": [...] or None
                }
        """
        # 1. Extract embeddings (caching handled by extractor)
        embeddings = self._get_embeddings()

        # 2. Cluster
        cluster_labels = self._cluster(embeddings)
        self.cluster_labels_ = cluster_labels

        # 3. Handle noise (HDBSCAN returns -1 for noise)
        cluster_labels = self._assign_noise_to_nearest(embeddings, cluster_labels)

        # 4. n_clients = num_clusters (determined by clustering)
        unique_clusters = sorted(set(cluster_labels))
        self.n_clients = len(unique_clusters)

        print(f"\nCreating {self.n_clients} clients from {len(unique_clusters)} clusters")

        # 5. Direct mapping: cluster_i -> client_i
        # Create mapping from original cluster labels to sequential client IDs
        cluster_to_client = {
            cluster_id: client_idx for client_idx, cluster_id in enumerate(unique_clusters)
        }

        # Get indices for each client
        client_train_indices = {
            f"client_{client_idx}": np.where(
                cluster_labels == cluster_id
            )[0].tolist()
            for cluster_id, client_idx in cluster_to_client.items()
        }

        # 6. Build splits
        splits: DatasetSplits = {"clients": {}, "global_val": None}

        # Handle validation set
        if self.valset_type == "global" and self.val_dataset is not None:
            # Global validation set (all clients share)
            splits["global_val"] = self._load_dataset_samples(self.val_dataset)

            # Clients have no local validation
            for client_id, train_indices in client_train_indices.items():
                splits["clients"][client_id] = ClientSplit(
                    train=self._load_dataset_samples(self.train_dataset, train_indices),
                    val=[],
                )

        elif self.valset_type == "local" and self.val_dataset is not None:
            # Split validation set across clients proportionally
            val_indices = list(range(len(self.val_dataset)))
            rng = np.random.RandomState(self.seed)
            rng.shuffle(val_indices)

            # Calculate split points
            client_train_sizes = [
                len(indices) for indices in client_train_indices.values()
            ]
            total_train = sum(client_train_sizes)
            val_split_points = []
            cumsum = 0
            for size in client_train_sizes[:-1]:
                cumsum += int(len(val_indices) * size / total_train)
                val_split_points.append(cumsum)

            val_splits = np.split(val_indices, val_split_points)

            for idx, (client_id, train_indices) in enumerate(
                client_train_indices.items()
            ):
                splits["clients"][client_id] = ClientSplit(
                    train=self._load_dataset_samples(self.train_dataset, train_indices),
                    val=self._load_dataset_samples(
                        self.val_dataset, val_splits[idx].tolist()
                    ),
                )

        else:
            # No validation set
            for client_id, train_indices in client_train_indices.items():
                splits["clients"][client_id] = ClientSplit(
                    train=self._load_dataset_samples(self.train_dataset, train_indices),
                    val=[],
                )

        return splits
