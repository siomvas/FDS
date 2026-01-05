"""Embedding-based data splitting using foundation models and clustering.

This module implements semantic data partitioning by:
1. Extracting image embeddings using foundation models (DINOv2, CLIP, SigLIP)
2. Applying pooling strategies to aggregate spatial features
3. Clustering embeddings to group semantically similar samples
4. Assigning clusters to clients for heterogeneous splits

This creates natural, semantically-meaningful heterogeneity where clients
receive data with similar visual characteristics, mimicking real-world
scenarios where different clients have different types of data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import json

from .base_splitter import BaseDatasetSplitter, DatasetSplits, ClientSplit
from data_loading.dataset import BaseDataset
from embedding.embedding_extractors import (
    EmbeddingExtractor,
    DINOv2Extractor,
    CLIPExtractor,
    SigLIPExtractor,
)
from .embedding_utils.pooling_strategies import (
    PoolingStrategy,
    get_pooling_strategy,
)
from .embedding_utils.clustering_methods import (
    ClusteringMethod,
    get_clustering_method,
)
from embedding.cache_manager import CacheManager
from .embedding_utils.config import EmbeddingConfig


class EmbeddingSplitter(BaseDatasetSplitter):
    """Splits datasets using embedding-based clustering for federated learning.

    This splitter creates semantically-meaningful heterogeneous partitions by:

    1. Extracting image embeddings using foundation models (DINOv2, CLIP, SigLIP)
    2. Applying pooling strategies to aggregate spatial features into vectors
    3. Clustering embeddings to group semantically similar samples
    4. Assigning clusters to clients based on the chosen strategy

    The resulting partitions mimic real-world federated scenarios where different
    clients have different types of data with similar visual characteristics.

    Parameters
    ----------
    train_dataset : BaseDataset
        Training dataset containing image paths and labels.
    val_dataset : BaseDataset or None
        Validation dataset. If None, no validation split is created.
    n_clients : int
        Number of federated learning clients to partition data across.
    heterogeneity : float
        Degree of heterogeneity in [0.0, 1.0]. Controls clustering granularity:

        - 0.0: Larger clusters, more IID-like distribution
        - 1.0: Smaller clusters, high non-IID (each client gets distinct data)
    valset_type : str
        How to handle validation data:

        - "global": All clients share the same validation set
        - "local": Split validation proportionally across clients
        - "none": No validation set for clients
    config : EmbeddingConfig
        Configuration object specifying model, pooling, clustering, and cache settings.
        Use :func:`get_preset_config` for convenient presets.
    seed : int, optional
        Random seed for reproducibility. Affects cluster assignment randomization.
    dataset_name : str, optional
        Dataset name for efficient cache key computation. Enables dataset-mode caching
        where embeddings can be reused across different subsets of the same dataset.
    base_path : str, optional
        Base path for path normalization in caching. Ensures cache compatibility
        across different user directories on shared storage (e.g., HPC).

    Attributes
    ----------
    config : EmbeddingConfig
        The configuration used for this splitter.
    embeddings : np.ndarray or None
        Extracted embeddings after :meth:`get_client_splits` is called.
    cluster_labels : np.ndarray or None
        Cluster assignments after :meth:`get_client_splits` is called.
    cluster_info : dict or None
        Clustering metadata (num_clusters, sizes, etc.) after splitting.

    Examples
    --------
    Basic usage with a preset configuration:

    >>> from partitioning.embedding_splitter import EmbeddingSplitter
    >>> from partitioning.embedding_utils.config import get_preset_config
    >>> from data_loading.dataset import BaseDataset
    >>>
    >>> train_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "train")
    >>> val_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "val")
    >>>
    >>> config = get_preset_config("default", n_clients=10)
    >>> splitter = EmbeddingSplitter(
    ...     train_dataset=train_dataset,
    ...     val_dataset=val_dataset,
    ...     n_clients=10,
    ...     heterogeneity=0.5,
    ...     valset_type="global",
    ...     config=config,
    ...     seed=42,
    ... )
    >>> splits = splitter.get_client_splits()
    >>> splitter.store_client_splits("embedding_splits.json")

    Custom configuration for high-quality partitioning:

    >>> from partitioning.embedding_utils.config import (
    ...     EmbeddingConfig, ModelConfig, PoolingConfig, ClusteringConfig
    ... )
    >>>
    >>> config = EmbeddingConfig(
    ...     model=ModelConfig(model_type="dinov2", variant="large"),
    ...     pooling=PoolingConfig(strategy="gem", params={"p": 3.0}),
    ...     clustering=ClusteringConfig(method="hdbscan", params={"min_cluster_size": 10}),
    ...     n_clients=20,
    ...     cache_enabled=True,
    ... )
    >>> splitter = EmbeddingSplitter(
    ...     train_dataset=train_dataset,
    ...     val_dataset=None,
    ...     n_clients=20,
    ...     heterogeneity=0.8,
    ...     valset_type="none",
    ...     config=config,
    ... )

    See Also
    --------
    get_preset_config : Get pre-configured embedding pipeline setups.
    EmbeddingConfig : Complete configuration class for the pipeline.
    DirichletSplitter : Alternative label-based non-IID splitting.
    MultivariateDirichletSplitter : VLM-based semantic splitting.

    Notes
    -----
    - Embeddings are automatically cached for re-use across runs.
    - The heterogeneity parameter affects HDBSCAN's min_cluster_size when not
      explicitly set in the config.
    - Noise samples from HDBSCAN (label=-1) are handled according to
      ``config.handle_noise`` setting.
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset],
        n_clients: int,
        heterogeneity: float,
        valset_type: str,
        config: EmbeddingConfig,
        seed: Optional[int] = None,
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
    ):
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=n_clients,
            heterogeneity=heterogeneity,
            valset_type=valset_type,
            seed=seed,
        )

        self.config = config
        self.dataset_name = dataset_name
        self.base_path = base_path

        # Initialize components
        self.extractor: Optional[EmbeddingExtractor] = None
        self.pooling: Optional[PoolingStrategy] = None
        self.clustering: Optional[ClusteringMethod] = None
        self.cache_manager = CacheManager(
            cache_dir=config.cache_dir,
            enabled=config.cache_enabled,
        )

        # Clustering results
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_info: Optional[Dict] = None

    def _initialize_extractor(self) -> EmbeddingExtractor:
        """Initialize the embedding extractor based on config.

        Creates the appropriate extractor class (DINOv2, CLIP, or SigLIP)
        based on the model configuration.

        Returns
        -------
        EmbeddingExtractor
            Initialized extractor ready for embedding extraction.

        Raises
        ------
        ValueError
            If the model_type in config is not recognized.
        """
        model_cfg = self.config.model

        if model_cfg.model_type.lower() == "dinov2":
            return DINOv2Extractor(
                variant=model_cfg.variant,
                device=model_cfg.device,
                batch_size=model_cfg.batch_size,
                num_workers=model_cfg.num_workers,
                proxy=model_cfg.proxy,
            )
        elif model_cfg.model_type.lower() == "clip":
            return CLIPExtractor(
                variant=model_cfg.variant,
                device=model_cfg.device,
                batch_size=model_cfg.batch_size,
                num_workers=model_cfg.num_workers,
            )
        elif model_cfg.model_type.lower() == "siglip":
            return SigLIPExtractor(
                variant=model_cfg.variant,
                device=model_cfg.device,
                batch_size=model_cfg.batch_size,
                num_workers=model_cfg.num_workers,
            )
        else:
            raise ValueError(f"Unknown model type: {model_cfg.model_type}")

    def _initialize_pooling(self) -> PoolingStrategy:
        """Initialize the pooling strategy based on config.

        Returns
        -------
        PoolingStrategy
            Initialized pooling strategy (CLS, Mean, Max, GeM, etc.).
        """
        return get_pooling_strategy(
            self.config.pooling.strategy,
            **self.config.pooling.params,
        )

    def _initialize_clustering(self) -> ClusteringMethod:
        """Initialize the clustering method based on config.

        For HDBSCAN, automatically computes min_cluster_size based on
        heterogeneity if not explicitly set in config.

        Returns
        -------
        ClusteringMethod
            Initialized clustering method (HDBSCAN, KMeans, KNN, Spectral).
        """
        # Adjust clustering parameters based on heterogeneity
        params = self.config.clustering.params.copy()

        # For HDBSCAN, heterogeneity affects min_cluster_size
        if self.config.clustering.method.lower() == "hdbscan":
            if "min_cluster_size" not in params:
                # Lower heterogeneity → larger clusters
                # Higher heterogeneity → smaller clusters
                base_size = len(self.train_dataset) // (self.n_clients * 3)
                min_size = max(5, int(base_size * (1.0 - self.heterogeneity)))
                params["min_cluster_size"] = min_size

        return get_clustering_method(
            self.config.clustering.method,
            **params,
        )

    def _extract_embeddings(self) -> np.ndarray:
        """Extract embeddings from training images using the configured model.

        Checks the cache first and returns cached embeddings if available.
        Otherwise, extracts fresh embeddings, applies pooling, and caches
        the results.

        Returns
        -------
        np.ndarray
            Pooled embeddings of shape (n_samples, embed_dim).
        """
        # Get image paths from dataset
        image_paths = [str(self.train_dataset[i][0]) for i in range(len(self.train_dataset))]

        # Initialize extractor to get cache-friendly model name
        if self.extractor is None:
            self.extractor = self._initialize_extractor()

        # Check cache using extractor's model_name (no slashes)
        cache_key_params = {
            "model_name": self.extractor.model_name,
            "pooling_strategy": self.config.pooling.strategy,
            **self.config.pooling.params,
        }

        cached_embeddings = self.cache_manager.get(
            image_paths=image_paths,
            dataset_name=self.dataset_name,
            base_path=self.base_path,
            **cache_key_params,
        )

        if cached_embeddings is not None:
            print("✓ Loaded embeddings from cache")
            return cached_embeddings

        # Extract fresh embeddings
        print(f"Extracting embeddings using {self.config.model.model_type}...")

        if self.extractor is None:
            self.extractor = self._initialize_extractor()

        # Determine if we need all tokens for pooling
        need_all_tokens = self.config.pooling.strategy.lower() in [
            "mean",
            "max",
            "gem",
        ]

        raw_embeddings = self.extractor.extract_embeddings(
            images=image_paths,
            return_all_tokens=need_all_tokens,
        )

        # Apply pooling
        if self.pooling is None:
            self.pooling = self._initialize_pooling()

        embeddings = self.pooling.pool(raw_embeddings)

        # Cache the results
        self.cache_manager.put(
            embeddings=embeddings,
            image_paths=image_paths,
            dataset_name=self.dataset_name,
            base_path=self.base_path,
            **cache_key_params,
        )

        print(f"✓ Extracted embeddings: {embeddings.shape}")
        return embeddings

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster the embeddings using the configured clustering method.

        Parameters
        ----------
        embeddings : np.ndarray
            Pooled embeddings of shape (n_samples, embed_dim).

        Returns
        -------
        np.ndarray
            Cluster labels of shape (n_samples,). Labels are integers >= 0,
            or -1 for noise/outliers (HDBSCAN only).
        """
        if self.clustering is None:
            self.clustering = self._initialize_clustering()

        print(f"Clustering with {self.config.clustering.method}...")
        cluster_labels = self.clustering.cluster(embeddings)
        self.cluster_info = self.clustering.get_cluster_info()

        print(f"✓ Found {self.cluster_info.get('num_clusters', 0)} clusters")
        if "noise_count" in self.cluster_info:
            print(f"  Noise samples: {self.cluster_info['noise_count']}")

        return cluster_labels

    def _assign_clusters_to_clients(
        self,
        cluster_labels: np.ndarray,
    ) -> Dict[str, List[int]]:
        """Assign cluster members to clients using the configured strategy.

        Distributes clusters to clients based on ``config.assignment_strategy``:

        - "round_robin": Assign clusters cyclically to clients
        - "random": Randomly assign clusters to clients
        - "size_balanced": Assign to minimize client size variance

        Noise samples (label=-1 from HDBSCAN) are handled according to
        ``config.handle_noise``: "distribute", "ignore", or "separate_client".

        Parameters
        ----------
        cluster_labels : np.ndarray
            Cluster assignment for each sample, shape (n_samples,).

        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping client_id (e.g., "client_0") to list of
            sample indices belonging to that client.
        """
        # Get unique clusters (excluding noise if present)
        unique_clusters = sorted(set(cluster_labels))
        has_noise = -1 in unique_clusters

        if has_noise:
            noise_indices = np.where(cluster_labels == -1)[0]
            unique_clusters = [c for c in unique_clusters if c != -1]
        else:
            noise_indices = np.array([])

        # Create cluster to indices mapping
        cluster_to_indices = {}
        for cluster_id in unique_clusters:
            cluster_to_indices[cluster_id] = np.where(cluster_labels == cluster_id)[0]

        # Assign clusters to clients based on strategy
        client_indices: Dict[str, List[int]] = {
            f"client_{i}": [] for i in range(self.n_clients)
        }

        strategy = self.config.assignment_strategy

        if strategy == "round_robin":
            # Distribute clusters round-robin
            for idx, cluster_id in enumerate(unique_clusters):
                client_id = f"client_{idx % self.n_clients}"
                client_indices[client_id].extend(cluster_to_indices[cluster_id].tolist())

        elif strategy == "random":
            # Randomly assign clusters to clients
            rng = np.random.RandomState(self.seed)
            shuffled_clusters = rng.permutation(unique_clusters)
            for idx, cluster_id in enumerate(shuffled_clusters):
                client_id = f"client_{idx % self.n_clients}"
                client_indices[client_id].extend(cluster_to_indices[cluster_id].tolist())

        elif strategy == "size_balanced":
            # Assign clusters to balance client sizes
            cluster_sizes = [(c, len(cluster_to_indices[c])) for c in unique_clusters]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)

            client_sizes = [0] * self.n_clients

            for cluster_id, size in cluster_sizes:
                # Assign to client with fewest samples
                min_client_idx = np.argmin(client_sizes)
                client_id = f"client_{min_client_idx}"
                client_indices[client_id].extend(cluster_to_indices[cluster_id].tolist())
                client_sizes[min_client_idx] += size

        else:
            raise ValueError(f"Unknown assignment strategy: {strategy}")

        # Handle noise samples
        if len(noise_indices) > 0:
            handle_noise = self.config.handle_noise

            if handle_noise == "distribute":
                # Distribute noise round-robin
                for idx, sample_idx in enumerate(noise_indices):
                    client_id = f"client_{idx % self.n_clients}"
                    client_indices[client_id].append(int(sample_idx))

            elif handle_noise == "separate_client":
                # Give all noise to one client
                client_indices["client_0"].extend(noise_indices.tolist())

            # "ignore" means noise samples are dropped (not recommended)

        return client_indices

    def _compute_client_splits(self) -> DatasetSplits:
        """Compute client splits using embedding clustering.

        This is the main method that orchestrates the embedding-based
        partitioning pipeline:

        1. Extract embeddings from training images
        2. Cluster embeddings into groups
        3. Assign clusters to clients
        4. Build the final splits structure with train/val data

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
        # Extract embeddings and cluster
        self.embeddings = self._extract_embeddings()
        self.cluster_labels = self._cluster_embeddings(self.embeddings)

        # Assign clusters to clients
        client_train_indices = self._assign_clusters_to_clients(self.cluster_labels)

        # Build splits
        splits: DatasetSplits = {
            "clients": {},
            "global_val": None,
        }

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
            client_train_sizes = [len(indices) for indices in client_train_indices.values()]
            total_train = sum(client_train_sizes)
            val_split_points = []
            cumsum = 0
            for size in client_train_sizes[:-1]:
                cumsum += int(len(val_indices) * size / total_train)
                val_split_points.append(cumsum)

            val_splits = np.split(val_indices, val_split_points)

            for idx, (client_id, train_indices) in enumerate(client_train_indices.items()):
                splits["clients"][client_id] = ClientSplit(
                    train=self._load_dataset_samples(self.train_dataset, train_indices),
                    val=self._load_dataset_samples(self.val_dataset, val_splits[idx].tolist()),
                )

        else:
            # No validation set
            for client_id, train_indices in client_train_indices.items():
                splits["clients"][client_id] = ClientSplit(
                    train=self._load_dataset_samples(self.train_dataset, train_indices),
                    val=[],
                )

        return splits

    def save_cluster_metadata(self, filepath: str) -> None:
        """Save clustering metadata to JSON file.

        Saves configuration, cluster information, and partitioning parameters
        for reproducibility and analysis.

        Parameters
        ----------
        filepath : str
            Path to save the JSON metadata file.

        Raises
        ------
        ValueError
            If called before :meth:`get_client_splits` has been run.

        Examples
        --------
        >>> splitter = EmbeddingSplitter(...)
        >>> splits = splitter.get_client_splits()
        >>> splitter.save_cluster_metadata("cluster_info.json")
        """
        if self.cluster_info is None:
            raise ValueError("No clustering info available. Run get_client_splits() first.")

        metadata = {
            "config": self.config.to_dict(),
            "cluster_info": self.cluster_info,
            "n_clients": self.n_clients,
            "heterogeneity": self.heterogeneity,
            "seed": self.seed,
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_embeddings(self, filepath: str) -> None:
        """Save extracted embeddings and cluster labels to NPZ file.

        Saves embeddings for visualization, debugging, or transfer to other
        pipelines without re-extraction.

        Parameters
        ----------
        filepath : str
            Path to save the compressed NPZ file.

        Raises
        ------
        ValueError
            If called before :meth:`get_client_splits` has been run.

        Examples
        --------
        >>> splitter = EmbeddingSplitter(...)
        >>> splits = splitter.get_client_splits()
        >>> splitter.save_embeddings("embeddings.npz")
        >>>
        >>> # Later, load for visualization
        >>> data = np.load("embeddings.npz")
        >>> embeddings = data["embeddings"]
        >>> cluster_labels = data["cluster_labels"]
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run get_client_splits() first.")

        np.savez_compressed(
            filepath,
            embeddings=self.embeddings,
            cluster_labels=self.cluster_labels,
        )
