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
    """
    Splits datasets using embedding-based clustering.

    Extracts embeddings from images using foundation models, applies
    pooling strategies, clusters embeddings, and assigns clusters to clients.
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
    ):
        """
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            n_clients: Number of clients
            heterogeneity: Degree of heterogeneity (controls clustering granularity)
            valset_type: "local", "global", or "none"
            config: EmbeddingConfig with model, pooling, clustering settings
            seed: Random seed
        """
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=n_clients,
            heterogeneity=heterogeneity,
            valset_type=valset_type,
            seed=seed,
        )

        self.config = config

        # Initialize components
        self.extractor: Optional[EmbeddingExtractor] = None
        self.pooling: Optional[PoolingStrategy] = None
        self.clustering: Optional[ClusteringMethod] = None
        self.cache_manager = CacheManager(
            cache_dir=config.cache.cache_dir,
            enabled=config.cache.enabled,
        )

        # Clustering results
        self.embeddings: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_info: Optional[Dict] = None

    def _initialize_extractor(self) -> EmbeddingExtractor:
        """Initialize the embedding extractor."""
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
        """Initialize the pooling strategy."""
        return get_pooling_strategy(
            self.config.pooling.strategy,
            **self.config.pooling.params,
        )

    def _initialize_clustering(self) -> ClusteringMethod:
        """Initialize the clustering method."""
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
        """Extract embeddings from training images."""
        # Get image paths from dataset
        image_paths = [str(self.train_dataset[i][0]) for i in range(len(self.train_dataset))]

        # Check cache
        cache_key_params = {
            "model_name": f"{self.config.model.model_type}/{self.config.model.variant}",
            "pooling_strategy": self.config.pooling.strategy,
            **self.config.pooling.params,
        }

        cached_embeddings = self.cache_manager.get(
            image_paths=image_paths,
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
            **cache_key_params,
        )

        print(f"✓ Extracted embeddings: {embeddings.shape}")
        return embeddings

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster the embeddings."""
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
        """
        Assign cluster members to clients.

        Args:
            cluster_labels: Cluster assignment for each sample

        Returns:
            Dictionary mapping client_id to list of sample indices
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

        strategy = self.config.assignment.strategy

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
            handle_noise = self.config.assignment.handle_noise

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
        """
        Compute client splits using embedding clustering.

        Returns:
            DatasetSplits with client train/val data and optional global val
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
        """
        Save clustering metadata to JSON file.

        Args:
            filepath: Path to save metadata
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
        """
        Save extracted embeddings to npz file.

        Args:
            filepath: Path to save embeddings
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run get_client_splits() first.")

        np.savez_compressed(
            filepath,
            embeddings=self.embeddings,
            cluster_labels=self.cluster_labels,
        )
