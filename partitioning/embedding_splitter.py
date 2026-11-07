"""Embedding-based data splitting using foundation models and clustering.

This module implements semantic data partitioning by:
1. Extracting image embeddings using foundation models (via ModelRegistry)
2. Applying optional pooling strategies to aggregate spatial features
3. Applying optional normalization and dimensionality reduction
4. Clustering embeddings to group semantically similar samples
5. Directly mapping clusters to clients (1 cluster = 1 client)

This creates natural, semantically-meaningful heterogeneity where each client
receives data from a distinct cluster, mimicking real-world scenarios where
different clients have different types of data.
"""

from pathlib import Path
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Any
import numpy as np
import time

try:
    import umap as _umap_module
except Exception as exc:
    umap = SimpleNamespace(UMAP=None)
    _UMAP_IMPORT_ERROR = exc
else:
    umap = _umap_module
    _UMAP_IMPORT_ERROR = None

from .base_splitter import BaseDatasetSplitter, DatasetSplits, ClientSplit
from data_loading.dataset import BaseDataset
from embedding.embedding_extractors import ModelRegistry, EmbeddingExtractor
from embedding.cache_manager import CacheManager
from .embedding_utils.pooling_strategies import get_pooling_strategy
from .embedding_utils.clustering_methods import get_clustering_method

logger = logging.getLogger(__name__)


def _require_umap():
    if getattr(umap, "UMAP", None) is None:
        raise ImportError(
            "UMAP dimensionality reduction requires the optional dependency "
            "'umap-learn' to be installed."
        ) from _UMAP_IMPORT_ERROR
    return umap

class EmbeddingSplitter(BaseDatasetSplitter):
    """Splits datasets using embedding-based clustering for federated learning.

    This splitter creates semantically-meaningful heterogeneous partitions by:

    1. Extracting image embeddings using foundation models (via ModelRegistry)
    2. Optionally applying pooling strategies to aggregate features
    3. Optionally applying normalization and dimensionality reduction
    4. Clustering embeddings to group semantically similar samples
    5. Directly mapping clusters to clients (n_clients = number of clusters)

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

        - HDBSCAN: min_cluster_size, min_samples, metric, method
        - KMeans: n_clusters (required), n_init
        - KNN: n_neighbors, metric
        - Spectral: n_clusters (required), affinity, gamma

    token_extraction_strategy : str, optional
        Which tokens to extract before pooling:

        - "cls": CLS token only
        - "all_last": All tokens from last layer
        - "k_last_all": All tokens from last K layers
        - "k_last_cls": CLS tokens from last K layers

        Default: "cls".
    
    token_params : dict, optional
        Strategy-specific token extraction parameters:
        - k_last_all / k_last_cls: k (number of layers, default 4)
        - layer_aggregation : How to aggregate layers when using last K layers:
            - "mean": Average across layers
            - "concat": Concatenate layers
            - "weighted": Weighted average (more weight to later layers)

            Default: "mean".

    pooling_strategy : str, optional
        How to aggregate token embeddings:

        - "mean": Average all tokens
        - "max": Max pooling across tokens
        - "gem": Generalized mean pooling
        - "lastk": Aggregate last K layers
        - "spectral": Spectral sharpening

        Default: "mean".

    pooling_params : dict, optional
        Strategy-specific pooling parameters:

        - gem: p (power parameter, default 3.0)
        - lastk: k (number of layers, default 4), aggregation ("mean"/"concat"/"weighted")
        - spectral: k (eigenvectors, default 50), sigma (RBF bandwidth)
    
    preprocessing_params : dict, optional
        Preprocessing parameters:
        - normalization : l2 or none
        - dim_reduction : umap or none (default "none")

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
        Batch size for embedding extraction. Default: 100.
    proxy : str, optional
        Optional proxy value forwarded to embedding extractors. Set to an empty
        string to explicitly disable proxy use in extractor download contexts.
        If None, extractor defaults apply.
    num_workers : int, optional
        Number of DataLoader worker processes for parallel data loading.
        Default: 4. Set to 0 for single-threaded loading (recommended for
        small datasets or to avoid worker process issues).

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
    ...     clustering_params={"min_cluster_size": 100},
    ...     token_extraction_strategy="cls",
    ...     valset_type="global",
    ...     seed=42,
    ... )
    >>> splits = splitter.get_client_splits()
    >>> f"Found {splitter.n_clients} clusters/clients"
    >>> splitter.store_client_splits("embedding_splits.json")

    Using KMeans with fixed number of clients:

    >>> splitter = EmbeddingSplitter(
    ...     train_dataset=train_dataset,
    ...     val_dataset=val_dataset,
    ...     extractor_name="clip-base-patch16",
    ...     clustering_method="kmeans",
    ...     clustering_params={"n_clusters": 10},
    ...     token_extraction_strategy="all_last",
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
        token_extraction_strategy: str = "cls",
        token_params: Optional[Dict] = None,
        pooling_strategy: str = "mean",
        pooling_params: Optional[Dict] = None,
        preprocessing_params: Optional[Dict] = None,
        valset_type: str = "global",
        seed: Optional[int] = None,
        cache_dir: str = "./.embedding_cache",
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 100,
        use_cache: bool = True,
        proxy: Optional[str] = None,
        num_workers: int = 0,
    ):
        """Initialize the embedding splitter."""
        # Validate token extraction strategy
        valid_extraction = [
            "cls",
            "all_last",
            "k_last_all",
            "k_last_cls",
        ]
        if token_extraction_strategy not in valid_extraction:
            raise ValueError(
                f"Invalid token extraction: {token_extraction_strategy}. "
                f"Choose from {valid_extraction}"
            )
        
        # Validate layer aggregation if using last K layers
        if token_extraction_strategy in ["k_last_all", "k_last_cls"]:
            if not isinstance(token_params, dict):
                raise ValueError(
                    "token_params must be provided as a mapping when using "
                    "'k_last_all' or 'k_last_cls' extraction."
                )
            try:
                layer_aggregation = token_params["layer_aggregation"]
            except KeyError:
                raise ValueError(
                    "token_params must include 'layer_aggregation' and 'lastk_k' "
                    "when using 'k_last_all' or 'k_last_cls' extraction."
                )
            valid_layer_agg = ["mean", "concat", "weighted"]
            if layer_aggregation not in valid_layer_agg:
                raise ValueError(
                    f"Invalid layer aggregation: {layer_aggregation}. "
                    f"Choose from {valid_layer_agg}"
                )
        else:
            token_params = token_params or {}

        # Turn off pooling if using CLS tokens only.
        if token_extraction_strategy == "cls":
            pooling_strategy = "cls"
        elif pooling_strategy is None:
            # Keep script defaults ergonomic: non-CLS token extraction falls back to mean pooling.
            pooling_strategy = "mean"

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
        self.clustering_params = {} if clustering_params is None else dict(clustering_params)
        self.token_extraction_strategy = token_extraction_strategy
        self.token_params = token_params
        self.pooling_strategy = pooling_strategy
        self.pooling_params = {} if pooling_params is None else dict(pooling_params)
        self.preprocessing_params = {} if preprocessing_params is None else dict(preprocessing_params)
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.device = device
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.proxy = proxy
        self.num_workers = num_workers

        # need to persist the umap reducer if used
        self.umap_reducer: Optional[Any] = None

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
        create_kwargs = dict(
            device=self.device,
            batch_size=self.batch_size,
            cache_manager=cache_manager,
            num_workers=self.num_workers,
        )
        # Explicit proxy is optional; if omitted, extractor defaults are used.
        if self.proxy is not None:
            create_kwargs["proxy"] = self.proxy

        return ModelRegistry.create(
            self.extractor_name,
            **create_kwargs,
        )

    def _get_embeddings(self, dataset: BaseDataset, dataset_type: str = "train") -> np.ndarray:
        """Extract and optionally pool embeddings for a specific dataset.
        
        Parameters
        ----------
        dataset : BaseDataset
            Dataset to extract embeddings from.
        dataset_type : str, optional
            Type of dataset ("train" or "val") for logging purposes.

        Returns
        -------
        np.ndarray
            Pooled embeddings of shape (n_samples, embed_dim).
        """
        extractor = self._get_extractor()
        image_paths = [
            str(dataset[i][0]) for i in range(len(dataset))
        ]

        logger.info("Extracting %s embeddings using %s...", dataset_type, self.extractor_name)
        # only print for train to reduce noise
        if dataset_type == "train":
            logger.info("Token extraction strategy: %s", self.token_extraction_strategy)
            if self.token_extraction_strategy in ["k_last_all", "k_last_cls"]:
                logger.info("Token layer aggregation: %s", self.token_params["layer_aggregation"])
                logger.info("Token last K layers: %s", self.token_params["lastk_k"])
            if self.pooling_strategy != "cls":
                logger.info("Pooling strategy: %s", self.pooling_strategy)

        # Extract embeddings (caching controlled by use_cache parameter)
            logger.info("Caching: %s", "enabled" if self.use_cache else "disabled")

        logger.info("Starting extraction for %d images.", len(image_paths))
        time_start = time.time()
        extractor_pooling_strategy = None
        extractor_pooling_params = None
        if self.pooling_strategy in {"mean", "max", "gem"}:
            extractor_pooling_strategy = self.pooling_strategy
            extractor_pooling_params = self.pooling_params

        raw_embeddings = extractor.extract_embeddings(
            images=image_paths,
            return_which_tokens=self.token_extraction_strategy,
            k_last=self.token_params.get("lastk_k", -1), # ignored if last_k not used
            use_cache=self.use_cache,
            dataset_name=self.dataset_name,
            base_path=self.base_path,
            pooling_strategy=extractor_pooling_strategy,
            pooling_params=extractor_pooling_params,
        )
        time_end = time.time()
        logger.info("Extraction completed in %.2f seconds.", time_end - time_start)
        logger.info("Raw embeddings shape: %s", raw_embeddings.shape)

        # Apply pooling if needed
        if self.pooling_strategy == "cls":
            # CLS token only, no pooling needed
            logger.debug("Using CLS token only, skipping pooling.")
            embeddings = raw_embeddings
        elif raw_embeddings.ndim == 2:
            logger.info("Extractor returned pooled embeddings directly.")
            embeddings = raw_embeddings
        else:
            if self.pooling_strategy == "lastk":
                if "token_aggregation" not in self.pooling_params:
                    raise ValueError(
                        "pooling_params must include 'token_aggregation' when pooling_strategy='lastk'."
                    )
                self.layer_aggregation = self.token_params["layer_aggregation"]
                token_agg = self.pooling_params["token_aggregation"]
                # Last K layers pooling
                logger.info(
                    "Applying Last K Layers pooling with layer aggregation %s and token aggregation %s.",
                    self.layer_aggregation,
                    token_agg,
                )
                pooler = get_pooling_strategy(
                    "lastk",
                    layer_aggregation=self.layer_aggregation,
                    token_aggregation=token_agg,
                )
            else:
                # Other pooling strategies
                logger.info("Applying %s pooling.", self.pooling_strategy)
                pooler = get_pooling_strategy(
                    self.pooling_strategy, **self.pooling_params
                )
            embeddings = pooler.pool(raw_embeddings)

        logger.info("Final embeddings shape: %s", embeddings.shape)
        return embeddings

    def _preprocess_embeddings(self, embeddings, fit=True) -> np.ndarray:
        """Apply preprocessing (dimensionality reduction, normalization) to embeddings."""
        if not self.preprocessing_params:
            raise ValueError("No preprocessing_params provided for embedding preprocessing.")
        args = self.preprocessing_params
        if args.get("normalization", "none") == "l2":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1, norms)
            logger.debug("Applied L2 normalization to embeddings.")
        
        if args.get("dim_reduction", "none") == "umap":
            if fit:
                umap_module = _require_umap()
                self.umap_reducer = umap_module.UMAP(
                    n_components=args["dr_dims"],
                    n_neighbors=args["umap_neighbors"],
                    min_dist=args["umap_min_dist"],
                    metric=args["umap_metric"],
                    # random_state=self.seed, # Uncomment for reproducibility, incompatible with n_jobs > 1 (parallelism)
                )
                logger.info(
                    "UMAP: %d -> %d (neighbors=%s, min_dist=%s)",
                    embeddings.shape[1],
                    args["dr_dims"],
                    args["umap_neighbors"],
                    args["umap_min_dist"],
                )
                embeddings = self.umap_reducer.fit_transform(embeddings)
            else:
                if self.umap_reducer is None:
                    raise ValueError("UMAP reducer not fitted yet.")
                embeddings = self.umap_reducer.transform(embeddings)
        
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
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape {embeddings.shape}"
            )
        clustering = get_clustering_method(
            self.clustering_method, **self.clustering_params
        )

        logger.info("Clustering with %s...", self.clustering_method)
        time_start = time.time()
        cluster_labels = clustering.cluster(embeddings)
        self.cluster_info_ = clustering.get_cluster_info()
        if "num_clusters" not in self.cluster_info_:
            raise RuntimeError(
                f"Clustering method '{self.clustering_method}' did not report 'num_clusters'."
            )

        num_clusters = int(self.cluster_info_["num_clusters"])
        time_end = time.time()
        logger.info("Found %d clusters in %.2f seconds.", num_clusters, time_end - time_start)

        if "noise_count" in self.cluster_info_:
            logger.info("Noise samples: %s", self.cluster_info_["noise_count"])

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

        logger.info("Assigning %d noise points to nearest clusters...", int(noise_mask.sum()))

        # Get cluster centroids
        unique_labels = [l for l in set(labels) if l != -1]
        
        # If no clusters found (all noise), return single cluster
        if not unique_labels:
            logger.warning("All points are noise; assigning all samples to cluster 0.")
            return np.zeros_like(labels)

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

        1. Extract embeddings from training set
        2. Cluster training embeddings into groups
        3. Handle noise points (assign to nearest cluster)
        4. Set n_clients = number of clusters
        5. Directly map clusters to clients (cluster_i -> client_i)
        6. Handle validation set according to valset_type
        7. Build the final splits structure with train/val data

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
        # 1. Extract training set embeddings (caching handled by extractor)
        train_embeddings = self._get_embeddings(self.train_dataset, dataset_type="train")

        # 2. Preprocess embeddings if specified
        if self.preprocessing_params:
            train_embeddings = self._preprocess_embeddings(train_embeddings, fit=True)

        # 3. Cluster Training embeddings
        cluster_labels = self._cluster(train_embeddings)
        self.cluster_labels_ = cluster_labels

        # 4. Handle noise (HDBSCAN returns -1 for noise)
        cluster_labels = self._assign_noise_to_nearest(train_embeddings, cluster_labels)

        # 5. n_clients = num_clusters (determined by clustering)
        unique_clusters = sorted(set(cluster_labels))
        self.n_clients = len(unique_clusters)

        logger.info("Creating %d clients from %d clusters.", self.n_clients, len(unique_clusters))

        # 6. Direct mapping: cluster_i -> client_i
        # Create mapping from original cluster labels to sequential client IDs
        cluster_to_client = {
            cluster_id: client_idx for client_idx, cluster_id in enumerate(unique_clusters)
        }

        # Build client train indices
        client_train_indices = {
            f"client_{client_idx}": np.where(
                cluster_labels == cluster_id
            )[0].tolist()
            for cluster_id, client_idx in cluster_to_client.items()
        }

        splits: DatasetSplits = {"clients": {}, "global_val": None}

        # 6. Handle validation logic
        if self.valset_type == "global" and self.val_dataset is not None:
            # Global validation set (all clients share original val set)
            splits["global_val"] = self._load_dataset_samples(self.val_dataset)

            # Clients have no local validation
            for client_id, train_indices in client_train_indices.items():
                splits["clients"][client_id] = ClientSplit(
                    train=self._load_dataset_samples(self.train_dataset, train_indices),
                    val=[],
                )

        elif self.valset_type == "local" and self.val_dataset is not None:
            # Local validation: split val set proportionally to train clusters
            val_embeddings = self._get_embeddings(self.val_dataset, dataset_type="val")   

            # Preprocess val embeddings if needed
            if self.preprocessing_params:
                val_embeddings = self._preprocess_embeddings(val_embeddings, fit=False)

            train_centroids = {}
            for cluster_id, client_idx in cluster_to_client.items():
                mask = cluster_labels == cluster_id
                if mask.any():
                    train_centroids[client_idx] = train_embeddings[mask].mean(axis=0)
                else:
                    train_centroids[client_idx] = np.zeros(train_embeddings.shape[1])

            # Assign Validation Samples to Nearest Client
            client_val_indices = {cid: [] for cid in client_train_indices.keys()}
            for i, val_emb in enumerate(val_embeddings):
                # Find nearest centroid
                nearest_client_idx = min(train_centroids.keys(), key=lambda c_idx: np.linalg.norm(val_emb - train_centroids[c_idx]))
                client_val_indices[f"client_{nearest_client_idx}"].append(i)

            # Build Client Splits
            for client_id, train_idx in client_train_indices.items():
                val_idx = client_val_indices.get(client_id, [])
                splits["clients"][client_id] = ClientSplit(
                    train=self._load_dataset_samples(self.train_dataset, train_idx),
                    val=self._load_dataset_samples(self.val_dataset, val_idx)
                )

        else:
            # No validation set
            for client_id, train_indices in client_train_indices.items():
                splits["clients"][client_id] = ClientSplit(
                    train=self._load_dataset_samples(self.train_dataset, train_indices),
                    val=[],
                )

        return splits
