#!/usr/bin/env python3
"""Generate embedding-based data splits for federated learning.

This script creates semantically-meaningful heterogeneous partitions by:
1. Extracting embeddings using foundation models (via ModelRegistry)
2. Optionally applying pooling strategies
3. Optional dimensionality reduction and normalization of the embeddings
4. Clustering embeddings
5. Directly mapping clusters to clients (1 cluster = 1 client)

For HDBSCAN/KNN: Number of clients determined automatically by clustering
For KMeans/Spectral: User specifies n_clusters (becomes n_clients)

Examples
--------
HDBSCAN with DINOv2 (auto-determines n_clients):
    python scripts/generate_embedding_partitions.py \\
        --dataset tiny_imagenet \\
        --base_path /path/to/data \\
        --extractor dinov2-base \\
        --clustering hdbscan \\
        --min_cluster_size 100 \\
        --tokens cls \\
        --output hdbscan_splits.json

KMeans with CLIP (fixed 10 clients):
    python scripts/generate_embedding_partitions.py \\
        --dataset tiny_imagenet \\
        --base_path /path/to/data \\
        --extractor clip-base-patch16 \\
        --clustering kmeans \\
        --n_clusters 10 \\
        --tokens k_last_cls \\
        --lastk_k 4 \\
        --layer_aggregation concat \\
        --pooling max \\
        --output kmeans_10_splits.json

With GeM pooling:
    python scripts/generate_embedding_partitions.py \\
        --dataset tiny_imagenet \\
        --base_path /path/to/data \\
        --extractor dinov2-large \\
        --clustering hdbscan \\
        --tokens all_last \\
        --pooling gem \\
        --gem_p 2 \\
        --output gem_pooling_splits.json
"""

import faulthandler
import signal
faulthandler.register(signal.SIGUSR1)

import argparse
import logging
import os
from pathlib import Path
import numpy as np

from data_loading.dataset import BaseDataset
from partitioning.embedding_splitter import EmbeddingSplitter
from utils.cli_common import add_standard_cli_args

log = logging.getLogger(__name__)


def parse_args():
    default_cache_dir = os.environ.get("FDS_CACHE_DIR", str(Path.cwd() / "embedding_cache"))
    parser = argparse.ArgumentParser(
        description="Generate embedding-based federated learning splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset")
    add_standard_cli_args(
        dataset_group,
        include_dataset=True,
        dataset_required=True,
        include_base_path=True,
        base_path_required=True,
        include_output=True,
        output_required=False,
        output_default="embedding_splits.json",
        output_help="Output path for splits JSON file.",
        include_seed=True,
        seed_default=42,
        include_dry_run=True,
    )
    dataset_group.add_argument(
        "--valset_type",
        type=str,
        default="global",
        choices=["global", "local", "none"],
        help="Validation set type: global (shared), local (per-client), or none",
    )
    dataset_group.add_argument(
        "--mock",
        action="store_true",
        help="Mock mode: use only 1000 samples for quick testing",
    )
    dataset_group.add_argument(
        "--subset_size_train",
        type=int,
        default=None,
        help="Optional deterministic subset size for train split (audit-focused runs).",
    )
    dataset_group.add_argument(
        "--subset_size_val",
        type=int,
        default=None,
        help="Optional deterministic subset size for val split (audit-focused runs).",
    )
    dataset_group.add_argument(
        "--subset_seed",
        type=int,
        default=None,
        help="Seed for deterministic subset selection (defaults to --seed).",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--extractor",
        type=str,
        required=True,
        help="Extractor name from ModelRegistry (e.g., dinov2-base, clip-large-patch14, mae-base)",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for computation (cuda, cpu, cuda:0, etc.). Auto-detects if not specified.",
    )
    model_group.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for embedding extraction (default: 100)",
    )
    model_group.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 4). Set to 0 for single-threaded loading.",
    )
    model_group.add_argument(
        "--proxy",
        type=str,
        default=None,
        help=(
            "Optional proxy forwarded to extractors. "
            "Set to an empty string to explicitly disable proxy use."
        ),
    )
    
    # Token extraction arguments
    token_group = parser.add_argument_group("Token Extraction")
    token_group.add_argument(
        "--tokens",
        type=str,
        default="cls",
        choices=["cls", "all_last", "k_last_cls", "k_last_all"],
        help="Token extraction strategy: cls (CLS token), all_last (all tokens from last layer), k_last_cls (CLS tokens from last K layers), k_last_all (all tokens from last K layers) (default: cls)",
    )
    token_group.add_argument(
        "--lastk_k",
        type=int,
        default=None,
        help="Number of last layers to use for k_last_cls or k_last_all token extraction (required if using these strategies)",
    )
    token_group.add_argument(
        "--layer_aggregation",
        type=str,
        default=None,
        choices=["mean", "concat", "weighted"],
        help="Layer aggregation method for k_last_cls or k_last_all (required if using these strategies)",
    )


    # Pooling arguments
    pooling_group = parser.add_argument_group("Pooling")
    pooling_group.add_argument(
        "--pooling",
        type=str,
        default=None,
        choices=["mean", "max", "gem", "lastk", "spectral"],
        help="Pooling strategy for aggregating embeddings (required when not using CLS tokens directly)",
    )
    pooling_group.add_argument(
        "--gem_p",
        type=float,
        default=None,
        help="Power parameter for GeM pooling (required if using GeM pooling)",
    )
    pooling_group.add_argument(
        "--spectral_k",
        type=int,
        default=50,
        help="Number of eigenvectors for spectral pooling (default: 50)",
    )
    pooling_group.add_argument(
        "--spectral_sigma",
        type=float,
        default=None,
        help="RBF bandwidth for spectral pooling (default: auto via median heuristic)",
    )
    pooling_group.add_argument(
        "--lastk_pooling",
        type=str,
        default=None,
        choices=["mean", "max", "gem", "spectral"],
        help="Token aggregation method for lastk pooling (required if using lastk pooling)",
    )

    # Preprocessing arguments
    preprocessing_group = parser.add_argument_group("Preprocessing")
    preprocessing_group.add_argument(
        "--normalize",
        action="store_true",
        help="L2 normalize embeddings before clustering",
    )
    preprocessing_group.add_argument(
        "--dim_reduction",
        type=str,
        default=None,
        choices=["pca", "umap"], # pca not yet implemented
        help="Dimensionality reduction method before clustering",
    )
    preprocessing_group.add_argument(
        "--dr_dims",
        type=int,
        default=50,
        help="Number of dimensions after dimensionality reduction (default: 50)",
    )
    preprocessing_group.add_argument(
        "--umap_neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP dimensionality reduction (default: 15)",
    )
    preprocessing_group.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.0,
        help="Minimum distance parameter for UMAP (default: 0.0)",
    )
    preprocessing_group.add_argument(
        "--umap_metric",
        type=str,
        default="euclidean",
        help="Distance metric for UMAP (default: euclidean)",
    )

    # Clustering arguments
    clustering_group = parser.add_argument_group("Clustering")
    clustering_group.add_argument(
        "--clustering",
        type=str,
        required=True,
        choices=["hdbscan", "kmeans", "knn", "spectral"],
        help="Clustering method",
    )

    # HDBSCAN parameters
    clustering_group.add_argument(
        "--min_cluster_size",
        type=int,
        default=100,
        help="Minimum cluster size for HDBSCAN (default: 15)",
    )
    clustering_group.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="Minimum samples for HDBSCAN core points (default: same as min_cluster_size)",
    )
    clustering_group.add_argument(
        "--hdbscan_metric",
        type=str,
        default=None,
        help="Distance metric for HDBSCAN (required if using HDBSCAN)",
    )
    clustering_group.add_argument(
        "--hdbscan_method",
        type=str,
        default="eom",
        choices=["eom", "leaf"],
        help="Cluster selection method for HDBSCAN (default: eom)",
    )

    # KMeans/Spectral parameters
    clustering_group.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters for KMeans/Spectral (required for these methods)",
    )

    # KNN parameters
    clustering_group.add_argument(
        "--knn_neighbors",
        type=int,
        default=15,
        help="Number of neighbors for KNN clustering (default: 15)",
    )
    clustering_group.add_argument(
        "--knn_metric",
        type=str,
        default="euclidean",
        help="Distance metric for KNN (default: euclidean)",
    )

    # Spectral parameters
    clustering_group.add_argument(
        "--spectral_affinity",
        type=str,
        default="rbf",
        choices=["rbf", "nearest_neighbors"],
        help="Affinity type for Spectral clustering (default: rbf)",
    )

    # Output and caching arguments
    output_group = parser.add_argument_group("Output and Caching")
    output_group.add_argument(
        "--cache_dir",
        type=str,
        default=default_cache_dir,
        help=f"Directory for caching embeddings (default: {default_cache_dir})",
    )

    return parser.parse_args()


class SubsetDataset:
    """Wrapper to create a subset of a BaseDataset for mock testing."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.dataset_name = getattr(dataset, "dataset_name", None)
        self.requested_dataset_name = getattr(dataset, "requested_dataset_name", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def _build_subset_indices(dataset_len: int, subset_size: int, seed: int) -> list[int]:
    if subset_size <= 0:
        raise ValueError("--subset_size_train/--subset_size_val must be > 0 when provided.")
    size = min(int(subset_size), int(dataset_len))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(dataset_len, size=size, replace=False)
    return sorted(indices.tolist())

def build_preprocessing_params(args):
    """Build preprocessing parameters dict from CLI args."""
    params = {}

    if args.normalize:
        params["normalization"] = "l2"

    if args.dim_reduction == "umap":
        params["dim_reduction"] = "umap"
        params["dr_dims"] = args.dr_dims
        params["umap_neighbors"] = args.umap_neighbors
        params["umap_min_dist"] = args.umap_min_dist
        params["umap_metric"] = args.umap_metric

    return params


def build_clustering_params(args):
    """Build clustering parameters dict from CLI args."""
    params = {}

    if args.clustering == "hdbscan":
        params["min_cluster_size"] = args.min_cluster_size
        params["hdbscan_method"] = args.hdbscan_method
        if args.min_samples:
            params["min_samples"] = args.min_samples
        if not args.hdbscan_metric:
            raise ValueError("--hdbscan_metric is required for HDBSCAN clustering")
        params["hdbscan_metric"] = args.hdbscan_metric

    elif args.clustering == "kmeans":
        if not args.n_clusters:
            raise ValueError("--n_clusters is required for kmeans clustering")
        params["n_clusters"] = args.n_clusters
        params["random_state"] = args.seed

    elif args.clustering == "knn":
        params["n_neighbors"] = args.knn_neighbors
        params["metric"] = args.knn_metric

    elif args.clustering == "spectral":
        if not args.n_clusters:
            raise ValueError("--n_clusters is required for spectral clustering")
        params["n_clusters"] = args.n_clusters
        params["affinity"] = args.spectral_affinity
        params["random_state"] = args.seed

    return params

def build_token_params(args):
    """Build token extraction parameters dict from CLI args."""
    params = {}

    if args.tokens in ["k_last_cls", "k_last_all"]:
        if args.lastk_k is None:
            raise ValueError("--lastk_k is required for k_last_cls or k_last_all token extraction")
        if args.layer_aggregation is None:
            raise ValueError("--layer_aggregation is required for k_last_cls or k_last_all token extraction")
        params["lastk_k"] = args.lastk_k
        params["layer_aggregation"] = args.layer_aggregation
        if args.pooling != "lastk":
            raise ValueError(
                f"--pooling must be 'lastk' when using {args.tokens} token extraction."
            )

    return params

def build_pooling_params(args):
    """Build pooling parameters dict from CLI args."""
    params = {}

    if args.pooling == "gem":
        if args.gem_p is None:
            raise ValueError("--gem_p is required for GeM pooling")
        params["p"] = args.gem_p
    elif args.pooling == "spectral":
        params["k"] = args.spectral_k
        if args.spectral_sigma is not None:
            params["sigma"] = args.spectral_sigma
    
    elif args.pooling == "lastk":
        if args.lastk_pooling is None:
            raise ValueError("--lastk_pooling is required for lastk pooling")
        params["token_aggregation"] = args.lastk_pooling

    return params


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    log.info("=" * 70)
    log.info("Embedding-based Federated Learning Split Generation")
    log.info("=" * 70)
    log.info("Dataset=%s base_path=%s extractor=%s clustering=%s", args.dataset, args.base_path, args.extractor, args.clustering)
    log.info(
        "tokens=%s pooling=%s normalize=%s dim_reduction=%s valset_type=%s seed=%s output=%s",
        args.tokens,
        args.pooling,
        args.normalize,
        args.dim_reduction,
        args.valset_type,
        args.seed,
        args.output,
    )

    # Build parameter dicts
    token_params = build_token_params(args)
    pooling_params = build_pooling_params(args)
    preprocessing_params = build_preprocessing_params(args)
    clustering_params = build_clustering_params(args)

    log.info("Token extraction params: %s", token_params)
    log.info("Pooling params: %s", pooling_params)
    log.info("Preprocessing params: %s", preprocessing_params)
    log.info("Clustering params: %s", clustering_params)

    if args.dry_run:
        log.info("[dry-run] Configuration validated. Skipping dataset loading and split generation.")
        return 0

    # Load datasets
    log.info("Loading datasets...")
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    log.info("Train samples: %s", len(train_dataset))

    val_dataset = None
    if args.valset_type in ["global", "local"]:
        val_dataset = BaseDataset(args.dataset, args.base_path, "val")
        log.info("Val samples: %s", len(val_dataset))

    if args.mock and (args.subset_size_train is not None or args.subset_size_val is not None):
        raise ValueError("--mock cannot be combined with --subset_size_train/--subset_size_val.")

    # Apply deterministic subset mode for tiered audit workflows.
    if not args.mock and (args.subset_size_train is not None or args.subset_size_val is not None):
        subset_seed = args.seed if args.subset_seed is None else args.subset_seed
        if args.subset_size_train is not None:
            train_indices = _build_subset_indices(len(train_dataset), args.subset_size_train, subset_seed)
            train_dataset = SubsetDataset(train_dataset, train_indices)
            log.info("Subset train samples: %s", len(train_dataset))
        if val_dataset is not None and args.subset_size_val is not None:
            val_indices = _build_subset_indices(len(val_dataset), args.subset_size_val, subset_seed)
            val_dataset = SubsetDataset(val_dataset, val_indices)
            log.info("Subset val samples: %s", len(val_dataset))

    # Apply mock mode if enabled
    if args.mock:
        MOCK_SAMPLES = 1000
        log.warning("MOCK MODE enabled: limiting to %s samples", MOCK_SAMPLES)

        train_indices = list(range(min(MOCK_SAMPLES, len(train_dataset))))
        train_dataset = SubsetDataset(train_dataset, train_indices)
        log.info("Mock train samples: %s", len(train_dataset))

        if val_dataset is not None:
            val_size = min(MOCK_SAMPLES // 5, len(val_dataset))  # 20% of mock size
            val_indices = list(range(val_size))
            val_dataset = SubsetDataset(val_dataset, val_indices)
            log.info("Mock val samples: %s", len(val_dataset))

    # Enforce cache disabling in mock mode
    use_cache = not args.mock
    num_workers = 0 if args.mock else args.num_workers

    if args.mock:
        log.warning("MOCK MODE: caching disabled (subset data is incompatible with cache reuse).")

    # Create splitter
    splitter = EmbeddingSplitter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        extractor_name=args.extractor,
        clustering_method=args.clustering,
        clustering_params=clustering_params,
        token_extraction_strategy=args.tokens,
        token_params=token_params,
        pooling_strategy=args.pooling,
        pooling_params=pooling_params,
        preprocessing_params=preprocessing_params,
        valset_type=args.valset_type,
        seed=args.seed,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset,
        base_path=args.base_path,
        device=args.device,
        batch_size=args.batch_size,
        use_cache=use_cache,
        proxy=args.proxy,
        num_workers=num_workers,
    )

    # Compute splits
    log.info("Computing client splits...")
    splitter.get_client_splits()

    # Print summary
    log.info("=" * 70)
    log.info("Split Summary")
    log.info("=" * 70)
    summary = splitter.get_split_summary()
    log.info("Number of clients: %s", summary["n_clients"])
    log.info("Total training samples: %s", summary["total_train_samples"])
    log.info("Validation type: %s", summary["valset_type"])

    if summary['valset_type'] == 'global' and 'global_val_samples' in summary:
        log.info("Global validation samples: %s", summary["global_val_samples"])
    elif summary['valset_type'] == 'local':
        log.info("Total local validation samples: %s", summary["total_val_samples"])

    log.info("Per-client sample counts:")
    for client_id, sizes in sorted(summary['client_sizes'].items()):
        train_count = sizes['train_samples']
        val_count = sizes['val_samples']
        if val_count > 0:
            log.info("%s: %s train, %s val", client_id, train_count, val_count)
        else:
            log.info("%s: %s train", client_id, train_count)

    # Save splits
    log.info("Saving splits to %s", args.output)
    splitter.store_client_splits(args.output, format="json")

    # Validate splits
    log.info("Validating splits...")
    try:
        splitter.validate_splits()
        log.info("All validations passed.")
    except ValueError as e:
        log.error("Validation failed: %s", e)
        return 1

    log.info("=" * 70)
    log.info("Split generation complete.")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
