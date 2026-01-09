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
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_loading.dataset import BaseDataset
from partitioning.embedding_splitter import EmbeddingSplitter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embedding-based federated learning splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., tiny_imagenet, food101, ADE20K, etc.)",
    )
    dataset_group.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path to dataset directory",
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
        "--output",
        type=str,
        default="embedding_splits.json",
        help="Output path for splits JSON file (default: embedding_splits.json)",
    )
    output_group.add_argument(
        "--cache_dir",
        type=str,
        default="/path/to/gtai/FDS/embedding_cache",
        help="Directory for caching embeddings",
    )
    output_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


class SubsetDataset:
    """Wrapper to create a subset of a BaseDataset for mock testing."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

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
        
        # Set pooling to 'lastk' if not already specified as such and raise warning
        if args.pooling != "lastk":
            print("⚠️  Warning: Overriding pooling strategy to 'lastk' for k_last_cls or k_last_all token extraction")
            args.pooling = "lastk"

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


def main():
    args = parse_args()

    print("=" * 70)
    print("Embedding-based Federated Learning Split Generation")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"Base path: {args.base_path}")
    print(f"Extractor: {args.extractor}")
    print(f"Token extraction: {args.tokens}")
    print(f"Pooling: {args.pooling}")
    print(f"Normalization: {args.normalize}")
    print(f"Dimensionality reduction: {args.dim_reduction}")
    print(f"Clustering: {args.clustering}")
    print(f"Validation type: {args.valset_type}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    print(f"  Train samples: {len(train_dataset)}")

    val_dataset = None
    if args.valset_type in ["global", "local"]:
        val_dataset = BaseDataset(args.dataset, args.base_path, "val")
        print(f"  Val samples: {len(val_dataset)}")

    # Apply mock mode if enabled
    if args.mock:
        MOCK_SAMPLES = 1000
        print(f"\n⚠️  MOCK MODE: Limiting to {MOCK_SAMPLES} samples for quick testing")

        train_indices = list(range(min(MOCK_SAMPLES, len(train_dataset))))
        train_dataset = SubsetDataset(train_dataset, train_indices)
        print(f"  Mock train samples: {len(train_dataset)}")

        if val_dataset is not None:
            val_size = min(MOCK_SAMPLES // 5, len(val_dataset))  # 20% of mock size
            val_indices = list(range(val_size))
            val_dataset = SubsetDataset(val_dataset, val_indices)
            print(f"  Mock val samples: {len(val_dataset)}")

    # Build parameter dicts
    token_params = build_token_params(args)
    pooling_params = build_pooling_params(args)
    preprocessing_params = build_preprocessing_params(args)
    clustering_params = build_clustering_params(args)
    
    print("\nToken extraction params:", token_params)
    print(f"Pooling params: {pooling_params}")
    print(f"Preprocessing params: {preprocessing_params}")
    print(f"\nClustering params: {clustering_params}")
    print()

    # Enforce cache disabling in mock mode
    use_cache = not args.mock
    num_workers = 0 if args.mock else args.num_workers

    if args.mock:
        print("⚠️  MOCK MODE: Caching DISABLED (enforced)")
        print("   Reason: Mock mode uses subset of data, incompatible with full caching")
        print()

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
        num_workers=num_workers,
    )

    # Compute splits
    print("Computing client splits...")
    print()
    splits = splitter.get_client_splits()

    # Print summary
    print()
    print("=" * 70)
    print("Split Summary")
    print("=" * 70)
    summary = splitter.get_split_summary()
    print(f"Number of clients: {summary['n_clients']}")
    print(f"Total training samples: {summary['total_train_samples']}")
    print(f"Validation type: {summary['valset_type']}")

    if summary['valset_type'] == 'global' and 'global_val_samples' in summary:
        print(f"Global validation samples: {summary['global_val_samples']}")
    elif summary['valset_type'] == 'local':
        print(f"Total local validation samples: {summary['total_val_samples']}")

    print("\nPer-client sample counts:")
    for client_id, sizes in sorted(summary['client_sizes'].items()):
        train_count = sizes['train_samples']
        val_count = sizes['val_samples']
        if val_count > 0:
            print(f"  {client_id}: {train_count} train, {val_count} val")
        else:
            print(f"  {client_id}: {train_count} train")

    # Save splits
    print(f"\nSaving splits to {args.output}...")
    splitter.store_client_splits(args.output, format="json")

    # Validate splits
    print("\nValidating splits...")
    try:
        splitter.validate_splits()
        print("✓ All validations passed")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("✓ Split generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
