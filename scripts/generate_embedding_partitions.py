#!/usr/bin/env python3
"""
Generate embedding-based data partitions with configurable components.

This script provides a user-friendly CLI for creating federated learning data
partitions using the embedding-based partitioning pipeline. Users can configure
all components individually or use a JSON config file.

Usage Examples:

  # Using individual CLI arguments:
  python scripts/generate_embedding_partitions.py \
    --dataset tiny_imagenet \
    --base_path /path/to/data \
    --extractor dinov2-base \
    --pooling cls \
    --clustering hdbscan \
    --n_clients 10 \
    --output splits.json

  # Using a JSON config file:
  python scripts/generate_embedding_partitions.py \
    --config configs/my_partition_config.json \
    --dataset tiny_imagenet \
    --base_path /path/to/data \
    --output splits.json

  # Using a preset with custom parameters:
  python scripts/generate_embedding_partitions.py \
    --preset high_quality \
    --dataset tiny_imagenet \
    --base_path /path/to/data \
    --n_clients 20 \
    --output splits.json

Component Options:

  Extractors (--extractor):
    - dinov2-small, dinov2-base, dinov2-large, dinov2-giant
    - clip-base-patch32, clip-base-patch16, clip-large-patch14
    - siglip-base-patch16-224, siglip-base-patch16-384, siglip-large-patch16-256
    - mae-base, mae-large, mae-huge
    - inception_v3
    - raw_pixel_resizer-64, raw_pixel_resizer-128
    - sparse_random_projection_resizer-64-1000
    - UAE-default
    - wavelet_scattering-224-3

  Pooling (--pooling):
    - cls (CLS token)
    - mean (average all tokens)
    - max (max over tokens)
    - gem (generalized mean, use --pooling_p to set p)
    - lastk (last K layers, use --pooling_k and --pooling_agg)
    - spectral (spectral sharpening)

  Clustering (--clustering):
    - hdbscan (use --min_cluster_size, --min_samples)
    - kmeans (use --n_clusters)
    - knn (k-NN graph, use --knn_neighbors)
    - spectral (use --n_clusters, --spectral_affinity)

  Presets (--preset):
    - default: DINOv2-base + CLS + HDBSCAN (balanced)
    - fast: DINOv2-small + CLS + KMeans (fastest)
    - high_quality: DINOv2-large + GeM + HDBSCAN (best quality)
    - spectral: DINOv2-base + Spectral pooling + Spectral clustering
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from partitioning.embedding_splitter import EmbeddingSplitter
from partitioning.embedding_utils.config import (
    EmbeddingConfig,
    ModelConfig,
    PoolingConfig,
    ClusteringConfig,
    get_preset_config,
)
from data_loading.dataset import BaseDataset


def parse_args():
    """Parse command-line arguments for embedding-based partitioning.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following groups:

        - Input/Output: dataset, base_path, output, save_metadata, save_embeddings
        - Configuration: config, preset, save_config
        - Partitioning: n_clients, heterogeneity, valset_type, seed
        - Components: extractor, pooling, clustering
        - Extractor params: batch_size, num_workers, device, proxy
        - Pooling params: pooling_p, pooling_k, pooling_agg, spectral_k
        - Clustering params: min_cluster_size, min_samples, n_clusters, etc.
        - Cache: cache_dir, no_cache
        - Assignment: assignment_strategy, handle_noise
    """
    parser = argparse.ArgumentParser(
        description="Generate embedding-based data partitions with configurable components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/Output
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., tiny_imagenet, cifar10, food101)",
    )
    io_group.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path to dataset directory",
    )
    io_group.add_argument(
        "--output",
        type=str,
        default="embedding_splits.json",
        help="Output path for splits JSON file (default: embedding_splits.json)",
    )
    io_group.add_argument(
        "--save_metadata",
        type=str,
        help="Optional path to save clustering metadata JSON",
    )
    io_group.add_argument(
        "--save_embeddings",
        type=str,
        help="Optional path to save embeddings NPZ file",
    )

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file (overrides individual component args)",
    )
    config_group.add_argument(
        "--preset",
        type=str,
        choices=["default", "fast", "high_quality", "spectral"],
        help="Use a preset configuration (can be customized with other args)",
    )
    config_group.add_argument(
        "--save_config",
        type=str,
        help="Save the final configuration to this JSON file",
    )

    # Partitioning Parameters
    partition_group = parser.add_argument_group("Partitioning Parameters")
    partition_group.add_argument(
        "--n_clients",
        type=int,
        default=10,
        help="Number of federated learning clients (default: 10)",
    )
    partition_group.add_argument(
        "--heterogeneity",
        type=float,
        default=0.5,
        help="Heterogeneity level 0.0-1.0 (default: 0.5). Lower=more IID, Higher=more non-IID",
    )
    partition_group.add_argument(
        "--valset_type",
        type=str,
        choices=["global", "local", "none"],
        default="global",
        help="Validation set type (default: global)",
    )
    partition_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Component Selection
    component_group = parser.add_argument_group("Component Selection")
    component_group.add_argument(
        "--extractor",
        type=str,
        help="Embedding extractor (e.g., dinov2-base, clip-base-patch16)",
    )
    component_group.add_argument(
        "--pooling",
        type=str,
        help="Pooling strategy (cls, mean, max, gem, lastk, spectral)",
    )
    component_group.add_argument(
        "--clustering",
        type=str,
        help="Clustering method (hdbscan, kmeans, knn, spectral)",
    )

    # Extractor Parameters
    extractor_group = parser.add_argument_group("Extractor Parameters")
    extractor_group.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)",
    )
    extractor_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )
    extractor_group.add_argument(
        "--device",
        type=str,
        help="Device for computation (cuda, cpu, or cuda:0, cuda:1, etc.)",
    )
    extractor_group.add_argument(
        "--proxy",
        type=str,
        help="HTTP proxy for model downloads (e.g., http://proxy:3128)",
    )

    # Pooling Parameters
    pooling_group = parser.add_argument_group("Pooling Parameters")
    pooling_group.add_argument(
        "--pooling_p",
        type=float,
        default=3.0,
        help="GeM pooling parameter p (default: 3.0, for --pooling gem)",
    )
    pooling_group.add_argument(
        "--pooling_k",
        type=int,
        default=4,
        help="Number of last layers for LastK pooling (default: 4)",
    )
    pooling_group.add_argument(
        "--pooling_agg",
        type=str,
        choices=["mean", "concat", "weighted"],
        default="mean",
        help="Aggregation method for LastK pooling (default: mean)",
    )
    pooling_group.add_argument(
        "--spectral_k",
        type=int,
        default=50,
        help="Number of eigenvectors for spectral pooling (default: 50)",
    )

    # Clustering Parameters
    clustering_group = parser.add_argument_group("Clustering Parameters")
    clustering_group.add_argument(
        "--min_cluster_size",
        type=int,
        help="Minimum cluster size for HDBSCAN (auto-computed if not set)",
    )
    clustering_group.add_argument(
        "--min_samples",
        type=int,
        help="Minimum samples for HDBSCAN (defaults to min_cluster_size)",
    )
    clustering_group.add_argument(
        "--n_clusters",
        type=int,
        help="Number of clusters for KMeans/Spectral (required for these methods)",
    )
    clustering_group.add_argument(
        "--knn_neighbors",
        type=int,
        default=15,
        help="Number of neighbors for KNN graph clustering (default: 15)",
    )
    clustering_group.add_argument(
        "--spectral_affinity",
        type=str,
        choices=["rbf", "nearest_neighbors"],
        default="rbf",
        help="Affinity type for spectral clustering (default: rbf)",
    )

    # Cache Settings
    cache_group = parser.add_argument_group("Cache Settings")
    cache_group.add_argument(
        "--cache_dir",
        type=str,
        default="./.embedding_cache",
        help="Directory for caching embeddings (default: ./.embedding_cache)",
    )
    cache_group.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable embedding caching",
    )

    # Assignment Strategy
    assignment_group = parser.add_argument_group("Assignment Strategy")
    assignment_group.add_argument(
        "--assignment_strategy",
        type=str,
        choices=["round_robin", "random", "size_balanced"],
        default="round_robin",
        help="How to assign clusters to clients (default: round_robin)",
    )
    assignment_group.add_argument(
        "--handle_noise",
        type=str,
        choices=["distribute", "ignore", "separate_client"],
        default="distribute",
        help="How to handle HDBSCAN noise points (default: distribute)",
    )

    return parser.parse_args()


def build_config_from_args(args) -> EmbeddingConfig:
    """Build EmbeddingConfig from parsed arguments.

    Constructs the configuration from one of three sources (in priority order):

    1. JSON config file (--config)
    2. Preset (--preset)
    3. Individual CLI arguments (--extractor, --pooling, --clustering)

    CLI arguments can override values from JSON or preset sources.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments from :func:`parse_args`.

    Returns
    -------
    EmbeddingConfig
        Complete configuration for the embedding partitioning pipeline.

    Raises
    ------
    ValueError
        If neither --config, --preset, nor all required component args
        (--extractor, --pooling, --clustering) are provided.
    ValueError
        If --clustering is kmeans/spectral but --n_clusters is not specified.
    """
    # If JSON config provided, load it as base
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = EmbeddingConfig.from_json(args.config)
        base_source = "JSON file"
    # If preset provided, use it as base
    elif args.preset:
        print(f"Using preset configuration: {args.preset}")
        config = get_preset_config(args.preset, n_clients=args.n_clients)
        base_source = f"preset '{args.preset}'"
    # Otherwise, build from scratch
    else:
        if not args.extractor:
            raise ValueError(
                "Must specify --extractor, --preset, or --config. "
                "Example: --extractor dinov2-base"
            )
        if not args.pooling:
            raise ValueError(
                "Must specify --pooling, --preset, or --config. "
                "Example: --pooling cls"
            )
        if not args.clustering:
            raise ValueError(
                "Must specify --clustering, --preset, or --config. "
                "Example: --clustering hdbscan"
            )

        # Parse extractor
        if "-" in args.extractor:
            model_type, variant = args.extractor.split("-", 1)
        else:
            model_type = args.extractor
            variant = "default"

        # Build pooling params
        pooling_params = {}
        if args.pooling == "gem":
            pooling_params["p"] = args.pooling_p
        elif args.pooling == "lastk":
            pooling_params["k"] = args.pooling_k
            pooling_params["aggregation"] = args.pooling_agg
        elif args.pooling == "spectral":
            pooling_params["k"] = args.spectral_k

        # Build clustering params
        clustering_params = {}
        if args.clustering == "hdbscan":
            if args.min_cluster_size:
                clustering_params["min_cluster_size"] = args.min_cluster_size
            if args.min_samples:
                clustering_params["min_samples"] = args.min_samples
        elif args.clustering in ["kmeans", "spectral"]:
            if not args.n_clusters:
                raise ValueError(f"{args.clustering} requires --n_clusters")
            clustering_params["n_clusters"] = args.n_clusters
            if args.clustering == "spectral":
                clustering_params["affinity"] = args.spectral_affinity
        elif args.clustering == "knn":
            clustering_params["n_neighbors"] = args.knn_neighbors

        config = EmbeddingConfig(
            model=ModelConfig(
                model_type=model_type,
                variant=variant,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                proxy=args.proxy,
            ),
            pooling=PoolingConfig(
                strategy=args.pooling,
                params=pooling_params,
            ),
            clustering=ClusteringConfig(
                method=args.clustering,
                params=clustering_params,
            ),
            n_clients=args.n_clients,
            cache_dir=args.cache_dir,
            cache_enabled=not args.no_cache,
            assignment_strategy=args.assignment_strategy,
            handle_noise=args.handle_noise,
        )
        base_source = "CLI arguments"

    # Override with CLI arguments (if provided)
    overrides = []

    if args.n_clients != 10:  # Only override if not default
        config.n_clients = args.n_clients
        if base_source != "CLI arguments":
            overrides.append(f"n_clients={args.n_clients}")

    if args.extractor and base_source != "CLI arguments":
        if "-" in args.extractor:
            model_type, variant = args.extractor.split("-", 1)
        else:
            model_type = args.extractor
            variant = "default"
        config.model.model_type = model_type
        config.model.variant = variant
        overrides.append(f"extractor={args.extractor}")

    if args.pooling and base_source != "CLI arguments":
        config.pooling.strategy = args.pooling
        overrides.append(f"pooling={args.pooling}")

    if args.clustering and base_source != "CLI arguments":
        config.clustering.method = args.clustering
        overrides.append(f"clustering={args.clustering}")

    if args.batch_size != 32:
        config.model.batch_size = args.batch_size
        if base_source != "CLI arguments":
            overrides.append(f"batch_size={args.batch_size}")

    if args.device:
        config.model.device = args.device
        if base_source != "CLI arguments":
            overrides.append(f"device={args.device}")

    if args.cache_dir != "./.embedding_cache":
        config.cache_dir = args.cache_dir
        if base_source != "CLI arguments":
            overrides.append(f"cache_dir={args.cache_dir}")

    if args.no_cache:
        config.cache_enabled = False
        if base_source != "CLI arguments":
            overrides.append("cache_enabled=False")

    # Print configuration source
    if overrides:
        print(f"Base configuration from {base_source}, overridden: {', '.join(overrides)}")
    else:
        print(f"Configuration from {base_source}")

    return config


def main():
    """Main entry point for the embedding partitioning CLI.

    Orchestrates the full partitioning workflow:

    1. Parse command-line arguments
    2. Build configuration (from config/preset/args)
    3. Load train/val datasets
    4. Create EmbeddingSplitter with configuration
    5. Generate client splits via embedding clustering
    6. Save splits JSON and optional metadata/embeddings

    The script prints progress information and a summary of the generated
    partitions including samples per client statistics.
    """
    args = parse_args()

    print("=" * 80)
    print("EMBEDDING-BASED DATA PARTITIONING")
    print("=" * 80)

    # Build configuration
    config = build_config_from_args(args)

    # Print configuration summary
    print("\nConfiguration:")
    print(config)

    # Save configuration if requested
    if args.save_config:
        config.to_json(args.save_config)
        print(f"\n✓ Configuration saved to {args.save_config}")

    # Load datasets
    print(f"\nLoading datasets from {args.base_path}...")
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    print(f"  Training samples: {len(train_dataset)}")

    val_dataset = None
    if args.valset_type in ["global", "local"]:
        try:
            val_dataset = BaseDataset(args.dataset, args.base_path, "val")
            print(f"  Validation samples: {len(val_dataset)}")
        except Exception as e:
            print(f"  Warning: Could not load validation set: {e}")
            print(f"  Continuing with valset_type='none'")
            args.valset_type = "none"

    # Create splitter
    print(f"\nCreating embedding splitter...")
    splitter = EmbeddingSplitter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_clients=args.n_clients,
        heterogeneity=args.heterogeneity,
        valset_type=args.valset_type,
        config=config,
        seed=args.seed,
        dataset_name=args.dataset,
        base_path=args.base_path,
    )

    # Generate splits
    print("\n" + "=" * 80)
    print("GENERATING PARTITIONS")
    print("=" * 80)
    splits = splitter.get_client_splits()

    # Print summary
    print("\n" + "=" * 80)
    print("PARTITION SUMMARY")
    print("=" * 80)
    print(f"Number of clients: {args.n_clients}")

    samples_per_client = [
        len(splits["clients"][f"client_{i}"]["train"])
        for i in range(args.n_clients)
    ]
    print(f"Samples per client: min={min(samples_per_client)}, "
          f"max={max(samples_per_client)}, "
          f"mean={sum(samples_per_client)/len(samples_per_client):.1f}")

    if splits["global_val"]:
        print(f"Global validation samples: {len(splits['global_val'])}")

    # Save splits
    print(f"\nSaving splits to {args.output}...")
    splitter.store_client_splits(args.output)
    print(f"✓ Splits saved successfully")

    # Save optional outputs
    if args.save_metadata:
        print(f"Saving metadata to {args.save_metadata}...")
        splitter.save_cluster_metadata(args.save_metadata)
        print(f"✓ Metadata saved successfully")

    if args.save_embeddings:
        print(f"Saving embeddings to {args.save_embeddings}...")
        splitter.save_embeddings(args.save_embeddings)
        print(f"✓ Embeddings saved successfully")

    print("\n" + "=" * 80)
    print("✓ PARTITIONING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
