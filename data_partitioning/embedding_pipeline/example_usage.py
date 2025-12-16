"""
Example usage of the embedding-based partitioning pipeline.

This script demonstrates how to:
1. Configure the embedding pipeline
2. Create an EmbeddingSplitter
3. Generate client splits
4. Save results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dataset import BaseDataset
from embedding_pipeline import (
    EmbeddingSplitter,
    EmbeddingConfig,
    ModelConfig,
    PoolingConfig,
    ClusteringConfig,
    get_preset_config,
)


def example_basic_usage():
    """Basic usage with preset configuration."""
    print("=" * 80)
    print("Example 1: Basic Usage with Preset Configuration")
    print("=" * 80)

    # Load your dataset
    train_dataset = BaseDataset(
        root="./data/cifar10/train",
        label_map_path="./data/cifar10/label_map.json",
    )

    val_dataset = BaseDataset(
        root="./data/cifar10/val",
        label_map_path="./data/cifar10/label_map.json",
    )

    # Use a preset configuration
    config = get_preset_config(preset="default", n_clients=10)

    # Create splitter
    splitter = EmbeddingSplitter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_clients=10,
        heterogeneity=0.5,  # Moderate heterogeneity
        valset_type="global",
        config=config,
        seed=42,
    )

    # Generate splits
    splits = splitter.get_client_splits()

    # Save splits
    splitter.store_client_splits("embedding_split_default.json")

    # Save metadata
    splitter.save_cluster_metadata("embedding_split_metadata.json")

    # Print summary
    summary = splitter.get_split_summary()
    print("\nSplit Summary:")
    for key, value in summary.items():
        if key != "client_sizes":
            print(f"  {key}: {value}")

    print("\nClient Sizes:")
    for client_id, sizes in summary["client_sizes"].items():
        print(f"  {client_id}: {sizes['train_samples']} train, {sizes['val_samples']} val")


def example_custom_configuration():
    """Custom configuration with specific parameters."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)

    # Load dataset
    train_dataset = BaseDataset(
        root="./data/food101/train",
        label_map_path="./data/food101/label_map.json",
    )

    val_dataset = BaseDataset(
        root="./data/food101/val",
        label_map_path="./data/food101/label_map.json",
    )

    # Create custom configuration
    config = EmbeddingConfig(
        model=ModelConfig(
            model_type="dinov2",
            variant="large",
            batch_size=16,
            num_workers=4,
        ),
        pooling=PoolingConfig(
            strategy="gem",
            params={"p": 3.0, "eps": 1e-6},
        ),
        clustering=ClusteringConfig(
            method="hdbscan",
            params={
                "min_cluster_size": 20,
                "min_samples": 10,
                "metric": "cosine",
            },
        ),
        n_clients=50,
        valset_type="local",
        output_dir="./embedding_splits/food101",
    )

    # Save configuration for reproducibility
    config.to_json("embedding_config_food101.json")

    # Create splitter
    splitter = EmbeddingSplitter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_clients=50,
        heterogeneity=0.8,  # High heterogeneity
        valset_type="local",
        config=config,
        seed=42,
    )

    # Generate splits
    splits = splitter.get_client_splits()

    # Save everything
    splitter.store_client_splits("embedding_split_food101.json")
    splitter.save_cluster_metadata("embedding_split_food101_metadata.json")
    splitter.save_embeddings("embedding_split_food101_embeddings.npz")

    print("\n✓ Splits saved successfully!")


def example_spectral_sharpening():
    """Using spectral sharpening pooling."""
    print("\n" + "=" * 80)
    print("Example 3: Spectral Sharpening")
    print("=" * 80)

    # Use preset spectral configuration
    config = get_preset_config(preset="spectral", n_clients=20)

    # Or customize spectral parameters
    config.pooling.params["k"] = 100  # Use top-100 eigenvectors
    config.pooling.params["sigma"] = None  # Auto-detect bandwidth

    train_dataset = BaseDataset(
        root="./data/tiny/train",
        label_map_path="./data/tiny/label_map.json",
    )

    splitter = EmbeddingSplitter(
        train_dataset=train_dataset,
        val_dataset=None,
        n_clients=20,
        heterogeneity=0.6,
        valset_type="none",
        config=config,
        seed=42,
    )

    splits = splitter.get_client_splits()
    splitter.store_client_splits("embedding_split_spectral.json")

    print("\n✓ Spectral sharpening splits created!")


def example_comparison():
    """Compare different models and strategies."""
    print("\n" + "=" * 80)
    print("Example 4: Comparing Different Configurations")
    print("=" * 80)

    train_dataset = BaseDataset(
        root="./data/cifar10/train",
        label_map_path="./data/cifar10/label_map.json",
    )

    # Compare different presets
    presets = ["default", "fast", "high_quality"]

    for preset_name in presets:
        print(f"\nTesting preset: {preset_name}")

        config = get_preset_config(preset=preset_name, n_clients=10)

        splitter = EmbeddingSplitter(
            train_dataset=train_dataset,
            val_dataset=None,
            n_clients=10,
            heterogeneity=0.5,
            valset_type="none",
            config=config,
            seed=42,
        )

        splits = splitter.get_client_splits()
        splitter.store_client_splits(f"embedding_split_{preset_name}.json")

        summary = splitter.get_split_summary()
        print(f"  Total train samples: {summary['total_train_samples']}")
        print(f"  Clusters found: {splitter.cluster_info.get('num_clusters', 'N/A')}")


def example_loading_existing_config():
    """Load configuration from file."""
    print("\n" + "=" * 80)
    print("Example 5: Loading Configuration from File")
    print("=" * 80)

    # First create and save a configuration
    config = get_preset_config("default", n_clients=10)
    config.to_json("my_config.json")

    # Later, load it back
    loaded_config = EmbeddingConfig.from_json("my_config.json")

    print("Loaded configuration:")
    print(loaded_config)


if __name__ == "__main__":
    # Run examples (comment out as needed)

    # Basic examples
    # example_basic_usage()
    # example_custom_configuration()
    # example_spectral_sharpening()
    # example_comparison()
    example_loading_existing_config()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
