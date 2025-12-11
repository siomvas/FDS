"""
Debug script to test VLM splitter with different alpha values.

This script runs the VLM splitter with α=0.1 and α=10.0 and compares
the resulting partitions to identify where the bug is.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from data_partitioning.dataset import BaseDataset
import numpy as np


def test_alpha_effect():
    """Test that different α values produce different partitions."""

    # Configuration
    dataset_name = "tiny_imagenet"
    base_path = "/path/to/gtai/FDS/"
    cluster_labels_dir = str(Path(__file__).parent.parent / "data_partitioning" / "legacy_vlm" / "data" / "tiny_imagenet")

    n_clients = 3
    seed = 42

    # Load dataset
    print("Loading dataset...")
    train_dataset = BaseDataset(dataset_name, base_path, "train")
    print(f"  Train samples: {len(train_dataset)}\n")

    # Test with α=0.1
    print("=" * 70)
    print("TESTING WITH α=0.1 (High Heterogeneity)")
    print("=" * 70)

    splitter_low = MultivariateDirichletSplitter(
        train_dataset=train_dataset,
        val_dataset=None,
        n_clients=n_clients,
        heterogeneity=0.1,
        cluster_labels_dir=cluster_labels_dir,
        valset_type="none",
        seed=seed,
        min_samples_per_client=100,
    )

    splits_low = splitter_low.get_client_splits()

    # Get sample counts
    samples_per_client_low = [len(splits_low["clients"][f"client_{i}"]["train"]) for i in range(n_clients)]
    print(f"\nSamples per client (α=0.1): {samples_per_client_low}")
    print(f"  Std: {np.std(samples_per_client_low):.2f}")

    # Test with α=10.0
    print("\n" + "=" * 70)
    print("TESTING WITH α=10.0 (Low Heterogeneity)")
    print("=" * 70)

    splitter_high = MultivariateDirichletSplitter(
        train_dataset=train_dataset,
        val_dataset=None,
        n_clients=n_clients,
        heterogeneity=10.0,
        cluster_labels_dir=cluster_labels_dir,
        valset_type="none",
        seed=seed,
        min_samples_per_client=100,
    )

    splits_high = splitter_high.get_client_splits()

    # Get sample counts
    samples_per_client_high = [len(splits_high["clients"][f"client_{i}"]["train"]) for i in range(n_clients)]
    print(f"\nSamples per client (α=10.0): {samples_per_client_high}")
    print(f"  Std: {np.std(samples_per_client_high):.2f}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"α=0.1  samples per client: {samples_per_client_low} (std={np.std(samples_per_client_low):.2f})")
    print(f"α=10.0 samples per client: {samples_per_client_high} (std={np.std(samples_per_client_high):.2f})")

    # Check if they're identical
    if samples_per_client_low == samples_per_client_high:
        print("\n⚠️  WARNING: Sample distributions are IDENTICAL!")
        print("This confirms the bug - α parameter is not having an effect.")
    else:
        print("\n✓ Sample distributions are DIFFERENT (expected behavior)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_alpha_effect()
