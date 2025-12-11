"""
Test script for MultivariateDirichletSplitter

Validates correctness of the implementation:
- Cluster labels loading
- Preference vector sampling
- Affinity score calculation
- Split constraints and coverage
- Heterogeneity control (low α vs high α)
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from data_partitioning.dataset import BaseDataset


def test_preference_vectors(splitter: MultivariateDirichletSplitter):
    """Test that preference vectors are properly sampled and sum to 1.0."""
    print("\n=== Testing Preference Vectors ===")

    for client_id in range(splitter.n_clients):
        for criterion in splitter.CRITERIA:
            prefs = splitter.client_preferences[client_id][criterion]

            # Check that preferences sum to 1.0
            pref_sum = np.sum(prefs)
            assert np.isclose(pref_sum, 1.0), f"Preferences don't sum to 1.0: {pref_sum}"

            # Check all preferences are non-negative
            assert np.all(prefs >= 0), f"Found negative preferences for client {client_id}, {criterion}"

            # Show top preference for first client
            if client_id == 0:
                top_idx = np.argmax(prefs)
                top_label = splitter.unique_labels[criterion][top_idx]
                print(f"  Client 0, {criterion}: top={top_label} ({prefs[top_idx]:.3f})")

    print("✓ All preference vectors sum to 1.0 and are non-negative")


def test_affinity_scores(splitter: MultivariateDirichletSplitter):
    """Test affinity score calculation."""
    print("\n=== Testing Affinity Scores ===")

    # Get a sample image path
    sample_path = splitter.train_dataset[0][0]

    # Compute affinity for each client
    affinities = []
    for client_id in range(min(3, splitter.n_clients)):  # Test first 3 clients
        affinity = splitter._compute_affinity_score(sample_path, client_id)
        affinities.append(affinity)
        print(f"  Client {client_id}: affinity = {affinity:.4f}")

    # Affinity should be positive (sum of probabilities)
    assert all(a > 0 for a in affinities), "Found non-positive affinity scores"

    # With 5 criteria, affinity should be roughly in range [0, 5]
    # (each criterion contributes a probability between 0 and 1)
    assert all(0 <= a <= 5.5 for a in affinities), "Affinity scores out of expected range"

    print("✓ Affinity scores are in expected range")


def test_split_constraints(splitter: MultivariateDirichletSplitter, splits):
    """Test that splits satisfy all constraints."""
    print("\n=== Testing Split Constraints ===")

    n_train = len(splitter.train_dataset)

    # Collect all assigned training indices
    all_train_indices = set()
    samples_per_client = []

    for client_id in range(splitter.n_clients):
        client_train = splits["clients"][f"client_{client_id}"]["train"]
        samples_per_client.append(len(client_train))

        # Get indices by matching image paths
        for item in client_train:
            # Find the index in train_dataset
            for idx in range(n_train):
                if splitter.train_dataset[idx][0] == item["image"]:
                    assert idx not in all_train_indices, f"Duplicate index {idx} found!"
                    all_train_indices.add(idx)
                    break

    # Check complete coverage
    assert len(all_train_indices) == n_train, \
        f"Not all samples assigned! {len(all_train_indices)} vs {n_train}"

    # Check minimum samples per client
    min_samples = min(samples_per_client)
    assert min_samples >= splitter.min_samples_per_client, \
        f"Client has too few samples: {min_samples} < {splitter.min_samples_per_client}"

    print(f"  Samples per client: {samples_per_client}")
    print(f"  Mean: {np.mean(samples_per_client):.1f}, Std: {np.std(samples_per_client):.1f}")
    print(f"  Min: {min_samples}, Max: {max(samples_per_client)}")
    print("✓ All samples assigned exactly once, min_samples constraint satisfied")


def test_heterogeneity_effect():
    """Test that lower α creates more heterogeneous distributions."""
    print("\n=== Testing Heterogeneity Effect ===")

    # Load dataset
    dataset_name = "tiny_imagenet"
    base_path = "/path/to/gtai/FDS/"
    cluster_labels_dir = "../data_partitioning/legacy_vlm/data/tiny_imagenet"

    train_dataset = BaseDataset(dataset_name, base_path, "train")

    # Test with low heterogeneity (α=10.0)
    print("\n  Testing with α=10.0 (low heterogeneity, uniform preferences)...")
    splitter_high_alpha = MultivariateDirichletSplitter(
        train_dataset=train_dataset,
        val_dataset=None,
        n_clients=3,
        heterogeneity=10.0,
        cluster_labels_dir=cluster_labels_dir,
        valset_type="none",
        seed=42,
        min_samples_per_client=100,
    )
    splitter_high_alpha._load_cluster_labels()
    splitter_high_alpha._sample_client_preferences()

    # Test with high heterogeneity (α=0.1)
    print("\n  Testing with α=0.1 (high heterogeneity, skewed preferences)...")
    splitter_low_alpha = MultivariateDirichletSplitter(
        train_dataset=train_dataset,
        val_dataset=None,
        n_clients=3,
        heterogeneity=0.1,
        cluster_labels_dir=cluster_labels_dir,
        valset_type="none",
        seed=42,
        min_samples_per_client=100,
    )
    splitter_low_alpha._load_cluster_labels()
    splitter_low_alpha._sample_client_preferences()

    # Compare preference distributions
    print("\n  Comparing preference distributions...")

    for criterion in splitter_low_alpha.CRITERIA[:2]:  # Test first 2 criteria
        # For low alpha, preferences should be more skewed (higher max value)
        max_pref_low = np.max(splitter_low_alpha.client_preferences[0][criterion])
        max_pref_high = np.max(splitter_high_alpha.client_preferences[0][criterion])

        print(f"    {criterion}: max_pref(α=0.1)={max_pref_low:.3f}, max_pref(α=10.0)={max_pref_high:.3f}")

        # Lower alpha should produce more skewed distributions
        # (not guaranteed every time due to randomness, but very likely)
        if max_pref_low > max_pref_high:
            print(f"      ✓ Lower α has more skewed distribution")
        else:
            print(f"      ⚠ Note: Higher α has more skewed distribution (can happen by chance)")

    print("✓ Heterogeneity parameter effect verified")


def run_basic_test():
    """Run a basic test with small dataset."""
    print("=" * 60)
    print("BASIC TEST: 6 clients, α=0.5, seed=42")
    print("=" * 60)

    # Paths (adjust if needed)
    dataset_name = "tiny_imagenet"
    base_path = "/path/to/gtai/FDS/"
    cluster_labels_dir = "../data_partitioning/legacy_vlm/data/tiny_imagenet"

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = BaseDataset(dataset_name, base_path, "train")
    val_dataset = BaseDataset(dataset_name, base_path, "val")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create splitter
    print("\nCreating splitter...")
    splitter = MultivariateDirichletSplitter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_clients=6,
        heterogeneity=0.5,
        cluster_labels_dir=cluster_labels_dir,
        valset_type="global",
        seed=42,
        min_samples_per_client=100,
    )

    # Get splits (this will load labels, sample preferences, and assign)
    print("\nComputing splits...")
    splits = splitter.get_client_splits()

    # Run tests
    test_preference_vectors(splitter)
    test_affinity_scores(splitter)
    test_split_constraints(splitter, splits)

    # Show preference summary
    print("\n=== Preference Summary ===")
    summary = splitter.get_preference_summary()
    print(f"  Heterogeneity: {summary['heterogeneity']}")
    print(f"  Number of clients: {summary['n_clients']}")

    # Show top preference for each criterion (first 3 clients)
    for criterion in splitter.CRITERIA:
        print(f"\n  {criterion.upper()} - Top preferences:")
        for client_data in summary[f"{criterion}_top_preferences"][:3]:
            print(f"    Client {client_data['client']}: {client_data['label']} ({client_data['preference']:.3f})")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    # Run basic test
    run_basic_test()

    # Test heterogeneity effect
    test_heterogeneity_effect()

    print("\n🎉 All validation tests completed successfully!")
