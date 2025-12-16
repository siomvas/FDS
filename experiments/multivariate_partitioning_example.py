"""
Simple example demonstrating multivariate Dirichlet partitioning.

This script shows basic usage of the MultivariateDirichletSplitter
and how to analyze the resulting partitions.
"""

import sys
from pathlib import Path
import numpy as np
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from data_loading.dataset import BaseDataset


def analyze_client_distribution(splits, client_id: int, cluster_labels: dict):
    """Analyze the distribution of clusters for a specific client."""
    client_name = f"client_{client_id}"
    client_data = splits["clients"][client_name]["train"]

    print(f"\n{'=' * 60}")
    print(f"Client {client_id} Analysis")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(client_data)}")

    # Analyze each criterion
    for criterion in ["color", "main_object", "size", "location", "time"]:
        cluster_counter = Counter()

        for item in client_data:
            image_path = item["image"]
            cluster_label = cluster_labels[criterion].get(image_path)
            if cluster_label:
                cluster_counter[cluster_label] += 1

        print(f"\n{criterion.upper()} distribution:")
        total = sum(cluster_counter.values())
        for label, count in cluster_counter.most_common(5):
            pct = 100 * count / total
            print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")


def main():
    # Configuration
    dataset_name = "tiny_imagenet"
    base_path = "/vol/research/tubui/tiny-imagenet-200/"
    cluster_labels_dir = "/path/to/Documents/GitHub/FDS/data_partitioning/legacy_vlm/data/tiny_imagenet"

    print("=" * 80)
    print("MULTIVARIATE DIRICHLET PARTITIONING EXAMPLE")
    print("=" * 80)

    # Load datasets
    print("\n[Step 1] Loading datasets...")
    train_dataset = BaseDataset(dataset_name, base_path, "train")
    val_dataset = BaseDataset(dataset_name, base_path, "val")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create splitter with high heterogeneity
    print("\n[Step 2] Creating splitter (n_clients=6, α=0.1)...")
    splitter = MultivariateDirichletSplitter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_clients=6,
        heterogeneity=0.1,  # High heterogeneity
        cluster_labels_dir=cluster_labels_dir,
        valset_type="global",
        seed=42,
        min_samples_per_client=100,
    )

    # Get splits
    print("\n[Step 3] Computing client partitions...")
    splits = splitter.get_client_splits()

    # Show basic statistics
    print("\n[Step 4] Basic Statistics")
    print("-" * 80)
    for client_id in range(splitter.n_clients):
        client_name = f"client_{client_id}"
        n_train = len(splits["clients"][client_name]["train"])
        n_val = len(splits["clients"][client_name]["val"])
        print(f"  {client_name}: {n_train} training samples, {n_val} validation samples")

    # Show preference summary
    print("\n[Step 5] Client Preferences")
    print("-" * 80)
    summary = splitter.get_preference_summary()

    for criterion in splitter.CRITERIA:
        print(f"\n{criterion.upper()} - Top preferences:")
        for client_data in summary[f"{criterion}_top_preferences"]:
            print(f"  Client {client_data['client']}: {client_data['label']:20s} "
                  f"(preference={client_data['preference']:.3f})")

    # Detailed analysis for first 2 clients
    print("\n[Step 6] Detailed Client Analysis")
    print("-" * 80)

    for client_id in range(min(2, splitter.n_clients)):
        analyze_client_distribution(splits, client_id, splitter.cluster_labels)

    # Show global validation set
    if splits["global_val"]:
        print(f"\n[Step 7] Global Validation Set")
        print("-" * 80)
        print(f"  Total samples: {len(splits['global_val'])}")

    print("\n" + "=" * 80)
    print("✓ Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
