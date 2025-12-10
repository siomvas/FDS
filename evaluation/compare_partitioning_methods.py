"""
Comprehensive Evaluation of Partitioning Methods

Compares three federated learning data partitioning strategies:
1. MultivariateDirichletSplitter - Uses 5 VLM cluster criteria with Dirichlet preferences
2. RandomSplitter - IID baseline
3. DirichletSplitter - Standard Dirichlet on original dataset labels

Test configurations:
- n_clients: [3, 6, 10, 20]
- heterogeneity: [0.1 (high), 10.0 (low)]

Metrics computed:
- Label distribution metrics (KL divergence, MMD)
- Distribution balance metrics
- Cluster distribution metrics (for multivariate method)

Outputs:
- CSV: evaluation/results/partitioning_comparison.csv
- JSON: evaluation/results/partitioning_comparison.json
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from data_partitioning.random_splitter import RandomSplitter
from data_partitioning.dirichlet_splitter import DirichletSplitter
from data_partitioning.dataset import BaseDataset
from evaluation import metrics


# Configuration
DATASET_NAME = "tiny_imagenet"
BASE_PATH = "/vol/research/tubui/tiny-imagenet-200/"
CLUSTER_LABELS_DIR = "/path/to/Documents/GitHub/FDS/data_partitioning/legacy_vlm/data/tiny_imagenet"
RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42

# Test configurations
N_CLIENTS_LIST = [3, 6, 10, 20]
HETEROGENEITY_LIST = [0.1, 10.0]  # High and low heterogeneity


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def convert_labels_to_int(client_data: List[Dict], label_mapping: Dict[str, int]) -> List[Dict]:
    """
    Convert string labels to integers for metric computation.

    Args:
        client_data: List of {"image": path, "label": str}
        label_mapping: Mapping from string labels to integers

    Returns:
        List of {"image": path, "label": int}
    """
    return [
        {"image": item["image"], "label": label_mapping[item["label"]]}
        for item in client_data
    ]


def create_label_mapping(splits: Dict) -> Dict[str, int]:
    """
    Create a mapping from string labels to integers.

    Args:
        splits: Client splits from splitter

    Returns:
        Dict mapping label strings to integers
    """
    all_labels = set()
    for client_name, client_data in splits["clients"].items():
        for item in client_data["train"]:
            all_labels.add(item["label"])

    sorted_labels = sorted(all_labels)
    return {label: idx for idx, label in enumerate(sorted_labels)}


def compute_label_distribution_metrics(splits: Dict, label_mapping: Dict[str, int], num_classes: int) -> Dict[str, float]:
    """
    Compute label distribution metrics between clients.

    Args:
        splits: Client splits
        label_mapping: Label to integer mapping
        num_classes: Total number of classes

    Returns:
        Dict with metric values
    """
    n_clients = len(splits["clients"])
    client_names = sorted(splits["clients"].keys())

    # Convert to integer labels
    client_datasets = []
    for client_name in client_names:
        client_train = splits["clients"][client_name]["train"]
        client_int_labels = convert_labels_to_int(client_train, label_mapping)
        client_datasets.append(client_int_labels)

    # Compute pairwise metrics
    kl_divs = []
    mmds = []

    for i, j in combinations(range(n_clients), 2):
        # KL divergence (symmetrized)
        kl_ij = metrics.KL_divergence(client_datasets[i], client_datasets[j], num_classes=num_classes)
        kl_ji = metrics.KL_divergence(client_datasets[j], client_datasets[i], num_classes=num_classes)
        kl_divs.append((kl_ij + kl_ji) / 2)

        # MMD
        mmd = metrics.MMD_label_distributions(client_datasets[i], client_datasets[j], num_classes=num_classes)
        mmds.append(mmd)

    return {
        "avg_kl_divergence": float(np.mean(kl_divs)),
        "std_kl_divergence": float(np.std(kl_divs)),
        "avg_mmd": float(np.mean(mmds)),
        "std_mmd": float(np.std(mmds)),
    }


def compute_cluster_distribution_metrics(
    splits: Dict,
    cluster_labels: Dict[str, Dict[str, str]],
    criteria: List[str]
) -> Dict[str, Any]:
    """
    Compute distribution metrics for cluster labels (only for multivariate method).

    Args:
        splits: Client splits
        cluster_labels: Cluster labels for each criterion
        criteria: List of criteria names

    Returns:
        Dict with cluster distribution metrics per criterion
    """
    n_clients = len(splits["clients"])
    client_names = sorted(splits["clients"].keys())

    cluster_metrics = {}

    for criterion in criteria:
        # Get cluster label distribution for each client
        client_cluster_counts = []

        for client_name in client_names:
            client_train = splits["clients"][client_name]["train"]
            cluster_counter = Counter()

            for item in client_train:
                image_path = item["image"]
                cluster_label = cluster_labels[criterion].get(image_path)
                if cluster_label:
                    cluster_counter[cluster_label] += 1

            client_cluster_counts.append(cluster_counter)

        # Compute entropy for each client (higher = more diverse)
        entropies = []
        for counter in client_cluster_counts:
            total = sum(counter.values())
            if total > 0:
                probs = np.array([count / total for count in counter.values()])
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)

        cluster_metrics[criterion] = {
            "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "std_entropy": float(np.std(entropies)) if entropies else 0.0,
        }

    return cluster_metrics


def compute_balance_metrics(splits: Dict) -> Dict[str, float]:
    """
    Compute metrics about data balance across clients.

    Args:
        splits: Client splits

    Returns:
        Dict with balance metrics
    """
    samples_per_client = []

    for client_name, client_data in splits["clients"].items():
        samples_per_client.append(len(client_data["train"]))

    return {
        "mean_samples_per_client": float(np.mean(samples_per_client)),
        "std_samples_per_client": float(np.std(samples_per_client)),
        "min_samples_per_client": int(np.min(samples_per_client)),
        "max_samples_per_client": int(np.max(samples_per_client)),
        "cv_samples_per_client": float(np.std(samples_per_client) / (np.mean(samples_per_client) + 1e-10)),
    }


def evaluate_method(
    method_name: str,
    splitter,
    n_clients: int,
    heterogeneity: float,
    num_classes: int,
    include_cluster_metrics: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single partitioning method.

    Args:
        method_name: Name of the method
        splitter: Splitter instance
        n_clients: Number of clients
        heterogeneity: Heterogeneity parameter
        num_classes: Number of classes in the dataset
        include_cluster_metrics: Whether to compute cluster distribution metrics

    Returns:
        Dict with all evaluation metrics
    """
    print(f"\n  Evaluating {method_name} with n_clients={n_clients}, α={heterogeneity}...")

    # Get splits
    start_time = time.time()
    splits = splitter.get_client_splits()
    split_time = time.time() - start_time

    # Create label mapping
    label_mapping = create_label_mapping(splits)

    # Compute metrics
    label_metrics = compute_label_distribution_metrics(splits, label_mapping, num_classes)
    balance_metrics = compute_balance_metrics(splits)

    # Combine results
    results = {
        "method": method_name,
        "n_clients": n_clients,
        "heterogeneity": heterogeneity,
        "split_time_seconds": float(split_time),
        **label_metrics,
        **balance_metrics,
    }

    # Add cluster metrics for multivariate method
    if include_cluster_metrics and hasattr(splitter, 'cluster_labels'):
        cluster_metrics = compute_cluster_distribution_metrics(
            splits,
            splitter.cluster_labels,
            splitter.CRITERIA
        )
        results["cluster_metrics"] = cluster_metrics

    print(f"    KL divergence: {label_metrics['avg_kl_divergence']:.4f} ± {label_metrics['std_kl_divergence']:.4f}")
    print(f"    MMD: {label_metrics['avg_mmd']:.4f} ± {label_metrics['std_mmd']:.4f}")
    print(f"    Samples/client: {balance_metrics['mean_samples_per_client']:.1f} ± {balance_metrics['std_samples_per_client']:.1f}")
    print(f"    Time: {split_time:.2f}s")

    return results


def run_comparison():
    """Run comprehensive comparison of all methods."""
    print("=" * 80)
    print("PARTITIONING METHODS COMPARISON")
    print("=" * 80)

    ensure_results_dir()

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = BaseDataset(DATASET_NAME, BASE_PATH, "train")
    val_dataset = BaseDataset(DATASET_NAME, BASE_PATH, "val")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Count number of unique classes
    all_labels = set(train_dataset[i][1] for i in range(len(train_dataset)))
    num_classes = len(all_labels)
    print(f"  Number of classes: {num_classes}")

    # Store all results
    all_results = []

    # Run experiments
    for n_clients in N_CLIENTS_LIST:
        for heterogeneity in HETEROGENEITY_LIST:
            print(f"\n{'=' * 80}")
            print(f"Configuration: n_clients={n_clients}, α={heterogeneity}")
            print(f"{'=' * 80}")

            # 1. Multivariate Dirichlet
            print("\n[1/3] MultivariateDirichletSplitter")
            mv_splitter = MultivariateDirichletSplitter(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                n_clients=n_clients,
                heterogeneity=heterogeneity,
                cluster_labels_dir=CLUSTER_LABELS_DIR,
                valset_type="global",
                seed=SEED,
                min_samples_per_client=10,
            )
            mv_results = evaluate_method(
                "MultivariateDirichlet",
                mv_splitter,
                n_clients,
                heterogeneity,
                num_classes,
                include_cluster_metrics=True
            )
            all_results.append(mv_results)

            # 2. Random (IID)
            print("\n[2/3] RandomSplitter (IID)")
            random_splitter = RandomSplitter(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                n_clients=n_clients,
                heterogeneity=0.0,  # Not used for random
                valset_type="global",
                seed=SEED,
            )
            random_results = evaluate_method(
                "Random_IID",
                random_splitter,
                n_clients,
                0.0,  # IID
                num_classes,
                include_cluster_metrics=False
            )
            all_results.append(random_results)

            # 3. Standard Dirichlet on original labels
            print("\n[3/3] DirichletSplitter (original labels)")
            dirichlet_splitter = DirichletSplitter(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                n_clients=n_clients,
                heterogeneity=heterogeneity,
                valset_type="global",
                seed=SEED,
                min_samples_per_client=10,
            )
            dirichlet_results = evaluate_method(
                "Dirichlet_OriginalLabels",
                dirichlet_splitter,
                n_clients,
                heterogeneity,
                num_classes,
                include_cluster_metrics=False
            )
            all_results.append(dirichlet_results)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save as JSON
    json_path = RESULTS_DIR / "partitioning_comparison.json"
    with open(json_path, 'w') as f:
        json.dump({
            "metadata": {
                "dataset": DATASET_NAME,
                "num_classes": num_classes,
                "n_train_samples": len(train_dataset),
                "n_val_samples": len(val_dataset),
                "seed": SEED,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": all_results,
        }, f, indent=2)
    print(f"  JSON saved to: {json_path}")

    # Save as CSV (flatten cluster_metrics for CSV)
    csv_results = []
    for result in all_results:
        csv_row = {k: v for k, v in result.items() if k != "cluster_metrics"}

        # Add cluster metrics as separate columns if present
        if "cluster_metrics" in result:
            for criterion, metrics_dict in result["cluster_metrics"].items():
                for metric_name, value in metrics_dict.items():
                    csv_row[f"{criterion}_{metric_name}"] = value

        csv_results.append(csv_row)

    df = pd.DataFrame(csv_results)
    csv_path = RESULTS_DIR / "partitioning_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    summary_cols = ["method", "n_clients", "heterogeneity", "avg_kl_divergence", "avg_mmd", "cv_samples_per_client"]
    if all(col in df.columns for col in summary_cols):
        summary_df = df[summary_cols].copy()
        summary_df = summary_df.round(4)
        print(summary_df.to_string(index=False))
    else:
        print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
