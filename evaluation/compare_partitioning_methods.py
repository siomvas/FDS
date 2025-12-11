"""
Comprehensive Comparison of Data Partitioning Methods for Federated Learning.

Compares four partitioning approaches across multiple configurations:
1. Random - IID baseline
2. Dirichlet - Label-based heterogeneity (original labels)
3. VLM - Multivariate Dirichlet with VLM clustering criteria
4. Embedding - Semantic clustering (with multiple encoder/pooling/clustering variants)

Metrics computed:
- Label-based: KL divergence, MMD, Mahalanobis, KS test, Vendi Score
- Feature-based: FID, MMD, Wasserstein, Cramér-von Mises, KS multivariate, Vendi Score
- Balance metrics: Distribution of samples across clients

Features:
- Checkpoint/resume functionality for long-running experiments
- Separate execution of label vs feature metrics
- Supports multiple embedding presets (default, fast, high_quality, spectral)
- Exports results to CSV, JSON, and markdown summary

Usage:
    # Compare Random, Dirichlet, and VLM
    python compare_partitioning_methods.py \\
      --base_path /path/to/dataset \\
      --methods random dirichlet vlm \\
      --cluster_labels_dir /path/to/cluster_labels \\
      --metrics label

    # Compare all methods including embedding variants
    python compare_partitioning_methods.py \\
      --base_path /path/to/dataset \\
      --methods random dirichlet vlm embedding \\
      --cluster_labels_dir /path/to/cluster_labels \\
      --embedding_presets default fast \\
      --n_clients 3 6 10 \\
      --heterogeneity 0.1 0.5 10.0 \\
      --metrics both
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_partitioning.random_splitter import RandomSplitter
from data_partitioning.dirichlet_splitter import DirichletSplitter
from data_partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from data_partitioning.embedding_pipeline.embedding_splitter import EmbeddingSplitter
from data_partitioning.embedding_pipeline.config import get_preset_config
from data_partitioning.dataset import BaseDataset
from data_partitioning.base_splitter import BaseDatasetSplitter, DatasetSplits
from evaluation import metrics


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    method: str  # "random", "dirichlet", "vlm", "embedding"
    n_clients: int
    heterogeneity: float
    seed: int
    embedding_preset: Optional[str] = None  # For embedding method only
    cluster_labels_dir: Optional[str] = None  # For VLM method only

    def get_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        if self.method == "embedding" and self.embedding_preset:
            return f"embedding-{self.embedding_preset}_n{self.n_clients}_a{self.heterogeneity}_s{self.seed}"
        return f"{self.method}_n{self.n_clients}_a{self.heterogeneity}_s{self.seed}"


@dataclass
class MetricResults:
    """Results from a single experiment."""
    # Label-based metrics (pairwise averaged)
    kl_divergence_mean: float = 0.0
    kl_divergence_std: float = 0.0
    mmd_labels_mean: float = 0.0
    mmd_labels_std: float = 0.0
    mahalanobis_mean: float = 0.0
    mahalanobis_std: float = 0.0
    ks_test_mean_statistic: float = 0.0
    ks_test_mean_pvalue: float = 0.0

    # Feature-based metrics (pairwise averaged)
    fid_mean: Optional[float] = None
    fid_std: Optional[float] = None
    mmd_features_mean: Optional[float] = None
    wasserstein_mean: Optional[float] = None
    cramer_von_mises_mean: Optional[float] = None
    ks_multivariate_reject_rate: Optional[float] = None

    # Multi-dataset feature metric
    vendi_score_features: Optional[float] = None

    # Balance metrics
    samples_per_client_mean: float = 0.0
    samples_per_client_std: float = 0.0
    samples_per_client_min: int = 0
    samples_per_client_max: int = 0
    coefficient_of_variation: float = 0.0

    # Timing
    split_time: float = 0.0
    label_metrics_time: float = 0.0
    feature_metrics_time: Optional[float] = None


@dataclass
class ExperimentResult:
    """Complete result from an experiment."""
    config: ExperimentConfig
    metrics: MetricResults
    timestamp: str
    completed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "timestamp": self.timestamp,
            "completed": self.completed
        }


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manages checkpointing and resume functionality."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed_experiments": []}

    def save_checkpoint(self, results: List[ExperimentResult]):
        """Save checkpoint."""
        checkpoint_data = {
            "checkpoint_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "completed_experiments": [r.to_dict() for r in results]
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)


# ============================================================================
# Splitter Factory
# ============================================================================

def create_splitter(
    exp_config: ExperimentConfig,
    train_dataset: BaseDataset,
    val_dataset: Optional[BaseDataset],
    base_path: str
) -> BaseDatasetSplitter:
    """
    Create appropriate splitter based on experiment configuration.

    Args:
        exp_config: Experiment configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        base_path: Base path for data

    Returns:
        Configured splitter instance
    """
    if exp_config.method == "random":
        return RandomSplitter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=exp_config.n_clients,
            heterogeneity=exp_config.heterogeneity,
            valset_type="global",
            seed=exp_config.seed,
        )

    elif exp_config.method == "dirichlet":
        return DirichletSplitter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=exp_config.n_clients,
            heterogeneity=exp_config.heterogeneity,
            valset_type="global",
            seed=exp_config.seed,
            min_samples_per_client=10,
        )

    elif exp_config.method == "vlm":
        if not exp_config.cluster_labels_dir:
            raise ValueError("VLM method requires cluster_labels_dir to be specified")

        return MultivariateDirichletSplitter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=exp_config.n_clients,
            heterogeneity=exp_config.heterogeneity,
            cluster_labels_dir=exp_config.cluster_labels_dir,
            valset_type="global",
            seed=exp_config.seed,
            min_samples_per_client=10,
        )

    elif exp_config.method == "embedding":
        # Get preset configuration
        preset = exp_config.embedding_preset or "default"
        config = get_preset_config(preset, n_clients=exp_config.n_clients)

        # Set proxy if needed
        if hasattr(config.model, 'proxy') and config.model.proxy is None:
            # Try to get proxy from environment or set default
            config.model.proxy = os.environ.get('HTTP_PROXY', 'http://proxy.example.org:3128')

        # Adjust clustering parameters based on heterogeneity
        # Lower heterogeneity → larger clusters
        # Higher heterogeneity → smaller clusters
        if config.clustering.method == "hdbscan":
            base_size = len(train_dataset) // (exp_config.n_clients * 3)
            min_size = max(5, int(base_size * (1.0 - exp_config.heterogeneity)))
            config.clustering.params["min_cluster_size"] = min_size
        elif config.clustering.method == "kmeans":
            # For k-means, adjust number of clusters
            n_clusters = max(exp_config.n_clients, int(exp_config.n_clients * (1.0 + 2.0 * exp_config.heterogeneity)))
            config.clustering.params["n_clusters"] = n_clusters

        return EmbeddingSplitter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=exp_config.n_clients,
            heterogeneity=exp_config.heterogeneity,
            valset_type="global",
            config=config,
            seed=exp_config.seed,
        )

    else:
        raise ValueError(f"Unknown method: {exp_config.method}")


# ============================================================================
# Label Mapping (for Tiny ImageNet WNIDs)
# ============================================================================

def get_or_create_wnid_mapping(dataset_name: str, train_dataset: BaseDataset) -> Optional[Dict[str, int]]:
    """
    Get or create WNID to integer mapping for Tiny ImageNet.

    Args:
        dataset_name: Name of the dataset
        train_dataset: Training dataset to extract labels from

    Returns:
        Mapping dictionary if needed, None if labels are already integers
    """
    # Check if we need mapping by looking at first label
    first_label = train_dataset[0][1]

    # If label is already an integer, no mapping needed
    if isinstance(first_label, int):
        return None

    # If label is a string (WNID), create/load mapping
    if isinstance(first_label, str):
        mapping_file = Path(__file__).parent.parent / f"{dataset_name}_wnid_mapping.json"

        # Try to load existing mapping
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            print(f"  Loaded WNID mapping from {mapping_file}")
            return mapping

        # Create new mapping
        print(f"  Creating WNID mapping for {dataset_name}...")
        all_labels = set()
        for i in range(len(train_dataset)):
            all_labels.add(train_dataset[i][1])

        # Sort for consistency
        sorted_labels = sorted(all_labels)
        mapping = {label: idx for idx, label in enumerate(sorted_labels)}

        # Save mapping
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"  Created mapping for {len(mapping)} classes")
        print(f"  Saved to {mapping_file}")

        return mapping

    return None


def apply_label_mapping(dataset: List[Dict], mapping: Optional[Dict[str, int]]) -> List[Dict]:
    """
    Apply label mapping to convert WNIDs to integers.

    Args:
        dataset: List of {"image": path, "label": label}
        mapping: WNID to integer mapping (None if not needed)

    Returns:
        Dataset with integer labels
    """
    if mapping is None:
        return dataset

    return [
        {"image": item["image"], "label": mapping[item["label"]]}
        for item in dataset
    ]


# ============================================================================
# Label Extraction
# ============================================================================

def extract_client_datasets(splits: DatasetSplits) -> List[List[Dict]]:
    """Extract client datasets from splits."""
    client_datasets = []

    for client_name in sorted(splits["clients"].keys()):
        client_split = splits["clients"][client_name]
        client_datasets.append(client_split["train"])

    return client_datasets


def extract_client_datasets_with_original_labels(
    splits: DatasetSplits,
    train_dataset: BaseDataset
) -> List[List[Dict]]:
    """
    Extract client datasets with original labels (not cluster labels).

    For embedding-based methods, splits may contain cluster labels,
    but metrics should be computed on original class labels.
    """
    # Build mapping once for efficiency
    original_label_map = {
        train_dataset[i][0]: train_dataset[i][1]
        for i in range(len(train_dataset))
    }

    client_datasets = []

    for client_name in sorted(splits["clients"].keys()):
        client_split = splits["clients"][client_name]
        client_data = []

        for item in client_split["train"]:
            image_path = item["image"]
            original_label = original_label_map.get(image_path)

            if original_label is None:
                raise ValueError(f"Original label not found for {image_path}")

            client_data.append({
                "image": image_path,
                "label": original_label
            })

        client_datasets.append(client_data)

    return client_datasets


# ============================================================================
# Metric Computation
# ============================================================================

def compute_balance_metrics(client_datasets: List[List[Dict]]) -> Dict[str, float]:
    """Compute data balance metrics across clients."""
    samples_per_client = [len(dataset) for dataset in client_datasets]

    mean_samples = np.mean(samples_per_client)
    std_samples = np.std(samples_per_client)

    return {
        "samples_per_client_mean": float(mean_samples),
        "samples_per_client_std": float(std_samples),
        "samples_per_client_min": int(np.min(samples_per_client)),
        "samples_per_client_max": int(np.max(samples_per_client)),
        "coefficient_of_variation": float(std_samples / (mean_samples + 1e-10)),
    }


def compute_pairwise_label_metrics(
    client_datasets: List[List[Dict]],
    num_classes: int
) -> Dict[str, float]:
    """Compute all pairwise label metrics and aggregate statistics."""
    metrics_dict = {
        'kl_divergence': [],
        'mmd_labels': [],
        'mahalanobis': [],
        'ks_statistic': [],
        'ks_pvalue': []
    }

    # Compute pairwise
    for i, j in combinations(range(len(client_datasets)), 2):
        dataset_i = client_datasets[i]
        dataset_j = client_datasets[j]

        # KL divergence (symmetrized)
        kl_ij = metrics.KL_divergence(dataset_i, dataset_j, num_classes)
        kl_ji = metrics.KL_divergence(dataset_j, dataset_i, num_classes)
        metrics_dict['kl_divergence'].append((kl_ij + kl_ji) / 2)

        # MMD
        mmd = metrics.MMD_label_distributions(dataset_i, dataset_j, num_classes)
        metrics_dict['mmd_labels'].append(mmd)

        # Mahalanobis
        mahal = metrics.mahalanobis_distance_labels(dataset_i, dataset_j, num_classes)
        metrics_dict['mahalanobis'].append(mahal)

        # KS test
        ks_result = metrics.KS_label_test(dataset_i, dataset_j)
        metrics_dict['ks_statistic'].append(ks_result['statistic'])
        metrics_dict['ks_pvalue'].append(ks_result['pvalue'])

    # Aggregate
    results = {}
    for metric_name, values in metrics_dict.items():
        results[f"{metric_name}_mean"] = np.mean(values)
        results[f"{metric_name}_std"] = np.std(values)

    return results

def compute_feature_metrics(
    client_datasets: List[List[Dict]],
    subsample: Optional[int] = None
) -> Dict[str, float]:
    """Compute feature-based metrics across clients."""
    if subsample:
        client_datasets_sub = [dataset[:subsample] for dataset in client_datasets]
    else:
        client_datasets_sub = client_datasets
    results = {}

    # 1. FID (pairwise)
    print("    Computing FID...")
    fid_scores = metrics.FID_distance(
        client_datasets_sub,
        return_all=True,
        batch_size=50
    )
    results['fid_mean'] = np.mean(fid_scores)
    results['fid_std'] = np.std(fid_scores)

    # 2. Pairwise feature metrics
    print("    Computing pairwise feature metrics...")
    mmd_values = []
    wasserstein_values = []
    cvm_values = []
    ks_reject_count = 0
    total_pairs = 0

    for i, j in combinations(range(len(client_datasets_sub)), 2):
        dataset_i = client_datasets_sub[i]
        dataset_j = client_datasets_sub[j]

        # MMD
        mmd = metrics.MMD_features(dataset_i, dataset_j, batch_size=50)
        mmd_values.append(mmd)

        # Wasserstein
        wass = metrics.wasserstein_distance_features(dataset_i, dataset_j, batch_size=50)
        wasserstein_values.append(wass)

        # Cramér-von Mises
        cvm = metrics.cramer_von_mises_features(dataset_i, dataset_j, batch_size=50)
        cvm_values.append(cvm)

        # KS multivariate
        ks_result = metrics.KS_multivariate_test(dataset_i, dataset_j, batch_size=50)
        if ks_result['reject_null']:
            ks_reject_count += 1
        total_pairs += 1

    results['mmd_features_mean'] = np.mean(mmd_values)
    results['wasserstein_mean'] = np.mean(wasserstein_values)
    results['cramer_von_mises_mean'] = np.mean(cvm_values)
    results['ks_multivariate_reject_rate'] = ks_reject_count / total_pairs

    # 3. Vendi Score (multi-dataset)
    print("    Computing Vendi Score (features)...")
    vendi_features = metrics.vendi_score_features(
        client_datasets_sub,
        batch_size=50
    )
    results['vendi_score_features'] = vendi_features

    return results


# ============================================================================
# Experiment Execution
# ============================================================================

def run_experiment(
    exp_config: ExperimentConfig,
    train_dataset: BaseDataset,
    val_dataset: BaseDataset,
    base_path: str,
    num_classes: int,
    metrics_mode: str,
    subsample: Optional[int],
    dataset_name: str,
    wnid_mapping: Optional[Dict[str, int]],
) -> ExperimentResult:
    """
    Execute a single experiment.

    Args:
        exp_config: Experiment configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        base_path: Base path for data
        num_classes: Number of classes
        metrics_mode: "label", "feature", or "both"
        dataset_name: Name of the dataset
        wnid_mapping: WNID to integer mapping (if needed)

    Returns:
        ExperimentResult with all computed metrics
    """
    # Create splitter
    splitter = create_splitter(exp_config, train_dataset, val_dataset, base_path)

    # Generate splits
    start_time = time.time()
    splits = splitter.get_client_splits()
    split_time = time.time() - start_time

    # Prepare client datasets with original labels
    # VLM and embedding methods use cluster labels internally, but we need original labels for metrics
    if exp_config.method in ["vlm", "embedding"]:
        client_datasets = extract_client_datasets_with_original_labels(splits, train_dataset)
    else:
        client_datasets = extract_client_datasets(splits)

    # Apply WNID mapping if needed (for Tiny ImageNet)
    if wnid_mapping is not None:
        client_datasets = [apply_label_mapping(dataset, wnid_mapping) for dataset in client_datasets]

    # Compute balance metrics
    balance_metrics_dict = compute_balance_metrics(client_datasets)

    # Initialize results
    metric_results = MetricResults(split_time=split_time, **balance_metrics_dict)

    # Compute label metrics
    if metrics_mode in ["label", "both"]:
        print("  Computing label metrics...")
        start_time = time.time()

        # Pairwise label metrics
        pairwise_label = compute_pairwise_label_metrics(client_datasets, num_classes)

        label_time = time.time() - start_time

        # Update results
        for key, value in pairwise_label.items():
            setattr(metric_results, key, value)

        metric_results.label_metrics_time = label_time

    # Compute feature metrics
    if metrics_mode in ["feature", "both"]:
        print("  Computing feature metrics...")
        start_time = time.time()

        feature_results = compute_feature_metrics(client_datasets, subsample=subsample)

        feature_time = time.time() - start_time

        # Update results
        for key, value in feature_results.items():
            setattr(metric_results, key, value)

        metric_results.feature_metrics_time = feature_time

    # Mark as completed
    return ExperimentResult(
        config=exp_config,
        metrics=metric_results,
        timestamp=datetime.now().isoformat(),
        completed=True
    )


# ============================================================================
# Experiment Generation
# ============================================================================

def generate_experiment_configs(
    methods: List[str],
    n_clients_list: List[int],
    heterogeneity_list: List[float],
    embedding_presets: List[str],
    cluster_labels_dir: Optional[str],
    seed: int
) -> List[ExperimentConfig]:
    """Generate all experiment configurations."""
    experiments = []

    for n_clients in n_clients_list:
        for heterogeneity in heterogeneity_list:
            # Random and Dirichlet
            for method in ["random", "dirichlet"]:
                if method in methods:
                    experiments.append(ExperimentConfig(
                        method=method,
                        n_clients=n_clients,
                        heterogeneity=heterogeneity,
                        seed=seed
                    ))

            # VLM
            if "vlm" in methods:
                if not cluster_labels_dir:
                    raise ValueError("VLM method requires --cluster_labels_dir to be specified")
                experiments.append(ExperimentConfig(
                    method="vlm",
                    n_clients=n_clients,
                    heterogeneity=heterogeneity,
                    seed=seed,
                    cluster_labels_dir=cluster_labels_dir
                ))

            # Embedding with presets
            if "embedding" in methods:
                for preset in embedding_presets:
                    experiments.append(ExperimentConfig(
                        method="embedding",
                        n_clients=n_clients,
                        heterogeneity=heterogeneity,
                        seed=seed,
                        embedding_preset=preset
                    ))

    return experiments


# ============================================================================
# Results Export
# ============================================================================

def export_results(results: List[ExperimentResult], output_dir: Path):
    """Export results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to JSON
    json_path = output_dir / "comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results]
        }, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")

    # Export to CSV
    csv_data = []
    for result in results:
        row = {
            "method": result.config.method,
            "n_clients": result.config.n_clients,
            "heterogeneity": result.config.heterogeneity,
            "embedding_preset": result.config.embedding_preset or "",
            **asdict(result.metrics)
        }
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")


def generate_summary_report(results: List[ExperimentResult], output_dir: Path):
    """Generate markdown summary report."""
    report = []
    report.append("# Partitioning Methods Comparison Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"Total experiments: {len(results)}\n")

    # Create summary DataFrame
    df_data = []
    for result in results:
        row = {
            "Method": result.config.method,
            "Preset": result.config.embedding_preset or "-",
            "n_clients": result.config.n_clients,
            "α": result.config.heterogeneity,
            "KL Div": f"{result.metrics.kl_divergence_mean:.4f}",
            "MMD": f"{result.metrics.mmd_labels_mean:.4f}",
            "CV": f"{result.metrics.coefficient_of_variation:.4f}",
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    report.append("\n## Summary Table\n")
    report.append("```")
    report.append(df.to_string(index=False))
    report.append("```\n")

    # Save report
    report_path = output_dir / "summary_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report))

    print(f"✓ Summary report saved to {report_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compare data partitioning methods for federated learning"
    )
    parser.add_argument("--dataset", type=str, default="tiny_imagenet", help="Dataset name")
    parser.add_argument("--base_path", type=str, default="/path/to/gtai/FDS/", help="Base path to dataset")
    parser.add_argument("--n_clients", type=int, nargs="+", default=[3, 6],
                       help="List of client counts to test")
    parser.add_argument("--heterogeneity", type=float, nargs="+", default=[0.1, 10.0],
                       help="List of heterogeneity (alpha) values to test")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["random", "dirichlet", "embedding"],
                       help="Methods to compare (random, dirichlet, vlm, embedding)")
    parser.add_argument("--embedding_presets", type=str, nargs="+",
                       default=["default", "fast"],
                       help="Embedding presets to test (default, fast, high_quality, spectral)")
    parser.add_argument("--cluster_labels_dir", type=str, default="../data_partitioning/legacy_vlm/data/tiny_imagenet",
                       help="Path to VLM cluster labels directory (required for VLM method)")
    parser.add_argument("--metrics", type=str, default="both",
                       choices=["label", "feature", "both"],
                       help="Which metrics to compute")
    parser.add_argument("--subsample", type=int, default=None,)
    parser.add_argument("--output_dir", type=str, default="../results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 80)
    print("PARTITIONING METHODS COMPARISON")
    print("=" * 80)

    # Setup
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    val_dataset = BaseDataset(args.dataset, args.base_path, "val")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Get number of classes and handle WNID mapping
    all_labels = set(train_dataset[i][1] for i in range(len(train_dataset)))
    num_classes = len(all_labels)
    print(f"  Number of classes: {num_classes}")

    # Create/load WNID mapping for Tiny ImageNet (or other datasets with string labels)
    wnid_mapping = get_or_create_wnid_mapping(args.dataset, train_dataset)

    # Load checkpoint
    checkpoint = checkpoint_manager.load_checkpoint()
    completed_experiments = checkpoint.get("completed_experiments", [])
    completed_ids = {
        ExperimentConfig(**exp["config"]).get_experiment_id()
        for exp in completed_experiments
    }

    # Generate all experiment configurations
    all_experiments = generate_experiment_configs(
        methods=args.methods,
        n_clients_list=args.n_clients,
        heterogeneity_list=args.heterogeneity,
        embedding_presets=args.embedding_presets,
        cluster_labels_dir=args.cluster_labels_dir,
        seed=args.seed
    )

    # Filter out completed
    pending_experiments = [
        exp for exp in all_experiments
        if exp.get_experiment_id() not in completed_ids
    ]

    print(f"\nExperiment Status:")
    print(f"  Total: {len(all_experiments)}")
    print(f"  Completed: {len(completed_experiments)}")
    print(f"  Pending: {len(pending_experiments)}")

    # Convert completed experiments back to ExperimentResult objects
    results = [
        ExperimentResult(
            config=ExperimentConfig(**exp["config"]),
            metrics=MetricResults(**exp["metrics"]),
            timestamp=exp["timestamp"],
            completed=exp["completed"]
        )
        for exp in completed_experiments
    ]

    # Run pending experiments
    if pending_experiments:
        print(f"\nRunning experiments...")

        for exp_config in tqdm(pending_experiments, desc="Progress"):
            exp_id = exp_config.get_experiment_id()
            print(f"\n[{exp_id}]")

            try:
                exp_result = run_experiment(
                    exp_config=exp_config,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    base_path=args.base_path,
                    num_classes=num_classes,
                    metrics_mode=args.metrics,
                    subsample=args.subsample,
                    dataset_name=args.dataset,
                    wnid_mapping=wnid_mapping,
                )

                results.append(exp_result)

                # Save checkpoint after each experiment
                checkpoint_manager.save_checkpoint(results)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                # Save what we have so far
                checkpoint_manager.save_checkpoint(results)
                raise

    # Export results
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)

    export_results(results, output_dir)
    generate_summary_report(results, output_dir)

    print("\n" + "=" * 80)
    print("✓ ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
