"""
Comprehensive Comparison of Data Partitioning Methods for Federated Learning.

Compares four partitioning approaches:
1. Random - IID baseline
2. Dirichlet - Label-based heterogeneity
3. VLM - Multivariate Dirichlet with VLM clustering
4. Embedding - Semantic clustering

Usage:
    python compare_partitioning_methods.py \
      --base_path /path/to/dataset \
      --methods random dirichlet vlm \
      --cluster_labels_dir /path/to/cluster_labels \
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    method: str
    n_clients: int
    heterogeneity: float
    seed: int
    embedding_preset: Optional[str] = None
    cluster_labels_dir: Optional[str] = None

    def get_experiment_id(self) -> str:
        if self.method == "embedding" and self.embedding_preset:
            return f"embedding-{self.embedding_preset}_n{self.n_clients}_a{self.heterogeneity}_s{self.seed}"
        return f"{self.method}_n{self.n_clients}_a{self.heterogeneity}_s{self.seed}"

    def get_display_name(self) -> str:
        if self.method == "embedding" and self.embedding_preset:
            return f"Embedding ({self.embedding_preset})"
        return self.method.capitalize()


@dataclass
class MetricResults:
    # Label metrics
    JS_divergence_mean: float = 0.0
    JS_divergence_std: float = 0.0
    mmd_labels_mean: float = 0.0
    mmd_labels_std: float = 0.0
    mahalanobis_mean: float = 0.0
    mahalanobis_std: float = 0.0
    ks_label_statistic_mean: float = 0.0
    ks_label_pvalue_mean: float = 0.0

    # Feature metrics
    fid_mean: Optional[float] = None
    fid_std: Optional[float] = None
    mmd_features_mean: Optional[float] = None
    mmd_features_std: Optional[float] = None
    wasserstein_mean: Optional[float] = None
    wasserstein_std: Optional[float] = None
    cramer_von_mises_mean: Optional[float] = None
    cramer_von_mises_std: Optional[float] = None
    ks_multivariate_reject_rate: Optional[float] = None
    ks_multivariate_min_pvalue: Optional[float] = None
    vendi_score_mean: Optional[float] = None
    vendi_score_std: Optional[float] = None

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
    config: ExperimentConfig
    metrics: MetricResults
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "timestamp": self.timestamp,
        }


# ============================================================================
# Splitter Factory
# ============================================================================

def create_splitter(
    exp_config: ExperimentConfig,
    train_dataset: BaseDataset,
    val_dataset: Optional[BaseDataset],
    base_path: str
) -> BaseDatasetSplitter:
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
            raise ValueError("VLM method requires cluster_labels_dir")

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
        preset = exp_config.embedding_preset or "default"
        config = get_preset_config(preset, n_clients=exp_config.n_clients)

        if hasattr(config.model, 'proxy') and config.model.proxy is None:
            config.model.proxy = os.environ.get('HTTP_PROXY', 'http://proxy.example.org:3128')

        if config.clustering.method == "hdbscan":
            base_size = len(train_dataset) // (exp_config.n_clients * 3)
            min_size = max(5, int(base_size * (1.0 - exp_config.heterogeneity)))
            config.clustering.params["min_cluster_size"] = min_size
        elif config.clustering.method == "kmeans":
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
# Dataset Extraction
# ============================================================================

def get_or_create_wnid_mapping(dataset_name: str, train_dataset: BaseDataset) -> Optional[Dict[str, int]]:
    first_label = train_dataset[0][1]

    if isinstance(first_label, int):
        return None

    if isinstance(first_label, str):
        mapping_file = Path(__file__).parent.parent / f"{dataset_name}_wnid_mapping.json"

        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)

        print(f"  Creating WNID mapping for {dataset_name}...")
        all_labels = sorted(set(train_dataset[i][1] for i in range(len(train_dataset))))
        mapping = {label: idx for idx, label in enumerate(all_labels)}

        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        return mapping

    return None


def apply_label_mapping(dataset: List[Dict], mapping: Optional[Dict[str, int]]) -> List[Dict]:
    if mapping is None:
        return dataset
    return [{"image": item["image"], "label": mapping[item["label"]]} for item in dataset]


def extract_client_datasets(
    splits: DatasetSplits,
    train_dataset: Optional[BaseDataset] = None,
    use_original_labels: bool = False
) -> List[List[Dict]]:
    """Extract client datasets, optionally remapping to original labels."""
    if use_original_labels and train_dataset is not None:
        original_label_map = {train_dataset[i][0]: train_dataset[i][1] for i in range(len(train_dataset))}

    client_datasets = []
    for client_name in sorted(splits["clients"].keys()):
        client_split = splits["clients"][client_name]

        if use_original_labels and train_dataset is not None:
            client_data = [
                {"image": item["image"], "label": original_label_map[item["image"]]}
                for item in client_split["train"]
            ]
        else:
            client_data = client_split["train"]

        client_datasets.append(client_data)

    return client_datasets


def subsample_datasets(
    client_datasets: List[List[Dict]],
    max_samples: int,
    seed: int
) -> List[List[Dict]]:
    """Randomly subsample each client dataset."""
    rng = np.random.default_rng(seed)
    subsampled = []

    for dataset in client_datasets:
        if len(dataset) <= max_samples:
            subsampled.append(dataset)
        else:
            indices = rng.choice(len(dataset), size=max_samples, replace=False)
            subsampled.append([dataset[i] for i in indices])

    return subsampled


# ============================================================================
# Metric Computation
# ============================================================================

def compute_balance_metrics(client_datasets: List[List[Dict]]) -> Dict[str, float]:
    samples_per_client = [len(d) for d in client_datasets]
    mean_samples = np.mean(samples_per_client)
    std_samples = np.std(samples_per_client)

    return {
        "samples_per_client_mean": float(mean_samples),
        "samples_per_client_std": float(std_samples),
        "samples_per_client_min": int(np.min(samples_per_client)),
        "samples_per_client_max": int(np.max(samples_per_client)),
        "coefficient_of_variation": float(std_samples / (mean_samples + 1e-10)),
    }


def compute_label_metrics(
    client_datasets: List[List[Dict]],
    num_classes: int
) -> Dict[str, float]:
    """Compute all label-based metrics using the refactored API."""
    results = {}

    # JS divergence
    JS_all = metrics.JS_divergence(client_datasets, num_classes, return_all=True)
    results['JS_divergence_mean'] = float(np.mean(JS_all))
    results['JS_divergence_std'] = float(np.std(JS_all))

    # MMD labels
    mmd_all = metrics.MMD_label_distributions(client_datasets, num_classes, return_all=True)
    results['mmd_labels_mean'] = float(np.mean(mmd_all))
    results['mmd_labels_std'] = float(np.std(mmd_all))

    # Mahalanobis
    mahal_all = metrics.mahalanobis_distance_labels(client_datasets, num_classes, return_all=True)
    results['mahalanobis_mean'] = float(np.mean(mahal_all))
    results['mahalanobis_std'] = float(np.std(mahal_all))

    # KS label test
    ks_all = metrics.KS_label_test(client_datasets, return_all=True)
    results['ks_label_statistic_mean'] = float(np.mean([r['statistic'] for r in ks_all]))
    results['ks_label_pvalue_mean'] = float(np.mean([r['pvalue'] for r in ks_all]))

    return results


def compute_feature_metrics(
    client_datasets: List[List[Dict]],
    batch_size: int = 50,
    use_cache: bool = True,
    cache_dir: str = "./.inception_cache"
) -> Dict[str, float]:
    """Compute all feature-based metrics using the refactored API."""
    results = {}

    print("    Computing FID...")
    fid_all = metrics.FID_distance(
        client_datasets, batch_size=batch_size, return_all=True,
        use_cache=use_cache, cache_dir=cache_dir
    )
    results['fid_mean'] = float(np.mean(fid_all))
    results['fid_std'] = float(np.std(fid_all))

    print("    Computing MMD features...")
    mmd_all = metrics.MMD_features(
        client_datasets, batch_size=batch_size, return_all=True,
        use_cache=use_cache, cache_dir=cache_dir
    )
    results['mmd_features_mean'] = float(np.mean(mmd_all))
    results['mmd_features_std'] = float(np.std(mmd_all))

    print("    Computing Wasserstein distance...")
    wass_all = metrics.wasserstein_distance_features(
        client_datasets, batch_size=batch_size, return_all=True,
        use_cache=use_cache, cache_dir=cache_dir
    )
    results['wasserstein_mean'] = float(np.mean(wass_all))
    results['wasserstein_std'] = float(np.std(wass_all))

    print("    Computing Cramér-von Mises...")
    cvm_all = metrics.cramer_von_mises_features(
        client_datasets, batch_size=batch_size, return_all=True,
        use_cache=use_cache, cache_dir=cache_dir
    )
    results['cramer_von_mises_mean'] = float(np.mean(cvm_all))
    results['cramer_von_mises_std'] = float(np.std(cvm_all))

    print("    Computing KS multivariate test...")
    ks_all = metrics.KS_multivariate_test(
        client_datasets, batch_size=batch_size, return_all=True,
        use_cache=use_cache, cache_dir=cache_dir
    )
    results['ks_multivariate_reject_rate'] = float(np.mean([r['reject_null'] for r in ks_all]))
    results['ks_multivariate_min_pvalue'] = float(np.min([r['min_pvalue'] for r in ks_all]))

    print("    Computing Vendi score...")
    vendi_all = metrics.vendi_score_features(
        client_datasets, batch_size=batch_size, return_all=True,
        use_cache=use_cache, cache_dir=cache_dir
    )
    results['vendi_score_mean'] = float(np.mean(vendi_all))
    results['vendi_score_std'] = float(np.std(vendi_all))

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
    wnid_mapping: Optional[Dict[str, int]],
    use_cache: bool,
    cache_dir: str,
) -> ExperimentResult:
    """Execute a single experiment."""
    # Create and run splitter
    splitter = create_splitter(exp_config, train_dataset, val_dataset, base_path)

    start_time = time.time()
    splits = splitter.get_client_splits()
    split_time = time.time() - start_time

    # Extract client datasets
    use_original = exp_config.method in ["vlm", "embedding"]
    client_datasets = extract_client_datasets(
        splits,
        train_dataset if use_original else None,
        use_original_labels=use_original
    )

    # Apply WNID mapping if needed
    if wnid_mapping is not None:
        client_datasets = [apply_label_mapping(d, wnid_mapping) for d in client_datasets]

    # Subsample for feature metrics if requested
    if subsample:
        client_datasets_sub = subsample_datasets(client_datasets, subsample, exp_config.seed)
    else:
        client_datasets_sub = client_datasets

    # Compute balance metrics
    balance = compute_balance_metrics(client_datasets)
    metric_results = MetricResults(split_time=split_time, **balance)

    # Compute label metrics
    if metrics_mode in ["label", "both"]:
        print("  Computing label metrics...")
        start_time = time.time()
        label_results = compute_label_metrics(client_datasets, num_classes)
        metric_results.label_metrics_time = time.time() - start_time

        for key, value in label_results.items():
            setattr(metric_results, key, value)

    # Compute feature metrics
    if metrics_mode in ["feature", "both"]:
        print("  Computing feature metrics...")
        start_time = time.time()
        feature_results = compute_feature_metrics(
            client_datasets_sub,
            use_cache=use_cache,
            cache_dir=cache_dir
        )
        metric_results.feature_metrics_time = time.time() - start_time

        for key, value in feature_results.items():
            setattr(metric_results, key, value)

    return ExperimentResult(
        config=exp_config,
        metrics=metric_results,
        timestamp=datetime.now().isoformat(),
    )


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
            for method in ["random", "dirichlet"]:
                if method in methods:
                    experiments.append(ExperimentConfig(
                        method=method,
                        n_clients=n_clients,
                        heterogeneity=heterogeneity,
                        seed=seed
                    ))

            if "vlm" in methods:
                if not cluster_labels_dir:
                    raise ValueError("VLM method requires --cluster_labels_dir")
                experiments.append(ExperimentConfig(
                    method="vlm",
                    n_clients=n_clients,
                    heterogeneity=heterogeneity,
                    seed=seed,
                    cluster_labels_dir=cluster_labels_dir
                ))

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
# Results Export & Visualization
# ============================================================================

def results_to_dataframe(results: List[ExperimentResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
    rows = []
    for r in results:
        row = {
            "method": r.config.get_display_name(),
            "n_clients": r.config.n_clients,
            "heterogeneity": r.config.heterogeneity,
            **asdict(r.metrics)
        }
        rows.append(row)
    return pd.DataFrame(rows)


def export_results(results: List[ExperimentResult], output_dir: Path):
    """Export results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results]
        }, f, indent=2)
    print(f"✓ Results saved to {json_path}")

    # CSV
    df = results_to_dataframe(results)
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")

    return df


def create_metric_comparison_plot(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    std_col: Optional[str] = None
):
    """Create a grouped bar chart comparing methods across configurations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = df['method'].unique()
    x_labels = df.groupby(['n_clients', 'heterogeneity']).ngroups
    configs = df[['n_clients', 'heterogeneity']].drop_duplicates().values

    x = np.arange(len(configs))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        method_df = df[df['method'] == method]
        values = []
        errors = []

        for n_clients, het in configs:
            row = method_df[(method_df['n_clients'] == n_clients) & (method_df['heterogeneity'] == het)]
            if len(row) > 0:
                values.append(row[metric_col].values[0])
                if std_col and std_col in row.columns:
                    errors.append(row[std_col].values[0])
                else:
                    errors.append(0)
            else:
                values.append(0)
                errors.append(0)

        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, yerr=errors if std_col else None, capsize=3)

    ax.set_xlabel('Configuration (n_clients, α)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'({n}, {h})' for n, h in configs], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path
):
    """Create heatmaps for each method showing metric across n_clients and heterogeneity."""
    methods = df['method'].unique()
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), squeeze=False)

    for i, method in enumerate(methods):
        method_df = df[df['method'] == method]
        pivot = method_df.pivot(index='heterogeneity', columns='n_clients', values=metric_col)

        ax = axes[0, i]
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar=True)
        ax.set_title(f'{method}')
        ax.set_xlabel('n_clients')
        ax.set_ylabel('α (heterogeneity)')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate all visualization plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Label metrics
    label_metrics = [
        ('JS_divergence_mean', 'JS_divergence_std', 'JS Divergence', 'JS Divergence'),
        ('mmd_labels_mean', 'mmd_labels_std', 'MMD (Labels)', 'MMD'),
        ('mahalanobis_mean', 'mahalanobis_std', 'Mahalanobis Distance', 'Distance'),
        ('ks_label_statistic_mean', None, 'KS Statistic (Labels)', 'Statistic'),
    ]

    for metric, std, title, ylabel in label_metrics:
        if metric in df.columns and df[metric].notna().any():
            create_metric_comparison_plot(
                df, metric, title, ylabel,
                plots_dir / f"{metric}_comparison.png", std
            )
            create_heatmap(df, metric, title, plots_dir / f"{metric}_heatmap.png")

    # Feature metrics
    feature_metrics = [
        ('fid_mean', 'fid_std', 'FID Score', 'FID'),
        ('mmd_features_mean', 'mmd_features_std', 'MMD (Features)', 'MMD'),
        ('wasserstein_mean', 'wasserstein_std', 'Wasserstein Distance', 'Distance'),
        ('cramer_von_mises_mean', 'cramer_von_mises_std', 'Cramér-von Mises', 'Statistic'),
        ('vendi_score_mean', 'vendi_score_std', 'Vendi Score', 'Score'),
        ('ks_multivariate_reject_rate', None, 'KS Multivariate Reject Rate', 'Rate'),
    ]

    for metric, std, title, ylabel in feature_metrics:
        if metric in df.columns and df[metric].notna().any():
            create_metric_comparison_plot(
                df, metric, title, ylabel,
                plots_dir / f"{metric}_comparison.png", std
            )
            create_heatmap(df, metric, title, plots_dir / f"{metric}_heatmap.png")

    # Balance metrics
    create_metric_comparison_plot(
        df, 'coefficient_of_variation', 'Sample Distribution Balance (CV)',
        'Coefficient of Variation',
        plots_dir / "balance_cv_comparison.png"
    )

    print(f"✓ Visualizations saved to {plots_dir}")


def generate_summary_report(df: pd.DataFrame, output_dir: Path, metrics_mode: str):
    """Generate formatted markdown summary report."""
    report = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report.append("# Partitioning Methods Comparison Report\n")
    report.append(f"**Generated:** {timestamp}\n")
    report.append(f"**Total experiments:** {len(df)}\n")
    report.append(f"**Methods:** {', '.join(df['method'].unique())}\n")
    report.append(f"**Client counts:** {sorted(df['n_clients'].unique())}\n")
    report.append(f"**Heterogeneity values:** {sorted(df['heterogeneity'].unique())}\n")

    # Summary statistics
    report.append("\n---\n")
    report.append("## Summary Statistics\n")

    # Label metrics table
    if metrics_mode in ["label", "both"]:
        report.append("### Label Distribution Metrics\n")
        report.append("| Method | n | α | JS Div | MMD | Mahalanobis | KS Stat |\n")
        report.append("|--------|---|---|--------|-----|-------------|--------|\n")

        for _, row in df.iterrows():
            report.append(
                f"| {row['method']} | {row['n_clients']} | {row['heterogeneity']} | "
                f"{row['JS_divergence_mean']:.4f} | {row['mmd_labels_mean']:.4f} | "
                f"{row['mahalanobis_mean']:.4f} | {row['ks_label_statistic_mean']:.4f} |\n"
            )

    # Feature metrics table
    if metrics_mode in ["feature", "both"] and 'fid_mean' in df.columns and df['fid_mean'].notna().any():
        report.append("\n### Feature Distribution Metrics\n")
        report.append("| Method | n | α | FID | MMD | Wasserstein | CvM | Vendi |\n")
        report.append("|--------|---|---|-----|-----|-------------|-----|-------|\n")

        for _, row in df.iterrows():
            fid = f"{row['fid_mean']:.2f}" if pd.notna(row.get('fid_mean')) else "N/A"
            mmd = f"{row['mmd_features_mean']:.4f}" if pd.notna(row.get('mmd_features_mean')) else "N/A"
            wass = f"{row['wasserstein_mean']:.4f}" if pd.notna(row.get('wasserstein_mean')) else "N/A"
            cvm = f"{row['cramer_von_mises_mean']:.4f}" if pd.notna(row.get('cramer_von_mises_mean')) else "N/A"
            vendi = f"{row['vendi_score_mean']:.2f}" if pd.notna(row.get('vendi_score_mean')) else "N/A"

            report.append(
                f"| {row['method']} | {row['n_clients']} | {row['heterogeneity']} | "
                f"{fid} | {mmd} | {wass} | {cvm} | {vendi} |\n"
            )

    # Balance metrics table
    report.append("\n### Data Balance Metrics\n")
    report.append("| Method | n | α | Mean Samples | Std | Min | Max | CV |\n")
    report.append("|--------|---|---|--------------|-----|-----|-----|----|\n")

    for _, row in df.iterrows():
        report.append(
            f"| {row['method']} | {row['n_clients']} | {row['heterogeneity']} | "
            f"{row['samples_per_client_mean']:.1f} | {row['samples_per_client_std']:.1f} | "
            f"{row['samples_per_client_min']} | {row['samples_per_client_max']} | "
            f"{row['coefficient_of_variation']:.4f} |\n"
        )

    # Method rankings
    report.append("\n---\n")
    report.append("## Method Rankings\n")
    report.append("*Lower is better for distance/divergence metrics, higher is better for Vendi score*\n")

    ranking_metrics = [
        ('JS_divergence_mean', 'JS Divergence', True),
        ('mmd_labels_mean', 'MMD (Labels)', True),
        ('fid_mean', 'FID', True),
        ('vendi_score_mean', 'Vendi Score', False),
        ('coefficient_of_variation', 'Balance (CV)', True),
    ]

    for metric, name, lower_better in ranking_metrics:
        if metric in df.columns and df[metric].notna().any():
            ranked = df.groupby('method')[metric].mean().sort_values(ascending=lower_better)
            report.append(f"\n**{name}:** ")
            rankings = [f"{i+1}. {m} ({v:.4f})" for i, (m, v) in enumerate(ranked.items())]
            report.append(" → ".join(rankings) + "\n")

    # Visualizations reference
    report.append("\n---\n")
    report.append("## Visualizations\n")
    report.append("See the `plots/` directory for:\n")
    report.append("- Bar charts comparing methods across configurations\n")
    report.append("- Heatmaps showing metric values across n_clients × heterogeneity\n")

    # Save report
    report_path = output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write("".join(report))

    print(f"✓ Summary report saved to {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare data partitioning methods")
    parser.add_argument("--dataset", type=str, default="tiny_imagenet")
    parser.add_argument("--base_path", type=str, default="/path/to/gtai/FDS/")
    parser.add_argument("--n_clients", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--heterogeneity", type=float, nargs="+", default=[0.1, 10.0])
    parser.add_argument("--methods", type=str, nargs="+", default=["random", "dirichlet", "vlm"])
    parser.add_argument("--embedding_presets", type=str, nargs="+", default=["default", "fast"])
    parser.add_argument("--cluster_labels_dir", type=str, default="../data_partitioning/legacy_vlm/data/tiny_imagenet")
    parser.add_argument("--metrics", type=str, default="both", choices=["label", "feature", "both"])
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./.inception_cache")

    args = parser.parse_args()
    use_cache = args.use_cache and not args.no_cache

    print("=" * 80)
    print("PARTITIONING METHODS COMPARISON")
    print("=" * 80)

    output_dir = Path(args.output_dir)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    val_dataset = BaseDataset(args.dataset, args.base_path, "val")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Get number of classes
    all_labels = set(train_dataset[i][1] for i in range(len(train_dataset)))
    num_classes = len(all_labels)
    print(f"  Number of classes: {num_classes}")

    # WNID mapping
    wnid_mapping = get_or_create_wnid_mapping(args.dataset, train_dataset)

    # Generate experiments
    experiments = generate_experiment_configs(
        methods=args.methods,
        n_clients_list=args.n_clients,
        heterogeneity_list=args.heterogeneity,
        embedding_presets=args.embedding_presets,
        cluster_labels_dir=args.cluster_labels_dir,
        seed=args.seed
    )

    print(f"\nTotal experiments: {len(experiments)}")

    # Run experiments
    results = []
    for exp_config in tqdm(experiments, desc="Running experiments"):
        exp_id = exp_config.get_experiment_id()
        print(f"\n[{exp_id}]")

        try:
            result = run_experiment(
                exp_config=exp_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                base_path=args.base_path,
                num_classes=num_classes,
                metrics_mode=args.metrics,
                subsample=args.subsample,
                wnid_mapping=wnid_mapping,
                use_cache=use_cache,
                cache_dir=args.cache_dir,
            )
            results.append(result)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Export results
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)

    df = export_results(results, output_dir)
    create_visualizations(df, output_dir)
    generate_summary_report(df, output_dir, args.metrics)

    print("\n" + "=" * 80)
    print("✓ ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()