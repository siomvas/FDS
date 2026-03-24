"""Compare partitioning methods using canonical research artifacts.

This script generates partitioning experiments, evaluates them, and writes one
canonical run artifact per experiment under the chosen output directory.
Aggregate outputs are derived from those artifacts:
- ``results_index.csv`` / ``results_index.json``
- ``report.html``
- ``method_summary.csv``
- ``metric_rankings.csv``
- ``pairwise_deltas.csv``
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loading.dataset import BaseDataset
from evaluation._partition_eval_core import (
    apply_label_mapping,
    compute_balance_metrics,
    compute_feature_metrics,
    compute_label_metrics,
    extract_client_datasets,
    get_or_create_label_mapping,
    persist_partition_eval_result,
    subsample_datasets,
)
from evaluation.statistics import available_metrics, pairwise_method_deltas, rank_methods, summarize_methods
from partitioning.base_splitter import BaseDatasetSplitter, DatasetSplits
from partitioning.dirichlet_splitter import DirichletSplitter
from partitioning.embedding_splitter import EmbeddingSplitter
from partitioning.embedding_utils.config import get_preset_config
from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from partitioning.random_splitter import RandomSplitter
from research import scan_results, write_results_index
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    n_clients: int
    heterogeneity: float
    seed: int
    embedding_preset: Optional[str] = None
    cluster_labels_dir: Optional[str] = None

    def run_id(self, dataset_name: str) -> str:
        preset = f"_{self.embedding_preset}" if self.embedding_preset else ""
        return (
            f"compare_{dataset_name}_{self.method}{preset}_"
            f"n{self.n_clients}_a{self.heterogeneity}_s{self.seed}"
        )

    def display_method(self) -> str:
        if self.method == "embedding" and self.embedding_preset:
            return f"Embedding ({self.embedding_preset})"
        return {
            "random": "Random",
            "dirichlet": "Dirichlet",
            "vlm": "VLM",
            "embedding": "Embedding",
        }.get(self.method, self.method)


def validate_experiment_request(
    *,
    methods: List[str],
    embedding_presets: List[str],
    cluster_labels_dir: Optional[str],
) -> None:
    if "vlm" in methods and not cluster_labels_dir:
        raise ValueError("--cluster_labels_dir is required when --methods includes 'vlm'.")
    if "embedding" in methods and not embedding_presets:
        raise ValueError("--embedding_presets must not be empty when --methods includes 'embedding'.")


def create_splitter(
    exp_config: ExperimentConfig,
    train_dataset: BaseDataset,
    val_dataset: Optional[BaseDataset],
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

    if exp_config.method == "dirichlet":
        return DirichletSplitter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=exp_config.n_clients,
            heterogeneity=exp_config.heterogeneity,
            valset_type="global",
            seed=exp_config.seed,
            min_samples_per_client=10,
        )

    if exp_config.method == "vlm":
        if not exp_config.cluster_labels_dir:
            raise ValueError("VLM method requires cluster_labels_dir.")
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

    if exp_config.method == "embedding":
        preset = exp_config.embedding_preset or "default"
        config = get_preset_config(preset, n_clients=exp_config.n_clients)
        if hasattr(config.model, "proxy") and config.model.proxy is None:
            config.model.proxy = None
        if config.clustering.method == "hdbscan":
            base_size = len(train_dataset) // max(exp_config.n_clients * 3, 1)
            config.clustering.params["min_cluster_size"] = max(
                5, int(base_size * (1.0 - exp_config.heterogeneity))
            )
        elif config.clustering.method == "kmeans":
            config.clustering.params["n_clusters"] = max(
                exp_config.n_clients,
                int(exp_config.n_clients * (1.0 + 2.0 * exp_config.heterogeneity)),
            )
        return EmbeddingSplitter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=exp_config.n_clients,
            heterogeneity=exp_config.heterogeneity,
            valset_type="global",
            config=config,
            seed=exp_config.seed,
        )

    raise ValueError(f"Unknown method: {exp_config.method}")


def run_experiment(
    *,
    exp_config: ExperimentConfig,
    dataset_name: str,
    train_dataset: BaseDataset,
    val_dataset: BaseDataset,
    base_path: str,
    num_classes: int,
    metrics_mode: str,
    subsample: int,
    label_mapping: Optional[Dict[str, int]],
    feature_model: str,
    use_cache: bool,
    cache_dir: str,
    output_dir: Path,
) -> str:
    run_id = exp_config.run_id(dataset_name)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    split_path = run_dir / "split.json"
    stage = "split"
    timing = {"split_time": 0.0, "label_metrics_time": 0.0, "feature_metrics_time": 0.0}

    try:
        splitter = create_splitter(exp_config, train_dataset, val_dataset)
        start = time.time()
        splits = splitter.get_client_splits()
        splitter.validate_splits(splits)
        timing["split_time"] = round(time.time() - start, 4)
        splitter.store_client_splits(str(split_path), format="json")

        use_original_labels = exp_config.method in {"vlm", "embedding"}
        client_datasets = extract_client_datasets(
            splits,
            train_dataset=train_dataset if use_original_labels else None,
            use_original_labels=use_original_labels,
        )
        if label_mapping is not None:
            client_datasets = [apply_label_mapping(dataset, label_mapping) for dataset in client_datasets]

        metrics_payload: Dict[str, Any] = {}

        if metrics_mode in {"label", "both"}:
            stage = "label_metrics"
            start = time.time()
            timing["label_metrics_time"] = round(time.time() - start, 4)

        if metrics_mode in {"feature", "both"}:
            stage = "feature_metrics"
            feature_input = subsample_datasets(client_datasets, subsample, exp_config.seed) if subsample else client_datasets
            start = time.time()
            timing["feature_metrics_time"] = round(time.time() - start, 4)
        feature_input = subsample_datasets(client_datasets, subsample, exp_config.seed) if subsample else client_datasets
        metrics_payload = {
            **compute_balance_metrics(client_datasets),
            **(compute_label_metrics(client_datasets, num_classes) if metrics_mode in {"label", "both"} else {}),
            **(
                compute_feature_metrics(
                    feature_input,
                    dataset_name=dataset_name,
                    base_path=base_path,
                    feature_model=feature_model,
                    batch_size=50,
                    use_cache=use_cache,
                    cache_dir=cache_dir,
                )
                if metrics_mode in {"feature", "both"}
                else {}
            ),
        }
        summary = {
            "method": exp_config.display_method(),
            "method_family": exp_config.method,
            "embedding_preset": exp_config.embedding_preset,
            "n_clients": exp_config.n_clients,
            "heterogeneity": exp_config.heterogeneity,
            "seed": exp_config.seed,
            **timing,
        }
        persist_partition_eval_result(
            output_dir=output_dir,
            run_id=run_id,
            script_name="evaluation/_compare_partitions.py",
            resolved_config={
                "experiment": asdict(exp_config),
                "feature_model": feature_model,
                "metrics_mode": metrics_mode,
                "subsample": subsample,
                "use_cache": use_cache,
                "cache_dir": cache_dir,
            },
            dataset_name=dataset_name,
            base_path=base_path,
            partition_path=split_path,
            method=exp_config.display_method(),
            n_clients=exp_config.n_clients,
            heterogeneity=exp_config.heterogeneity,
            seed=exp_config.seed,
            feature_model=feature_model,
            status="success",
            summary=summary,
            metrics_payload=metrics_payload,
        )
    except Exception as exc:  # noqa: BLE001 - comparison sweep should continue
        log.exception("Experiment failed: %s", run_id)
        persist_partition_eval_result(
            output_dir=output_dir,
            run_id=run_id,
            script_name="evaluation/_compare_partitions.py",
            resolved_config={
                "experiment": asdict(exp_config),
                "feature_model": feature_model,
                "metrics_mode": metrics_mode,
                "subsample": subsample,
                "use_cache": use_cache,
                "cache_dir": cache_dir,
            },
            dataset_name=dataset_name,
            base_path=base_path,
            partition_path=split_path,
            method=exp_config.display_method(),
            n_clients=exp_config.n_clients,
            heterogeneity=exp_config.heterogeneity,
            seed=exp_config.seed,
            feature_model=feature_model,
            status="failed",
            summary={
                "method": exp_config.display_method(),
                "method_family": exp_config.method,
                "embedding_preset": exp_config.embedding_preset,
                "n_clients": exp_config.n_clients,
                "heterogeneity": exp_config.heterogeneity,
                "seed": exp_config.seed,
                **timing,
            },
            metrics_payload={},
            failure={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stage": stage,
            },
        )
    return run_id


def generate_experiment_configs(
    *,
    methods: List[str],
    n_clients_list: List[int],
    heterogeneity_list: List[float],
    embedding_presets: List[str],
    cluster_labels_dir: Optional[str],
    seed: int,
) -> List[ExperimentConfig]:
    experiments: List[ExperimentConfig] = []
    for n_clients in n_clients_list:
        for heterogeneity in heterogeneity_list:
            for method in ("random", "dirichlet"):
                if method in methods:
                    experiments.append(
                        ExperimentConfig(
                            method=method,
                            n_clients=n_clients,
                            heterogeneity=heterogeneity,
                            seed=seed,
                        )
                    )
            if "vlm" in methods:
                experiments.append(
                    ExperimentConfig(
                        method="vlm",
                        n_clients=n_clients,
                        heterogeneity=heterogeneity,
                        seed=seed,
                        cluster_labels_dir=cluster_labels_dir,
                    )
                )
            if "embedding" in methods:
                for preset in embedding_presets:
                    experiments.append(
                        ExperimentConfig(
                            method="embedding",
                            n_clients=n_clients,
                            heterogeneity=heterogeneity,
                            seed=seed,
                            embedding_preset=preset,
                        )
                    )
    return experiments


def _successful_dataframe(output_dir: Path) -> pd.DataFrame:
    rows = scan_results(output_dir, result_type="partition_eval")
    if not rows:
        raise ValueError(f"No canonical partition_eval artifacts found in {output_dir}")
    df = pd.DataFrame(rows)
    successful = df[df["status"] == "success"].copy()
    if successful.empty:
        raise ValueError(f"No successful comparison runs found in {output_dir}")
    return successful


def write_statistics_bundle(output_dir: Path, df: pd.DataFrame) -> None:
    metrics_list = available_metrics(df)
    summarize_methods(df, metrics_list).to_csv(output_dir / "method_summary.csv", index=False)
    rank_methods(df, metrics_list).to_csv(output_dir / "metric_rankings.csv", index=False)
    pairwise_method_deltas(df, metrics_list).to_csv(output_dir / "pairwise_deltas.csv", index=False)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Compare data partitioning methods.")
    parser.add_argument("--dataset", type=str, default="tiny_imagenet")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--n_clients", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--heterogeneity", type=float, nargs="+", default=[0.1, 10.0])
    parser.add_argument("--methods", type=str, nargs="+", default=["random", "dirichlet", "vlm"])
    parser.add_argument("--embedding_presets", type=str, nargs="+", default=["default", "fast"])
    parser.add_argument("--cluster_labels_dir", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="both", choices=["label", "feature", "both"])
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--feature_model", type=str, default="dinov2-small")
    parser.add_argument("--output_dir", type=str, default=str(Path.cwd() / "results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(Path(Path.cwd()) / "embedding_cache"),
    )
    args = parser.parse_args()
    use_cache = args.use_cache and not args.no_cache
    validate_experiment_request(
        methods=args.methods,
        embedding_presets=args.embedding_presets,
        cluster_labels_dir=args.cluster_labels_dir,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading datasets for %s from %s", args.dataset, args.base_path)
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    val_dataset = BaseDataset(args.dataset, args.base_path, "val")
    label_mapping = get_or_create_label_mapping(train_dataset)
    if label_mapping is None:
        num_classes = len(set(train_dataset[i][1] for i in range(len(train_dataset))))
    else:
        num_classes = len(label_mapping)

    experiments = generate_experiment_configs(
        methods=args.methods,
        n_clients_list=args.n_clients,
        heterogeneity_list=args.heterogeneity,
        embedding_presets=args.embedding_presets,
        cluster_labels_dir=args.cluster_labels_dir,
        seed=args.seed,
    )
    log.info("Running %s comparison experiments", len(experiments))

    for exp_config in tqdm(experiments, desc="Comparisons"):
        run_experiment(
            exp_config=exp_config,
            dataset_name=args.dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            base_path=args.base_path,
            num_classes=num_classes,
            metrics_mode=args.metrics,
            subsample=args.subsample,
            label_mapping=label_mapping,
            feature_model=args.feature_model,
            use_cache=use_cache,
            cache_dir=args.cache_dir,
            output_dir=output_dir,
        )

    write_results_index(output_dir, result_type="partition_eval")
    df = _successful_dataframe(output_dir)
    from scripts.generate_report import generate_report

    generate_report(df=df, output_path=output_dir / "report.html", dataset_name=args.dataset)
    write_statistics_bundle(output_dir, df)

    failures = pd.DataFrame(scan_results(output_dir, result_type="partition_eval"))
    failed_rows = failures[failures["status"] != "success"]
    if not failed_rows.empty:
        preview = failed_rows[["run_id", "error_type", "error_message"]].head(5)
        raise RuntimeError(
            "One or more comparison runs failed:\n"
            + "\n".join(
                f"{row.run_id}: {row.error_type}: {row.error_message}" for row in preview.itertuples()
            )
        )
    log.info("Comparison artifacts written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
