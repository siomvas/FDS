from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_loading.dataset import BaseDataset
from evaluation._compare_partitions import write_statistics_bundle
from evaluation._partition_eval_core import (
    apply_label_mapping,
    compute_feature_metrics,
    compute_label_metrics,
    extract_client_datasets,
    get_or_create_label_mapping,
    persist_partition_eval_result,
    subsample_datasets,
)
from partitioning.base_splitter import DatasetSplits
from partitioning.embedding_splitter import EmbeddingSplitter
from partitioning.embedding_utils.config import get_preset_config
from research import scan_results, write_results_index

log = logging.getLogger(__name__)


def generate_report(df, output_path, dataset_name):
    from scripts.generate_report import generate_report as _generate_report

    return _generate_report(df=df, output_path=output_path, dataset_name=dataset_name)


@dataclass(frozen=True)
class NaturalComparisonConfig:
    dataset: str
    base_path: str
    embedding_preset: str
    target_n_clients: Optional[int]
    heterogeneity: float
    metrics_mode: str
    feature_model: str
    subsample: int
    use_cache: bool
    cache_dir: str
    seed: int


def _sanitize_client_id(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Natural client id cannot be empty.")
    return text.replace("/", "_")


def build_natural_splits(train_dataset: BaseDataset) -> DatasetSplits:
    clients: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for idx in range(len(train_dataset)):
        metadata = train_dataset.get_metadata(idx)
        natural_client_id = metadata.get("natural_client_id")
        if natural_client_id is None:
            raise ValueError(
                f"Dataset '{train_dataset.dataset_name}' does not expose natural_client_id metadata."
            )
        client_name = _sanitize_client_id(str(natural_client_id))
        if client_name not in clients:
            clients[client_name] = {"train": [], "val": []}
        clients[client_name]["train"].append({"image": train_dataset.data[idx], "label": train_dataset.labels[idx]})
    return {"clients": clients, "global_val": None}


def _store_splits(path: Path, splits: DatasetSplits) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)


def _evaluate_partition(
    *,
    output_dir: Path,
    run_id: str,
    method: str,
    partition_path: Path,
    splits: DatasetSplits,
    dataset_name: str,
    base_path: str,
    train_dataset: BaseDataset,
    num_classes: int,
    metrics_mode: str,
    feature_model: str,
    subsample: int,
    use_cache: bool,
    cache_dir: str,
    seed: int,
) -> Dict[str, Any]:
    timing = {"label_metrics_time": 0.0, "feature_metrics_time": 0.0}
    label_mapping = get_or_create_label_mapping(train_dataset)
    client_datasets = extract_client_datasets(splits)
    if label_mapping is not None:
        client_datasets = [apply_label_mapping(dataset, label_mapping) for dataset in client_datasets]

    if metrics_mode in {"label", "both"}:
        start = time.time()
        label_metrics = compute_label_metrics(client_datasets, num_classes)
        timing["label_metrics_time"] = round(time.time() - start, 4)
    else:
        label_metrics = {}

    if metrics_mode in {"feature", "both"}:
        feature_input = subsample_datasets(client_datasets, subsample, seed) if subsample else client_datasets
        start = time.time()
        feature_metrics = compute_feature_metrics(
            feature_input,
            dataset_name=dataset_name,
            base_path=base_path,
            feature_model=feature_model,
            batch_size=50,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
        timing["feature_metrics_time"] = round(time.time() - start, 4)
    else:
        feature_metrics = {}

    sizes = [len(client_dataset) for client_dataset in client_datasets]
    metrics_payload = {
        "samples_per_client_mean": float(sum(sizes) / max(len(sizes), 1)),
        "samples_per_client_std": float(pd.Series(sizes).std(ddof=0)) if sizes else 0.0,
        "samples_per_client_min": int(min(sizes)) if sizes else 0,
        "samples_per_client_max": int(max(sizes)) if sizes else 0,
        "balance_cv": float((pd.Series(sizes).std(ddof=0) / ((sum(sizes) / len(sizes)) + 1e-10))) if sizes else 0.0,
        "min_size": int(min(sizes)) if sizes else 0,
        "max_size": int(max(sizes)) if sizes else 0,
        **label_metrics,
        **feature_metrics,
    }
    summary = {
        "method": method,
        "method_family": method.lower(),
        "n_clients": len(splits["clients"]),
        "seed": seed,
        **timing,
    }
    persist_partition_eval_result(
        output_dir=output_dir,
        run_id=run_id,
        script_name="evaluation/_natural_partition_compare.py",
        resolved_config={
            "dataset_name": dataset_name,
            "feature_model": feature_model,
            "metrics_mode": metrics_mode,
            "subsample": subsample,
            "use_cache": use_cache,
            "cache_dir": cache_dir,
        },
        dataset_name=dataset_name,
        base_path=base_path,
        partition_path=partition_path,
        method=method,
        n_clients=len(splits["clients"]),
        heterogeneity=None,
        seed=seed,
        feature_model=feature_model,
        status="success",
        summary=summary,
        metrics_payload=metrics_payload,
    )
    return metrics_payload


def run_comparison(cfg: NaturalComparisonConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = BaseDataset(cfg.dataset, cfg.base_path, "train")
    natural_splits = build_natural_splits(train_dataset)
    natural_partition_path = output_dir / "natural_partition.json"
    _store_splits(natural_partition_path, natural_splits)

    natural_n_clients = len(natural_splits["clients"])
    target_n_clients = int(cfg.target_n_clients or natural_n_clients)
    embed_config = get_preset_config(cfg.embedding_preset, n_clients=target_n_clients)
    embedding_splitter = EmbeddingSplitter(
        train_dataset=train_dataset,
        val_dataset=None,
        n_clients=target_n_clients,
        heterogeneity=float(cfg.heterogeneity),
        valset_type="none",
        config=embed_config,
        seed=cfg.seed,
    )
    embedding_splits = embedding_splitter.get_client_splits()
    embedding_partition_path = output_dir / "embedding_partition.json"
    embedding_splitter.store_client_splits(str(embedding_partition_path), format="json")

    label_mapping = get_or_create_label_mapping(train_dataset)
    num_classes = len(label_mapping) if label_mapping is not None else len(set(train_dataset.labels))
    natural_metrics = _evaluate_partition(
        output_dir=output_dir,
        run_id=f"natural_{cfg.dataset}_s{cfg.seed}",
        method="Natural",
        partition_path=natural_partition_path,
        splits=natural_splits,
        dataset_name=cfg.dataset,
        base_path=cfg.base_path,
        train_dataset=train_dataset,
        num_classes=num_classes,
        metrics_mode=cfg.metrics_mode,
        feature_model=cfg.feature_model,
        subsample=cfg.subsample,
        use_cache=cfg.use_cache,
        cache_dir=cfg.cache_dir,
        seed=cfg.seed,
    )
    embedding_metrics = _evaluate_partition(
        output_dir=output_dir,
        run_id=f"embedding_{cfg.dataset}_{cfg.embedding_preset}_s{cfg.seed}",
        method=f"Embedding ({cfg.embedding_preset})",
        partition_path=embedding_partition_path,
        splits=embedding_splits,
        dataset_name=cfg.dataset,
        base_path=cfg.base_path,
        train_dataset=train_dataset,
        num_classes=num_classes,
        metrics_mode=cfg.metrics_mode,
        feature_model=cfg.feature_model,
        subsample=cfg.subsample,
        use_cache=cfg.use_cache,
        cache_dir=cfg.cache_dir,
        seed=cfg.seed,
    )

    rows = []
    for metric_name in sorted(set(natural_metrics) | set(embedding_metrics)):
        natural_value = natural_metrics.get(metric_name)
        embedding_value = embedding_metrics.get(metric_name)
        delta = None
        if isinstance(natural_value, (int, float)) and isinstance(embedding_value, (int, float)):
            delta = float(embedding_value - natural_value)
        rows.append(
            {
                "metric": metric_name,
                "natural": natural_value,
                "embedding": embedding_value,
                "delta_embedding_minus_natural": delta,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)

    write_results_index(output_dir, result_type="partition_eval")
    df = pd.DataFrame(scan_results(output_dir, result_type="partition_eval"))
    successful = df[df["status"] == "success"].copy()
    generate_report(df=successful, output_path=output_dir / "report.html", dataset_name=cfg.dataset)
    write_statistics_bundle(output_dir, successful)
    return output_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Compare natural federated partitions against embedding repartitions.")
    parser.add_argument("--dataset", type=str, required=True, choices=["fed_isic2019", "femnist"])
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--embedding_preset", type=str, default="default")
    parser.add_argument("--target_n_clients", type=int, default=None)
    parser.add_argument("--heterogeneity", type=float, default=1.0)
    parser.add_argument("--metrics", type=str, default="feature", choices=["label", "feature", "both"])
    parser.add_argument("--feature_model", type=str, default="dinov2-small")
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=str(Path.cwd() / "embedding_cache"))
    parser.add_argument("--output_dir", type=str, default=str(Path.cwd() / "results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = NaturalComparisonConfig(
        dataset=args.dataset,
        base_path=args.base_path,
        embedding_preset=args.embedding_preset,
        target_n_clients=args.target_n_clients,
        heterogeneity=args.heterogeneity,
        metrics_mode=args.metrics,
        feature_model=args.feature_model,
        subsample=args.subsample,
        use_cache=bool(args.use_cache and not args.no_cache),
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    run_comparison(cfg, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
