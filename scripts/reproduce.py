#!/usr/bin/env python3
"""Portable reproduction entrypoints for reviewer-facing FDS experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ("tiny_imagenet", "food101", "ade20k", "pascal_voc")
CLASSIFICATION_DATASETS = {"tiny_imagenet", "food101"}
DEFAULT_ALPHAS = (0.1, 0.3, 0.5, 1.0, 5.0)
DEFAULT_METHODS = ("random", "dirichlet", "vlm")
DATASET_MODELS = {
    "tiny_imagenet": "resnet18",
    "food101": "resnet18",
    "ade20k": "deeplabv3_resnet50",
    "pascal_voc": "fasterrcnn_mobilenet_v3",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    partitions = subparsers.add_parser("partitions", help="Generate reviewer split JSON files.")
    _add_common_dataset_args(partitions)
    partitions.add_argument("--output-root", type=Path, default=Path("outputs/reproduction"))
    partitions.add_argument("--vlm-root", type=Path, default=Path("resources/vlm_labels"))
    partitions.add_argument("--n-clients", type=int, default=10)
    partitions.add_argument("--alphas", type=str, default=",".join(str(x) for x in DEFAULT_ALPHAS))
    partitions.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    partitions.add_argument("--valset-type", choices=("global", "local", "none"), default="global")
    partitions.add_argument("--seed", type=int, default=42)
    partitions.add_argument("--min-samples-per-client", type=int, default=10)
    partitions.add_argument("--max-retries", type=int, default=1000)
    partitions.add_argument(
        "--assignment-mode",
        choices=("joint_dirichlet", "product_dirichlet_balanced", "hierarchical_dirichlet_balanced"),
        default="product_dirichlet_balanced",
    )
    partitions.add_argument("--joint-min-cell-count", type=int, default=10)
    partitions.add_argument("--skip-existing", action="store_true")
    partitions.add_argument("--dry-run", action="store_true")

    metrics = subparsers.add_parser("metrics", help="Evaluate generated partition files.")
    _add_common_dataset_args(metrics)
    metrics.add_argument("--output-root", type=Path, default=Path("outputs/reproduction"))
    metrics.add_argument("--feature-model", type=str, default="dinov2-small")
    metrics.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("FDS_CACHE_DIR", "embedding_cache")))
    metrics.add_argument("--batch-size", type=int, default=100)
    metrics.add_argument("--subsample", type=int, default=0)
    metrics.add_argument("--force", action="store_true")
    metrics.add_argument("--dry-run", action="store_true")

    fl = subparsers.add_parser("fl", help="Run or print FL training commands for generated splits.")
    _add_common_dataset_args(fl)
    fl.add_argument("--split-root", type=Path, default=Path("outputs/reproduction/splits"))
    fl.add_argument("--output-root", type=Path, default=Path("outputs/reproduction/fl"))
    fl.add_argument("--rounds", type=int, default=100)
    fl.add_argument("--limit", type=int, default=1, help="Maximum split files per dataset.")
    fl.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATASETS_ROOT", "local_data")))
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help="Comma-separated subset of: tiny_imagenet, food101, ade20k, pascal_voc.",
    )


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_datasets(value: str) -> list[str]:
    datasets = _parse_csv(value)
    invalid = [dataset for dataset in datasets if dataset not in DATASETS]
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}. Expected one of {DATASETS}.")
    return datasets


def _parse_alphas(value: str) -> list[float]:
    return [float(item) for item in _parse_csv(value)]


def _parse_methods(value: str) -> list[str]:
    methods = _parse_csv(value)
    if methods == ["all"]:
        return list(DEFAULT_METHODS)
    invalid = [method for method in methods if method not in DEFAULT_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Expected one of {DEFAULT_METHODS}.")
    return methods


def _dry_print(command: Iterable[str], env: dict[str, str] | None = None) -> None:
    prefix = ""
    if env:
        prefix = " ".join(f"{key}={value}" for key, value in sorted(env.items())) + " "
    print(prefix + " ".join(str(part) for part in command))


def _load_dataset_pair(dataset: str, data_root: Path, valset_type: str):
    from data_loading.dataset import BaseDataset

    train_dataset = BaseDataset(dataset, str(data_root), "train")
    val_dataset = None if valset_type == "none" else BaseDataset(dataset, str(data_root), "val")
    return train_dataset, val_dataset


def _write_split(path: Path, splitter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splitter.store_client_splits(str(path), format="json")


def run_partitions(args: argparse.Namespace) -> int:
    from partitioning import DirichletSplitter, MultivariateDirichletSplitter, RandomSplitter

    datasets = _parse_datasets(args.datasets)
    alphas = _parse_alphas(args.alphas)
    methods = _parse_methods(args.methods)
    split_root = args.output_root / "splits"

    if args.dry_run:
        for dataset in datasets:
            for path in _planned_partition_paths(
                dataset=dataset,
                split_root=split_root,
                methods=methods,
                alphas=alphas,
                n_clients=args.n_clients,
                seed=args.seed,
            ):
                print(f"[dry-run] would write {path}")
        return 0

    for dataset in datasets:
        train_dataset, val_dataset = _load_dataset_pair(dataset, args.data_root, args.valset_type)
        dataset_out = split_root / dataset

        if "random" in methods:
            output_path = dataset_out / f"random_n{args.n_clients}_seed{args.seed}.json"
            if output_path.exists() and args.skip_existing:
                print(f"[skip] {output_path}")
            else:
                splitter = RandomSplitter(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    n_clients=args.n_clients,
                    valset_type=args.valset_type,
                    seed=args.seed,
                )
                _write_split(output_path, splitter)
                print(f"[write] {output_path}")

        if "dirichlet" in methods and dataset in CLASSIFICATION_DATASETS:
            for alpha in alphas:
                output_path = dataset_out / f"dirichlet_n{args.n_clients}_alpha{alpha:g}_seed{args.seed}.json"
                if output_path.exists() and args.skip_existing:
                    print(f"[skip] {output_path}")
                    continue
                splitter = DirichletSplitter(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    n_clients=args.n_clients,
                    heterogeneity=alpha,
                    valset_type=args.valset_type,
                    seed=args.seed,
                    min_samples_per_client=args.min_samples_per_client,
                    max_retries=args.max_retries,
                )
                _write_split(output_path, splitter)
                print(f"[write] {output_path}")

        if "vlm" in methods:
            cluster_dir = args.vlm_root / dataset
            if not cluster_dir.exists():
                raise FileNotFoundError(f"VLM labels not found for {dataset}: {cluster_dir}")
            for alpha in alphas:
                output_path = dataset_out / f"vlm_n{args.n_clients}_alpha{alpha:g}_seed{args.seed}.json"
                if output_path.exists() and args.skip_existing:
                    print(f"[skip] {output_path}")
                    continue
                splitter = MultivariateDirichletSplitter(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    n_clients=args.n_clients,
                    heterogeneity=alpha,
                    cluster_labels_dir=str(cluster_dir),
                    valset_type=args.valset_type,
                    seed=args.seed,
                    min_samples_per_client=args.min_samples_per_client,
                    max_retries=args.max_retries,
                    assignment_mode=args.assignment_mode,
                    joint_min_cell_count=args.joint_min_cell_count,
                )
                _write_split(output_path, splitter)
                print(f"[write] {output_path}")

    return 0


def _planned_partition_paths(
    *,
    dataset: str,
    split_root: Path,
    methods: list[str],
    alphas: list[float],
    n_clients: int,
    seed: int,
) -> list[Path]:
    paths = []
    dataset_out = split_root / dataset
    if "random" in methods:
        paths.append(dataset_out / f"random_n{n_clients}_seed{seed}.json")
    if "dirichlet" in methods and dataset in CLASSIFICATION_DATASETS:
        paths.extend(dataset_out / f"dirichlet_n{n_clients}_alpha{alpha:g}_seed{seed}.json" for alpha in alphas)
    if "vlm" in methods:
        paths.extend(dataset_out / f"vlm_n{n_clients}_alpha{alpha:g}_seed{seed}.json" for alpha in alphas)
    return paths


def run_metrics(args: argparse.Namespace) -> int:
    datasets = _parse_datasets(args.datasets)
    status = 0
    for dataset in datasets:
        partition_dir = args.output_root / "splits" / dataset
        output_dir = args.output_root / "metrics" / dataset
        command = [
            sys.executable,
            "-m",
            "evaluation.evaluate_partitions",
            "--partition_dir",
            str(partition_dir),
            "--dataset",
            dataset,
            "--base_path",
            str(args.data_root),
            "--output",
            str(output_dir),
            "--feature_model",
            args.feature_model,
            "--batch_size",
            str(args.batch_size),
            "--subsample",
            str(args.subsample),
            "--cache_dir",
            str(args.cache_dir),
        ]
        if args.force:
            command.append("--force")
        if args.dry_run:
            command.append("--dry-run")
            _dry_print(command)
            continue
        if not partition_dir.exists():
            print(f"[skip] partition directory not found: {partition_dir}", file=sys.stderr)
            status = max(status, 1)
            continue
        status = max(status, subprocess.call(command))
    return status


def _split_env_key(dataset: str) -> str:
    return f"{dataset}_chosen_split_path"


def _split_files(split_root: Path, dataset: str, limit: int) -> list[Path]:
    dataset_dir = split_root / dataset
    if not dataset_dir.exists():
        return []
    files = sorted(dataset_dir.glob("*.json"))
    return files[:limit] if limit > 0 else files


def _infer_client_count(split_path: Path) -> int:
    match = re.search(r"_n(\d+)", split_path.stem)
    if match:
        return int(match.group(1))
    with open(split_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return len(payload.get("clients", {}))


def run_fl(args: argparse.Namespace) -> int:
    datasets = _parse_datasets(args.datasets)
    status = 0
    for dataset in datasets:
        split_files = _split_files(args.split_root, dataset, args.limit)
        if not split_files:
            message = f"[skip] no split files found for {dataset} under {args.split_root / dataset}"
            if args.dry_run:
                print(message)
                continue
            print(message, file=sys.stderr)
            status = max(status, 1)
            continue

        for split_path in split_files:
            n_clients = _infer_client_count(split_path)
            output_dir = args.output_root / dataset / split_path.stem
            command = [
                sys.executable,
                "-m",
                "fl_training.run_experiment",
                f"dataset={dataset}",
                f"model={DATASET_MODELS[dataset]}",
                f"fed.n_clients={n_clients}",
                f"fed.rounds={args.rounds}",
                f"paths.output_dir={output_dir}",
            ]
            env = {
                "DATASETS_ROOT": str(args.data_root),
                _split_env_key(dataset): str(split_path),
            }
            if args.dry_run:
                _dry_print(command, env=env)
                continue
            run_env = os.environ.copy()
            run_env.update(env)
            status = max(status, subprocess.call(command, env=run_env))
    return status


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "partitions":
        return run_partitions(args)
    if args.command == "metrics":
        return run_metrics(args)
    if args.command == "fl":
        return run_fl(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
