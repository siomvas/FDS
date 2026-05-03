#!/usr/bin/env python3
"""Generate and evaluate partition heterogeneity matrices.

This script intentionally keeps the workflow in one place:

1. ``generate`` creates reusable split JSON files and a manifest.
2. ``evaluate`` reads that manifest and computes selected metrics only.
3. ``both`` runs generation then evaluation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loading.dataset import BaseDataset
from evaluation import metrics
from partitioning.dirichlet_splitter import DirichletSplitter
from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from partitioning.random_splitter import RandomSplitter

log = logging.getLogger(__name__)

DEFAULT_DATASETS = ["tiny_imagenet", "food101", "ade20k", "pascal_voc"]
DEFAULT_LABEL_DATASETS = {"tiny_imagenet", "food101"}
DEFAULT_ALPHAS = [0.1, 0.5, 1.0]
DEFAULT_N_CLIENTS = [5, 10, 20]
DEFAULT_METHODS = ["random", "label_dirichlet", "vlm_joint"]
VLM_DIR_NAMES = {
    "tiny_imagenet": "tiny_imagenet",
    "food101": "food101",
    "ade20k": "ade20k",
    "pascal_voc": "PascalVOC",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "n_clients",
        "alpha",
        "seed",
        "has_labels",
        "partition_path",
        "base_path",
        "cluster_labels_dir",
        "joint_min_cell_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _infer_dataset_from_partition_path(path: Path) -> Optional[str]:
    parts = {part.lower() for part in path.parts}
    aliases = {
        "tiny_imagenet": {"tiny_imagenet", "tiny-imagenet", "tiny-imagenet-200"},
        "food101": {"food101", "food-101"},
        "ade20k": {"ade20k"},
        "pascal_voc": {"pascal_voc", "pascalvoc", "pascal-voc", "voc", "voc2012"},
    }
    for dataset, names in aliases.items():
        if parts & names:
            return dataset
    return None


def _partition_client_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return len(payload["clients"])


def _dataset_base_path(dataset: str, base_path: str) -> str:
    # BaseDataset expects the common root, not the dataset-specific subdirectory.
    return base_path


def _load_train_dataset(dataset: str, base_path: str) -> BaseDataset:
    return BaseDataset(dataset, _dataset_base_path(dataset, base_path), "train")


def _save_split(splitter: Any, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        log.info("Keeping existing split: %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = splitter._make_serializable(splitter.get_client_splits())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=_json_default)
    log.info("Wrote split: %s", output_path)


def _partition_filename(method: str, n_clients: int, alpha: Optional[float], seed: int, joint_min_cell_count: int) -> str:
    if method == "random":
        return f"random_n{n_clients}_seed{seed}.json"
    if method == "label_dirichlet":
        return f"label_dirichlet_n{n_clients}_alpha{alpha:g}_seed{seed}.json"
    if method == "vlm_joint":
        return f"vlm_joint_n{n_clients}_alpha{alpha:g}_seed{seed}_mincell{joint_min_cell_count}.json"
    raise ValueError(f"Unsupported method: {method}")


def _make_manifest_row(
    *,
    dataset: str,
    method: str,
    n_clients: int,
    alpha: Optional[float],
    seed: int,
    has_labels: bool,
    partition_path: Path,
    base_path: str,
    cluster_labels_dir: Optional[Path] = None,
    joint_min_cell_count: int = 1,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "n_clients": n_clients,
        "alpha": "" if alpha is None else alpha,
        "seed": seed,
        "has_labels": str(has_labels).lower(),
        "partition_path": str(partition_path),
        "base_path": base_path,
        "cluster_labels_dir": "" if cluster_labels_dir is None else str(cluster_labels_dir),
        "joint_min_cell_count": joint_min_cell_count,
    }


def generate_splits(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    split_root = output_root / "splits"
    manifest_rows: List[Dict[str, Any]] = []
    label_datasets = set(args.label_datasets)

    for dataset in args.datasets:
        has_labels = dataset in label_datasets
        log.info("Loading train dataset: %s", dataset)
        train_dataset = _load_train_dataset(dataset, args.base_path)

        for n_clients in args.n_clients:
            if "random" in args.methods:
                method = "random"
                split_path = split_root / dataset / _partition_filename(
                    method, n_clients, None, args.seed, args.joint_min_cell_count
                )
                splitter = RandomSplitter(
                    train_dataset=train_dataset,
                    val_dataset=None,
                    n_clients=n_clients,
                    heterogeneity=0.0,
                    valset_type="none",
                    seed=args.seed,
                )
                _save_split(splitter, split_path, args.force)
                manifest_rows.append(
                    _make_manifest_row(
                        dataset=dataset,
                        method=method,
                        n_clients=n_clients,
                        alpha=None,
                        seed=args.seed,
                        has_labels=has_labels,
                        partition_path=split_path,
                        base_path=args.base_path,
                    )
                )

            if "label_dirichlet" in args.methods and has_labels:
                for alpha in args.alphas:
                    method = "label_dirichlet"
                    split_path = split_root / dataset / _partition_filename(
                        method, n_clients, alpha, args.seed, args.joint_min_cell_count
                    )
                    splitter = DirichletSplitter(
                        train_dataset=train_dataset,
                        val_dataset=None,
                        n_clients=n_clients,
                        heterogeneity=alpha,
                        valset_type="none",
                        seed=args.seed,
                        min_samples_per_client=args.min_samples_per_client,
                    )
                    _save_split(splitter, split_path, args.force)
                    manifest_rows.append(
                        _make_manifest_row(
                            dataset=dataset,
                            method=method,
                            n_clients=n_clients,
                            alpha=alpha,
                            seed=args.seed,
                            has_labels=has_labels,
                            partition_path=split_path,
                            base_path=args.base_path,
                        )
                    )

            if "vlm_joint" in args.methods:
                vlm_dir = Path(args.vlm_root) / VLM_DIR_NAMES[dataset]
                for alpha in args.alphas:
                    method = "vlm_joint"
                    split_path = split_root / dataset / _partition_filename(
                        method, n_clients, alpha, args.seed, args.joint_min_cell_count
                    )
                    splitter = MultivariateDirichletSplitter(
                        train_dataset=train_dataset,
                        val_dataset=None,
                        n_clients=n_clients,
                        heterogeneity=alpha,
                        cluster_labels_dir=str(vlm_dir),
                        valset_type="none",
                        seed=args.seed,
                        min_samples_per_client=args.min_samples_per_client,
                        joint_min_cell_count=args.joint_min_cell_count,
                        joint_drop_order=args.joint_drop_order,
                    )
                    _save_split(splitter, split_path, args.force)
                    manifest_rows.append(
                        _make_manifest_row(
                            dataset=dataset,
                            method=method,
                            n_clients=n_clients,
                            alpha=alpha,
                            seed=args.seed,
                            has_labels=has_labels,
                            partition_path=split_path,
                            base_path=args.base_path,
                            cluster_labels_dir=vlm_dir,
                            joint_min_cell_count=args.joint_min_cell_count,
                        )
                    )

    seen_embedding_keys = set()
    for pattern in args.embedding_partition_glob:
        for split_name in sorted(glob.glob(pattern)):
            split_path = Path(split_name)
            dataset = _infer_dataset_from_partition_path(split_path)
            if dataset is None:
                log.warning("Could not infer dataset for embedding partition, skipping: %s", split_path)
                continue
            n_clients = _partition_client_count(split_path)
            embedding_key = (dataset, n_clients, split_path.stem)
            if embedding_key in seen_embedding_keys:
                log.info("Skipping duplicate embedding split for %s: %s", embedding_key, split_path)
                continue
            seen_embedding_keys.add(embedding_key)
            manifest_rows.append(
                _make_manifest_row(
                    dataset=dataset,
                    method="embedding",
                    n_clients=n_clients,
                    alpha=None,
                    seed=args.seed,
                    has_labels=dataset in label_datasets,
                    partition_path=split_path,
                    base_path=args.base_path,
                )
            )

    manifest_path = output_root / "partition_manifest.csv"
    _write_manifest(manifest_path, manifest_rows)
    log.info("Wrote manifest with %d rows: %s", len(manifest_rows), manifest_path)
    return manifest_path


def _label_mapping(dataset: BaseDataset) -> Optional[Dict[Any, int]]:
    if len(dataset) == 0:
        return None
    first_label = dataset[0][1]
    if isinstance(first_label, int):
        return None
    labels = sorted({dataset[i][1] for i in range(len(dataset))}, key=lambda value: str(value))
    return {label: idx for idx, label in enumerate(labels)}


def _num_classes(dataset: BaseDataset, mapping: Optional[Dict[Any, int]]) -> int:
    if mapping is not None:
        return len(mapping)
    return len({dataset[i][1] for i in range(len(dataset))})


def _dataset_map(dataset: BaseDataset, mapping: Optional[Dict[Any, int]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        result[str(image)] = {"image": str(image), "label": mapping[label] if mapping else label}
    return result


def _load_clients_from_split(partition_path: Path, dataset_map: Dict[str, Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    with open(partition_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    clients: List[List[Dict[str, Any]]] = []
    for client_id in sorted(payload["clients"], key=lambda name: int(name.rsplit("_", 1)[-1])):
        rows = []
        for item in payload["clients"][client_id]["train"]:
            image = item["image"] if isinstance(item, dict) else item
            if image not in dataset_map:
                raise KeyError(f"Partition image missing from dataset map: {image}")
            rows.append(dataset_map[image])
        clients.append(rows)
    return clients


def _validate_partition_coverage(clients: List[List[Dict[str, Any]]], expected_images: Iterable[str]) -> Dict[str, Any]:
    expected = set(expected_images)
    observed: List[str] = [sample["image"] for client in clients for sample in client]
    observed_set = set(observed)
    duplicate_count = len(observed) - len(observed_set)
    missing_count = len(expected - observed_set)
    extra_count = len(observed_set - expected)
    if duplicate_count or missing_count or extra_count:
        raise ValueError(
            "Partition coverage mismatch: "
            f"duplicates={duplicate_count}, missing={missing_count}, extra={extra_count}"
        )
    return {
        "n_samples": len(observed),
        "duplicates": duplicate_count,
        "missing": missing_count,
        "extra": extra_count,
    }


def _balance_metrics(clients: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    sizes = np.array([len(client) for client in clients], dtype=float)
    return {
        "samples_per_client_mean": float(np.mean(sizes)),
        "samples_per_client_std": float(np.std(sizes)),
        "samples_per_client_min": int(np.min(sizes)),
        "samples_per_client_max": int(np.max(sizes)),
        "balance_cv": float(np.std(sizes) / (np.mean(sizes) + 1e-10)),
    }


def _label_metrics(clients: List[List[Dict[str, Any]]], num_classes: int) -> Dict[str, Any]:
    jsd = metrics.JS_divergence(clients, num_classes=num_classes, return_all=True)
    jsd_global = metrics.JSD_client_to_global(clients, num_classes=num_classes, return_all=True)
    tv = metrics.TV_label_distributions(clients, num_classes=num_classes, return_all=True)
    return {
        "JSD_mean": float(np.mean(jsd)),
        "JSD_std": float(np.std(jsd)),
        "JSD_global_mean": float(np.mean(jsd_global)),
        "JSD_global_std": float(np.std(jsd_global)),
        "TV_mean": float(np.mean(tv)),
        "TV_std": float(np.std(tv)),
    }


def _feature_metrics(
    clients: List[List[Dict[str, Any]]],
    *,
    dataset: str,
    base_path: str,
    feature_model: str,
    batch_size: int,
    cache_dir: str,
    seed: int,
    device: Optional[str],
    subsample: int,
) -> Dict[str, Any]:
    eval_clients = _subsample_clients(clients, subsample, seed) if subsample > 0 else clients
    common_kwargs = {
        "dataset_name": dataset,
        "base_path": base_path,
        "batch_size": batch_size,
        "use_cache": True,
        "cache_dir": cache_dir,
        "device": device,
    }
    model_kwargs = {**common_kwargs, "model_name": feature_model}
    fid = metrics.FID_distance(eval_clients, return_all=True, **common_kwargs)
    mmd = metrics.MMD(eval_clients, return_all=True, **model_kwargs)
    return {
        "fid_mean": float(np.mean(fid)),
        "fid_std": float(np.std(fid)),
        "mmd_features_mean": float(np.mean(mmd)),
        "mmd_features_std": float(np.std(mmd)),
    }


def _subsample_clients(clients: List[List[Dict[str, Any]]], max_samples: int, seed: int) -> List[List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    result: List[List[Dict[str, Any]]] = []
    for client in clients:
        if len(client) <= max_samples:
            result.append(client)
            continue
        indices = rng.choice(len(client), size=max_samples, replace=False)
        result.append([client[int(idx)] for idx in indices])
    return result


def _result_path(eval_dir: Path, row: Dict[str, str]) -> Path:
    alpha = row.get("alpha") or "none"
    partition_path = row["partition_path"]
    partition_stem = Path(partition_path).stem.replace(" ", "_")
    partition_hash = hashlib.sha1(partition_path.encode("utf-8")).hexdigest()[:10]
    name = (
        f"{row['dataset']}__{row['method']}__n{row['n_clients']}__"
        f"a{alpha}__s{row['seed']}__{partition_stem}__{partition_hash}.json"
    )
    return eval_dir / "per_partition" / name


def _write_results(eval_dir: Path, results: List[Dict[str, Any]]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = eval_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, default=_json_default) + "\n")

    if not results:
        return
    keys = sorted({key for row in results for key in row})
    csv_path = eval_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    log.info("Wrote evaluation outputs: %s and %s", jsonl_path, csv_path)


def evaluate_manifest(args: argparse.Namespace, manifest_path: Path) -> None:
    rows = _read_manifest(manifest_path)
    eval_dir = Path(args.output_root) / "evaluation"
    per_partition_dir = eval_dir / "per_partition"
    per_partition_dir.mkdir(parents=True, exist_ok=True)
    label_datasets = set(args.label_datasets)

    dataset_cache: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []

    for row in rows:
        dataset = row["dataset"]
        result_path = _result_path(eval_dir, row)
        if result_path.exists() and not args.force:
            with open(result_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            log.info("Using existing result: %s", result_path)
            continue

        if dataset not in dataset_cache:
            log.info("Indexing dataset for evaluation: %s", dataset)
            train_dataset = _load_train_dataset(dataset, row["base_path"] or args.base_path)
            mapping = _label_mapping(train_dataset) if dataset in label_datasets else None
            dataset_cache[dataset] = {
                "dataset": train_dataset,
                "map": _dataset_map(train_dataset, mapping),
                "num_classes": _num_classes(train_dataset, mapping) if dataset in label_datasets else 0,
                "cache_dataset_name": str(getattr(train_dataset, "dataset_name", dataset)),
            }

        partition_path = Path(row["partition_path"])
        dataset_info = dataset_cache[dataset]
        start = time.time()
        status = "success"
        failure = ""
        result: Dict[str, Any] = {
            **row,
            "feature_model": args.feature_model,
            "cache_dir": args.cache_dir,
            "subsample": args.subsample,
        }

        try:
            clients = _load_clients_from_split(partition_path, dataset_info["map"])
            coverage = _validate_partition_coverage(clients, dataset_info["map"].keys())
            result.update(coverage)
            result.update(_balance_metrics(clients))
            if dataset in label_datasets:
                result.update(_label_metrics(clients, dataset_info["num_classes"]))
            result.update(
                _feature_metrics(
                    clients,
                    dataset=dataset_info["cache_dataset_name"],
                    base_path=row["base_path"] or args.base_path,
                    feature_model=args.feature_model,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    seed=int(row["seed"]),
                    device=args.device,
                    subsample=args.subsample,
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve failure row for matrix runs.
            status = "failed"
            failure = f"{type(exc).__name__}: {exc}"
            log.exception("Failed evaluating %s", partition_path)

        result["status"] = status
        result["failure"] = failure
        result["duration_sec"] = round(time.time() - start, 4)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=_json_default)
        results.append(result)

    _write_results(eval_dir, results)
    failures = [row for row in results if row.get("status") != "success"]
    if failures:
        preview = "\n".join(
            f"{row['dataset']} {row['method']} n={row['n_clients']} a={row.get('alpha')}: {row['failure']}"
            for row in failures[:5]
        )
        raise RuntimeError(f"{len(failures)} evaluations failed:\n{preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generate", "evaluate", "both"], default="both")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--label_datasets", nargs="+", default=sorted(DEFAULT_LABEL_DATASETS))
    parser.add_argument("--n_clients", nargs="+", type=int, default=DEFAULT_N_CLIENTS)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument(
        "--embedding_partition_glob",
        action="append",
        default=[],
        help="Optional glob for existing embedding split JSON files to include in the manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_path", default="/workspace/datasets/raw")
    parser.add_argument("--vlm_root", default="/workspace/FDS/partitioning/vlm/data")
    parser.add_argument("--output_root", default="/workspace/runs/joint_vlm_heterogeneity_matrix")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--min_samples_per_client", type=int, default=10)
    parser.add_argument("--joint_min_cell_count", type=int, default=10)
    parser.add_argument("--joint_drop_order", nargs="+", default=None)
    parser.add_argument("--feature_model", default="dinov2-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cache_dir", default="/workspace/caches/fds")
    parser.add_argument("--device", default=None)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.mode in {"generate", "both"}:
        manifest_path = generate_splits(args)
    else:
        if not args.manifest:
            raise ValueError("--manifest is required when --mode=evaluate")
        manifest_path = Path(args.manifest)
    if args.mode in {"evaluate", "both"}:
        evaluate_manifest(args, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
