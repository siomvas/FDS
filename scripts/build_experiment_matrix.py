#!/usr/bin/env python3
"""Build a bulk-experiment matrix for embedding and FL sweeps."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from research import build_run_id
from scripts.matrix_utils import STAGE_ORDER, load_spec, stage_rank, write_matrix

log = logging.getLogger(__name__)

_DEFAULT_RESOURCE_CLASS = {
    "cache_embeddings": "gpu-embed",
    "generate_partition": "gpu-embed",
    "preflight_fl": "cpu-light",
    "fl_train": "gpu-fl",
}
_ALLOWED_STAGE_KEYS = {
    "name",
    "extractor",
    "device",
    "batch_size",
    "num_workers",
    "proxy",
    "tokens",
    "lastk_k",
    "layer_aggregation",
    "pooling",
    "gem_p",
    "spectral_k",
    "spectral_sigma",
    "lastk_pooling",
    "normalize",
    "dim_reduction",
    "dr_dims",
    "umap_neighbors",
    "umap_min_dist",
    "umap_metric",
    "clustering",
    "min_cluster_size",
    "min_samples",
    "hdbscan_metric",
    "hdbscan_method",
    "n_clusters",
    "knn_neighbors",
    "knn_metric",
    "spectral_affinity",
    "valset_type",
    "subset_size_train",
    "subset_size_val",
    "subset_seed",
    "seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bulk experimentation matrix from a YAML/JSON spec.")
    parser.add_argument("--spec", type=str, required=True, help="Path to matrix YAML/JSON spec.")
    parser.add_argument(
        "--output",
        "--output_dir",
        dest="output",
        type=str,
        default=None,
        help="Output JSONL path for the matrix. Defaults to <matrix_root>/matrix.jsonl.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print matrix summary only.")
    return parser.parse_args()


def _merged(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _resource_settings(
    resources: Dict[str, Dict[str, Any]],
    *,
    stage: str,
    row_config: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    resource_class = str(row_config.get("resource_class") or _DEFAULT_RESOURCE_CLASS[stage])
    slurm = resources.get(resource_class, {})
    if slurm and not isinstance(slurm, dict):
        raise ValueError(f"Resource class '{resource_class}' must map to a settings dict.")
    return resource_class, dict(slurm)


def _require_mapping(spec: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = spec.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping.")
    return value


def _require_list(spec: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = spec.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be a list.")
    return value


def _cache_identity(config: Dict[str, Any], *, dataset: str, base_path: str) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "base_path": base_path,
        "model": config["extractor"],
        "tokens": config.get("tokens", "cls"),
        "k_last": int(config.get("lastk_k") or 0),
    }


def _build_cache_row(
    *,
    matrix_name: str,
    matrix_root: Path,
    dataset: str,
    base_path: str,
    cache_dir: str,
    config: Dict[str, Any],
    resources: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    identity = _cache_identity(config, dataset=dataset, base_path=base_path)
    source = json.dumps(identity, sort_keys=True)
    row_id = build_run_id("cache", matrix_name, config.get("name", config["extractor"]), source=source)
    resource_class, slurm = _resource_settings(resources, stage="cache_embeddings", row_config=config)
    payload = {
        **identity,
        "batch_size": int(config.get("batch_size") or 200),
        "cache_dir": cache_dir,
        "proxy": config.get("proxy"),
    }
    return {
        "row_id": row_id,
        "matrix_name": matrix_name,
        "stage": "cache_embeddings",
        "stage_rank": stage_rank("cache_embeddings"),
        "resource_class": resource_class,
        "slurm": slurm,
        "depends_on": [],
        "stage_result_dir": str(matrix_root / "stage_results" / row_id),
        "payload": payload,
    }


def _validate_embedding_config(config: Dict[str, Any]) -> None:
    if "name" not in config or not str(config["name"]).strip():
        raise ValueError("Each embedding_runs entry requires a non-empty 'name'.")
    if "extractor" not in config or not str(config["extractor"]).strip():
        raise ValueError(f"Embedding run '{config.get('name', '<unknown>')}' is missing 'extractor'.")
    if "clustering" not in config or not str(config["clustering"]).strip():
        raise ValueError(f"Embedding run '{config['name']}' is missing 'clustering'.")

    unknown = set(config) - (_ALLOWED_STAGE_KEYS | {"resource_class", "materialize_embeddings"})
    if unknown:
        raise ValueError(f"Embedding run '{config['name']}' has unsupported keys: {sorted(unknown)}")


def _build_partition_row(
    *,
    matrix_name: str,
    matrix_root: Path,
    partition_root: Path,
    dataset: str,
    base_path: str,
    cache_dir: str,
    config: Dict[str, Any],
    resources: Dict[str, Dict[str, Any]],
    depends_on: List[str],
) -> Dict[str, Any]:
    source = json.dumps({"dataset": dataset, "base_path": base_path, **config}, sort_keys=True)
    row_id = build_run_id("partition", matrix_name, config["name"], source=source)
    resource_class, slurm = _resource_settings(resources, stage="generate_partition", row_config=config)
    output_partition_path = partition_root / f"{row_id}.json"
    payload = {
        "dataset": dataset,
        "base_path": base_path,
        "output_partition_path": str(output_partition_path),
        "cache_dir": cache_dir,
    }
    for key in _ALLOWED_STAGE_KEYS:
        if key in config:
            payload[key] = config[key]
    return {
        "row_id": row_id,
        "matrix_name": matrix_name,
        "stage": "generate_partition",
        "stage_rank": stage_rank("generate_partition"),
        "resource_class": resource_class,
        "slurm": slurm,
        "depends_on": list(depends_on),
        "stage_result_dir": str(matrix_root / "stage_results" / row_id),
        "payload": payload,
    }


def _validate_fl_config(config: Dict[str, Any]) -> None:
    if "name" not in config or not str(config["name"]).strip():
        raise ValueError("Each fl_runs entry requires a non-empty 'name'.")
    overrides = config.get("overrides", [])
    if not isinstance(overrides, list):
        raise ValueError(f"FL run '{config['name']}' overrides must be a list.")


def _build_preflight_row(
    *,
    matrix_name: str,
    matrix_root: Path,
    fl_output_root: Path,
    dataset: str,
    base_path: str,
    partition_row: Dict[str, Any],
    fl_defaults: Dict[str, Any],
    fl_config: Dict[str, Any],
    resources: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "dataset": dataset,
        "base_path": base_path,
        "partition_path": partition_row["payload"]["output_partition_path"],
        "config_dir": fl_config.get("config_dir") or fl_defaults.get("config_dir") or str(Path("fl_training") / "conf"),
        "config_name": fl_config.get("config_name") or fl_defaults.get("config_name") or "config",
        "client_idx": int(fl_config.get("client_idx", fl_defaults.get("client_idx", 0))),
        "check_paths": bool(fl_config.get("check_paths", fl_defaults.get("check_paths", False))),
        "max_path_check": int(fl_config.get("max_path_check", fl_defaults.get("max_path_check", 200))),
        "overrides": list(fl_defaults.get("overrides", [])) + list(fl_config.get("overrides", [])),
        "env": dict(fl_defaults.get("env", {})) | dict(fl_config.get("env", {})),
        "run_name": build_run_id("fl", matrix_name, fl_config["name"], partition_row["row_id"]),
        "output_dir": str(fl_output_root),
    }
    source = json.dumps(payload, sort_keys=True)
    row_id = build_run_id("preflight", matrix_name, fl_config["name"], partition_row["row_id"], source=source)
    resource_class, slurm = _resource_settings(resources, stage="preflight_fl", row_config=fl_config)
    return {
        "row_id": row_id,
        "matrix_name": matrix_name,
        "stage": "preflight_fl",
        "stage_rank": stage_rank("preflight_fl"),
        "resource_class": resource_class,
        "slurm": slurm,
        "depends_on": [partition_row["row_id"]],
        "stage_result_dir": str(matrix_root / "stage_results" / row_id),
        "payload": payload,
    }


def _build_fl_row(
    *,
    matrix_name: str,
    matrix_root: Path,
    preflight_row: Dict[str, Any],
    fl_config: Dict[str, Any],
    resources: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    payload = dict(preflight_row["payload"])
    source = json.dumps(payload, sort_keys=True)
    row_id = build_run_id("train", matrix_name, fl_config["name"], payload["partition_path"], source=source)
    resource_class, slurm = _resource_settings(resources, stage="fl_train", row_config=fl_config)
    return {
        "row_id": row_id,
        "matrix_name": matrix_name,
        "stage": "fl_train",
        "stage_rank": stage_rank("fl_train"),
        "resource_class": resource_class,
        "slurm": slurm,
        "depends_on": [preflight_row["row_id"]],
        "stage_result_dir": str(matrix_root / "stage_results" / row_id),
        "payload": payload,
    }


def build_rows(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    matrix_name = str(spec.get("name") or "wave")
    dataset = str(spec["dataset"])
    base_path = str(spec["base_path"])
    matrix_root = Path(str(spec["matrix_root"])).resolve()
    partition_root = Path(str(spec.get("partition_root") or (matrix_root / "partitions"))).resolve()
    fl_output_root = Path(str(spec.get("fl_output_root") or (matrix_root / "fl"))).resolve()
    cache_dir = str(spec.get("cache_dir") or os.environ.get("FDS_CACHE_DIR") or (matrix_root / "embedding_cache"))
    resources = _require_mapping(spec, "resources")
    embedding_defaults = _require_mapping(spec, "embedding_defaults")
    embedding_runs = _require_list(spec, "embedding_runs")
    fl_defaults = _require_mapping(spec, "fl_defaults")
    fl_runs = _require_list(spec, "fl_runs")

    if not dataset.strip():
        raise ValueError("Spec must include a non-empty 'dataset'.")
    if not base_path.strip():
        raise ValueError("Spec must include a non-empty 'base_path'.")
    if not embedding_runs:
        raise ValueError("Spec must include at least one embedding_runs entry.")

    rows: List[Dict[str, Any]] = []
    cache_row_ids: Dict[str, str] = {}
    partition_rows: List[Dict[str, Any]] = []

    for run in embedding_runs:
        if not isinstance(run, dict):
            raise ValueError("Each embedding_runs entry must be a mapping.")
        config = _merged(embedding_defaults, run)
        _validate_embedding_config(config)
        materialize_embeddings = bool(config.get("materialize_embeddings", True))
        depends_on: List[str] = []

        if materialize_embeddings:
            cache_identity = json.dumps(_cache_identity(config, dataset=dataset, base_path=base_path), sort_keys=True)
            cache_row_id = cache_row_ids.get(cache_identity)
            if cache_row_id is None:
                cache_row = _build_cache_row(
                    matrix_name=matrix_name,
                    matrix_root=matrix_root,
                    dataset=dataset,
                    base_path=base_path,
                    cache_dir=cache_dir,
                    config=config,
                    resources=resources,
                )
                cache_row_id = cache_row["row_id"]
                cache_row_ids[cache_identity] = cache_row_id
                rows.append(cache_row)
            depends_on.append(cache_row_id)

        partition_row = _build_partition_row(
            matrix_name=matrix_name,
            matrix_root=matrix_root,
            partition_root=partition_root,
            dataset=dataset,
            base_path=base_path,
            cache_dir=cache_dir,
            config=config,
            resources=resources,
            depends_on=depends_on,
        )
        rows.append(partition_row)
        partition_rows.append(partition_row)

    for run in fl_runs:
        if not isinstance(run, dict):
            raise ValueError("Each fl_runs entry must be a mapping.")
        _validate_fl_config(run)
        for partition_row in partition_rows:
            preflight_row = _build_preflight_row(
                matrix_name=matrix_name,
                matrix_root=matrix_root,
                fl_output_root=fl_output_root,
                dataset=dataset,
                base_path=base_path,
                partition_row=partition_row,
                fl_defaults=fl_defaults,
                fl_config=run,
                resources=resources,
            )
            rows.append(preflight_row)
            rows.append(
                _build_fl_row(
                    matrix_name=matrix_name,
                    matrix_root=matrix_root,
                    preflight_row=preflight_row,
                    fl_config=run,
                    resources=resources,
                )
            )

    return sorted(rows, key=lambda row: (row["stage_rank"], row["row_id"]))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    spec_path = Path(args.spec).resolve()
    spec = load_spec(spec_path)
    rows = build_rows(spec)
    matrix_root = Path(str(spec["matrix_root"])).resolve()
    output_path = Path(args.output).resolve() if args.output else matrix_root / "matrix.jsonl"

    if args.dry_run:
        counts = {stage: sum(1 for row in rows if row["stage"] == stage) for stage in STAGE_ORDER}
        log.info("Validated matrix spec: %s", spec_path)
        log.info("Planned output: %s", output_path)
        for stage in STAGE_ORDER:
            if counts.get(stage):
                log.info("%s: %s rows", stage, counts[stage])
        return 0

    write_matrix(output_path, rows)
    log.info("Wrote %s matrix rows to %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
