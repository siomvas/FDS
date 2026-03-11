#!/usr/bin/env python3
"""Execute one bulk-experiment matrix row."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from research import create_manifest, create_result, write_manifest, write_result
from scripts.matrix_utils import REPO_ROOT, dataset_split_env_var, load_matrix, read_partition_client_count, select_rows

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a single matrix task.")
    parser.add_argument("--matrix", type=str, required=True, help="Path to matrix JSONL/JSON file.")
    parser.add_argument("--row-id", type=str, default=None, help="Explicit matrix row_id to run.")
    parser.add_argument("--stage", type=str, default=None, help="Stage filter when using --row-index.")
    parser.add_argument("--resource-class", type=str, default=None, help="Resource-class filter when using --row-index.")
    parser.add_argument("--row-index", type=int, default=None, help="0-based index within the selected row subset.")
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated command without executing it.")
    return parser.parse_args()


def _load_row(args: argparse.Namespace) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = load_matrix(Path(args.matrix))
    if args.row_id:
        for row in rows:
            if row["row_id"] == args.row_id:
                return row, rows
        raise ValueError(f"Matrix row not found: {args.row_id}")

    if args.row_index is None:
        raise ValueError("Provide either --row-id or --row-index with --stage.")
    selected = select_rows(rows, stage=args.stage, resource_class=args.resource_class)
    if args.row_index < 0 or args.row_index >= len(selected):
        raise IndexError(
            f"row-index {args.row_index} is out of range for "
            f"{len(selected)} selected rows (stage={args.stage}, resource_class={args.resource_class})."
        )
    return selected[args.row_index], rows


def _row_lookup(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["row_id"]: row for row in rows}


def _dependency_errors(row: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    lookup = _row_lookup(rows)
    errors = []
    for dep_id in row.get("depends_on", []):
        dep = lookup.get(dep_id)
        if dep is None:
            errors.append(f"Unknown dependency row_id: {dep_id}")
            continue
        result_path = Path(dep["stage_result_dir"]) / "result.json"
        if not result_path.exists():
            errors.append(f"Dependency has not produced a result artifact: {dep_id}")
            continue
        try:
            with open(result_path, "r", encoding="utf-8") as handle:
                status = __import__("json").load(handle)["status"]
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"Dependency result is unreadable: {dep_id}: {exc}")
            continue
        if status != "success":
            errors.append(f"Dependency did not succeed: {dep_id} ({status})")
    return errors


def _string_flag(argv: List[str], name: str, value: Optional[Any]) -> None:
    if value is None:
        return
    argv.extend([name, str(value)])


def _bool_flag(argv: List[str], name: str, enabled: bool) -> None:
    if enabled:
        argv.append(name)


def _cache_command(payload: Dict[str, Any]) -> List[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "embedding" / "cache_embeddings.py"),
        "--dataset",
        str(payload["dataset"]),
        "--base_path",
        str(payload["base_path"]),
        "--model",
        str(payload["model"]),
        "--tokens",
        str(payload.get("tokens", "cls")),
        "--cache_dir",
        str(payload["cache_dir"]),
        "--batch_size",
        str(payload.get("batch_size", 200)),
    ]
    if int(payload.get("k_last", 0)) > 0:
        argv.extend(["--k_last", str(payload["k_last"])])
    if payload.get("proxy") is not None:
        argv.extend(["--proxy", str(payload.get("proxy"))])
    return argv


def _partition_command(payload: Dict[str, Any]) -> List[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_embedding_partitions.py"),
        "--dataset",
        str(payload["dataset"]),
        "--base_path",
        str(payload["base_path"]),
        "--extractor",
        str(payload["extractor"]),
        "--clustering",
        str(payload["clustering"]),
        "--output",
        str(payload["output_partition_path"]),
        "--cache_dir",
        str(payload["cache_dir"]),
        "--seed",
        str(payload.get("seed", 42)),
    ]
    for name in ("device", "batch_size", "num_workers", "proxy", "tokens", "valset_type"):
        _string_flag(argv, f"--{name}", payload.get(name))
    for name in (
        "lastk_k",
        "layer_aggregation",
        "pooling",
        "gem_p",
        "spectral_k",
        "spectral_sigma",
        "lastk_pooling",
        "dim_reduction",
        "dr_dims",
        "umap_neighbors",
        "umap_min_dist",
        "umap_metric",
        "min_cluster_size",
        "min_samples",
        "hdbscan_metric",
        "hdbscan_method",
        "n_clusters",
        "knn_neighbors",
        "knn_metric",
        "spectral_affinity",
        "subset_size_train",
        "subset_size_val",
        "subset_seed",
    ):
        _string_flag(argv, f"--{name}", payload.get(name))
    _bool_flag(argv, "--normalize", bool(payload.get("normalize", False)))
    return argv


def _fl_env(payload: Dict[str, Any]) -> Dict[str, str]:
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in dict(payload.get("env", {})).items()})
    env["DATASETS_ROOT"] = str(payload["base_path"])
    env[dataset_split_env_var(str(payload["dataset"]))] = str(payload["partition_path"])
    return env


def _fl_overrides(payload: Dict[str, Any]) -> List[str]:
    n_clients = read_partition_client_count(Path(payload["partition_path"]))
    overrides = list(payload.get("overrides", []))
    overrides.extend(
        [
            f"fed.n_clients={n_clients}",
            f"name={payload['run_name']}",
            f"paths.output_dir={payload['output_dir']}",
        ]
    )
    return overrides


def _preflight_command(payload: Dict[str, Any]) -> tuple[List[str], Dict[str, str]]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "preflight_fl.py"),
        "--config-dir",
        str(payload["config_dir"]),
        "--config-name",
        str(payload["config_name"]),
        "--client-idx",
        str(payload.get("client_idx", 0)),
        "--max-path-check",
        str(payload.get("max_path_check", 200)),
    ]
    if bool(payload.get("check_paths", False)):
        argv.append("--check-paths")
    for override in _fl_overrides(payload):
        argv.extend(["--override", override])
    return argv, _fl_env(payload)


def _train_command(payload: Dict[str, Any]) -> tuple[List[str], Dict[str, str]]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "fl_training" / "run_experiment.py"),
        "--config-path",
        str(payload["config_dir"]),
        "--config-name",
        str(payload["config_name"]),
        *_fl_overrides(payload),
    ]
    return argv, _fl_env(payload)


def delegated_command(row: Dict[str, Any]) -> tuple[List[str], Dict[str, str], Dict[str, Any]]:
    payload = row["payload"]
    stage = row["stage"]
    artifacts: Dict[str, Any] = {
        "stage_result_dir": row["stage_result_dir"],
    }
    if stage == "cache_embeddings":
        artifacts["cache_dir"] = payload["cache_dir"]
        return _cache_command(payload), dict(os.environ), artifacts
    if stage == "generate_partition":
        artifacts["partition_path"] = payload["output_partition_path"]
        return _partition_command(payload), dict(os.environ), artifacts
    if stage == "preflight_fl":
        artifacts["partition_path"] = payload["partition_path"]
        return *_preflight_command(payload), artifacts
    if stage == "fl_train":
        artifacts["partition_path"] = payload["partition_path"]
        artifacts["workspace_root"] = str(Path(payload["output_dir"]) / payload["run_name"])
        return *_train_command(payload), artifacts
    raise ValueError(f"Unsupported matrix stage: {stage}")


def _write_stage_result(
    row: Dict[str, Any],
    *,
    status: str,
    summary: Dict[str, Any],
    artifacts: Dict[str, Any],
    failure: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = Path(row["stage_result_dir"])
    manifest = create_manifest(
        run_id=row["row_id"],
        script_name="scripts/run_matrix_task.py",
        command=list(sys.argv),
        resolved_config=row,
        dataset={
            "name": row["payload"].get("dataset"),
            "base_path": row["payload"].get("base_path"),
            "split": row["stage"],
        },
        seed=row["payload"].get("seed"),
        artifacts={
            "run_dir": str(run_dir),
            "manifest_path": str(run_dir / "run_manifest.json"),
            "result_path": str(run_dir / "result.json"),
            **artifacts,
        },
    )
    write_manifest(run_dir, manifest)
    write_result(
        run_dir,
        create_result(
            run_id=row["row_id"],
            result_type="audit_run",
            status=status,
            summary=summary,
            artifacts=artifacts,
            failure=failure,
        ),
    )


def run_row(row: Dict[str, Any], rows: List[Dict[str, Any]], *, dry_run: bool = False) -> int:
    dependency_errors = _dependency_errors(row, rows)
    if dependency_errors:
        _write_stage_result(
            row,
            status="failed",
            summary={"stage": row["stage"], "resource_class": row["resource_class"]},
            artifacts={"stage_result_dir": row["stage_result_dir"]},
            failure={
                "error_type": "DependencyError",
                "error_message": "; ".join(dependency_errors),
                "stage": "dependency_check",
            },
        )
        return 1

    command, env, artifacts = delegated_command(row)
    if dry_run:
        log.info("Dry run for %s", row["row_id"])
        log.info("Command: %s", " ".join(command))
        return 0

    Path(row["stage_result_dir"]).mkdir(parents=True, exist_ok=True)
    start = time.time()
    proc = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
    duration = round(time.time() - start, 3)
    status = "success" if proc.returncode == 0 else "failed"
    _write_stage_result(
        row,
        status=status,
        summary={
            "stage": row["stage"],
            "resource_class": row["resource_class"],
            "duration_sec": duration,
            "returncode": proc.returncode,
        },
        artifacts=artifacts,
        failure=None
        if proc.returncode == 0
        else {
            "error_type": "SubprocessError",
            "error_message": f"Delegated command exited with code {proc.returncode}",
            "stage": row["stage"],
        },
    )
    return proc.returncode


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    row, rows = _load_row(args)
    return run_row(row, rows, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
