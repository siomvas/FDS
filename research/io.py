"""Helpers for creating, validating, and indexing research artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - torch is available in normal runtime
    torch = None  # type: ignore[assignment]

from research.manifest import Manifest
from research.results_schema import Result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "run"


def build_run_id(prefix: str, *parts: Any, source: Optional[str] = None) -> str:
    stem = "-".join(slugify(str(part)) for part in parts if part not in (None, "", False))
    digest_source = source or stem or prefix
    digest = hashlib.sha1(str(digest_source).encode("utf-8")).hexdigest()[:10]
    if stem:
        return f"{slugify(prefix)}-{stem}-{digest}"
    return f"{slugify(prefix)}-{digest}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(args: Sequence[str], cwd: Optional[Path]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_metadata(cwd: Optional[Path] = None) -> Dict[str, Any]:
    cwd = cwd or Path.cwd()
    repo_root = _run_git(["rev-parse", "--show-toplevel"], cwd)
    commit = _run_git(["rev-parse", "HEAD"], cwd)
    dirty = _run_git(["status", "--porcelain"], cwd)
    return {
        "repo_root": repo_root,
        "commit": commit,
        "dirty": None if dirty is None else bool(dirty),
    }


def environment_summary(*, device: Optional[str] = None, gpu_config: Optional[str] = None) -> Dict[str, Any]:
    summary = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "device": device,
        "gpu_config": gpu_config,
        "torch_version": None,
        "cuda_available": None,
    }
    if torch is not None:
        summary["torch_version"] = getattr(torch, "__version__", None)
        summary["cuda_available"] = bool(torch.cuda.is_available())
    return summary


def create_manifest(
    *,
    run_id: str,
    script_name: str,
    command: Sequence[str],
    resolved_config: Dict[str, Any],
    dataset: Dict[str, Any],
    partition: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
    cwd: Optional[Path] = None,
    device: Optional[str] = None,
    gpu_config: Optional[str] = None,
) -> Manifest:
    return Manifest(
        run_id=run_id,
        created_at_utc=utc_now(),
        producer={"script_name": script_name, "command": list(command)},
        git=git_metadata(cwd),
        resolved_config=_normalize(resolved_config),
        dataset=_normalize(dataset),
        partition=_normalize(partition or {}),
        model=_normalize(model or {}),
        seed=seed,
        environment=_normalize(environment_summary(device=device, gpu_config=gpu_config)),
        artifacts=_normalize(artifacts or {}),
        extras=_normalize(extras or {}),
    )


def create_result(
    *,
    run_id: str,
    result_type: str,
    status: str,
    summary: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    failure: Optional[Dict[str, Any]] = None,
) -> Result:
    return Result(
        run_id=run_id,
        result_type=result_type,
        status=status,
        completed_at_utc=utc_now(),
        summary=_normalize(summary or {}),
        metrics=_normalize(metrics or {}),
        artifacts=_normalize(artifacts or {}),
        failure=_normalize(failure),
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def write_manifest(run_dir: Path, manifest: Manifest) -> Path:
    path = run_dir / "run_manifest.json"
    _write_json(path, manifest.to_dict())
    return path


def write_result(run_dir: Path, result: Result) -> Path:
    path = run_dir / "result.json"
    _write_json(path, result.to_dict())
    return path


def load_manifest(path: Path) -> Manifest:
    with open(path, "r", encoding="utf-8") as handle:
        return Manifest.from_dict(json.load(handle))


def load_result(path: Path) -> Result:
    with open(path, "r", encoding="utf-8") as handle:
        return Result.from_dict(json.load(handle))


def _flatten_mapping(mapping: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, full_key))
        else:
            flat[full_key] = value
    return flat


def flatten_result(manifest: Manifest, result: Result, run_dir: Path) -> Dict[str, Any]:
    row = {
        "run_id": result.run_id,
        "result_type": result.result_type,
        "status": result.status,
        "created_at_utc": manifest.created_at_utc,
        "completed_at_utc": result.completed_at_utc,
        "script_name": manifest.producer.get("script_name"),
        "dataset_name": manifest.dataset.get("name"),
        "run_dir": str(run_dir),
        "manifest_path": str(run_dir / "run_manifest.json"),
        "result_path": str(run_dir / "result.json"),
    }
    row.update(_flatten_mapping(result.summary))
    row.update(_flatten_mapping(result.metrics))
    if result.failure:
        row["error_type"] = result.failure.get("error_type")
        row["error_message"] = result.failure.get("error_message")
        row["failure_stage"] = result.failure.get("stage")
    return row


def scan_results(output_root: Path, *, result_type: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result_path in sorted(output_root.rglob("result.json")):
        run_dir = result_path.parent
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing run_manifest.json for result artifact: {result_path}")
        manifest = load_manifest(manifest_path)
        result = load_result(result_path)
        if manifest.run_id != result.run_id:
            raise ValueError(
                f"Mismatched run_id between {manifest_path} and {result_path}: "
                f"{manifest.run_id} != {result.run_id}"
            )
        if result_type and result.result_type != result_type:
            continue
        rows.append(flatten_result(manifest, result, run_dir))
    return rows


def _csv_columns(rows: Iterable[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    return columns


def build_results_index(output_root: Path, *, result_type: Optional[str] = None) -> List[Dict[str, Any]]:
    return scan_results(output_root, result_type=result_type)


def write_results_index(output_root: Path, *, result_type: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = build_results_index(output_root, result_type=result_type)
    json_path = output_root / "results_index.json"
    csv_path = output_root / "results_index.csv"
    _write_json(json_path, {"rows": rows})
    columns = _csv_columns(rows)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return rows
