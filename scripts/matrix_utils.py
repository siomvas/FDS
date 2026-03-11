"""Helpers for bulk experimentation matrices."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import OmegaConf

STAGE_ORDER = (
    "cache_embeddings",
    "generate_partition",
    "preflight_fl",
    "fl_train",
)
STAGE_RANK = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def load_spec(path: Path) -> Dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Matrix spec must resolve to a mapping: {path}")
    return normalize(payload)


def load_matrix(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"Matrix JSON must contain a 'rows' list: {path}")
        return [normalize(row) for row in rows]

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(normalize(json.loads(stripped)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row at {path}:{line_no}") from exc
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


def write_matrix(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize(row) for row in rows]
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, sort_keys=False) + "\n")

    csv_path = output_path.with_suffix(".csv")
    csv_rows = []
    for row in normalized:
        flat = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, sort_keys=True)
            else:
                flat[key] = value
        csv_rows.append(flat)

    columns = _csv_columns(csv_rows)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def select_rows(
    rows: List[Dict[str, Any]],
    *,
    stage: Optional[str] = None,
    resource_class: Optional[str] = None,
) -> List[Dict[str, Any]]:
    selected = rows
    if stage is not None:
        selected = [row for row in selected if row.get("stage") == stage]
    if resource_class is not None:
        selected = [row for row in selected if row.get("resource_class") == resource_class]
    return selected


def dataset_split_env_var(dataset_name: str) -> str:
    return f"{str(dataset_name).lower()}_chosen_split_path"


def read_partition_client_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    clients = payload.get("clients")
    if not isinstance(clients, dict) or not clients:
        raise ValueError(f"Partition payload must contain a non-empty 'clients' mapping: {path}")
    return len(clients)


def stage_rank(stage: str) -> int:
    if stage not in STAGE_RANK:
        raise ValueError(f"Unknown matrix stage: {stage}")
    return STAGE_RANK[stage]
