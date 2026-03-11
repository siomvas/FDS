#!/usr/bin/env python3
"""Validate HPC/offline prerequisites for a bulk matrix."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Set

from embedding.embedding_extractors import ModelRegistry
from scripts.matrix_utils import load_matrix

log = logging.getLogger(__name__)

_OFFLINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
_REQUIRED_PATH_ENVS = (
    "FDS_CACHE_DIR",
    "HF_HOME",
    "TORCH_HOME",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate offline HPC prerequisites for a matrix.")
    parser.add_argument("--matrix", type=str, required=True, help="Path to matrix JSONL/JSON file.")
    parser.add_argument("--probe-extractors", action="store_true", help="Instantiate unique extractors in offline mode.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for extractor probing.")
    parser.add_argument(
        "--require-env",
        action="append",
        default=[],
        help="Additional environment variable that must be set. Repeatable.",
    )
    parser.add_argument(
        "--skip-offline-env-check",
        action="store_true",
        help="Skip validation of offline Hugging Face environment flags.",
    )
    return parser.parse_args()


def _collect_extractors(rows: List[Dict[str, object]]) -> List[str]:
    extractors: Set[str] = set()
    for row in rows:
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        extractor = payload.get("extractor") or payload.get("model")
        if isinstance(extractor, str) and extractor.strip():
            extractors.add(extractor)
    return sorted(extractors)


def _collect_paths(rows: List[Dict[str, object]]) -> Dict[str, List[Path]]:
    existing: List[Path] = []
    writable: List[Path] = []
    for row in rows:
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        base_path = payload.get("base_path")
        if isinstance(base_path, str):
            existing.append(Path(base_path))
        for key in ("cache_dir", "output_dir", "stage_result_dir"):
            value = payload.get(key) if key != "stage_result_dir" else row.get("stage_result_dir")
            if isinstance(value, str):
                writable.append(Path(value))
        partition_path = payload.get("output_partition_path")
        if isinstance(partition_path, str):
            writable.append(Path(partition_path).parent)
    return {"existing": existing, "writable": writable}


def run_checks(
    *,
    matrix_path: Path,
    probe_extractors: bool,
    device: str,
    require_env: List[str],
    skip_offline_env_check: bool,
) -> Dict[str, object]:
    rows = load_matrix(matrix_path)
    missing_env = []

    if not skip_offline_env_check:
        for name in _OFFLINE_ENV_VARS:
            value = os.environ.get(name, "")
            if value.lower() not in {"1", "true", "yes"}:
                missing_env.append(name)

    for name in (*_REQUIRED_PATH_ENVS, *require_env):
        if not os.environ.get(name):
            missing_env.append(name)

    if missing_env:
        raise ValueError(f"Missing required HPC environment variables: {sorted(set(missing_env))}")

    paths = _collect_paths(rows)
    for path in paths["existing"]:
        if not path.exists():
            raise FileNotFoundError(f"Required input path does not exist: {path}")
    for path in paths["writable"]:
        path.mkdir(parents=True, exist_ok=True)

    extractors = _collect_extractors(rows)
    if probe_extractors:
        for extractor in extractors:
            ModelRegistry.create(extractor, device=device, proxy="")

    return {
        "rows": len(rows),
        "extractors": extractors,
        "checked_existing_paths": [str(path) for path in paths["existing"]],
        "prepared_writable_paths": [str(path) for path in paths["writable"]],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    summary = run_checks(
        matrix_path=Path(args.matrix).resolve(),
        probe_extractors=args.probe_extractors,
        device=args.device,
        require_env=list(args.require_env or []),
        skip_offline_env_check=args.skip_offline_env_check,
    )
    log.info("Validated matrix with %s rows", summary["rows"])
    if summary["extractors"]:
        log.info("Extractor probes: %s", ", ".join(summary["extractors"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
