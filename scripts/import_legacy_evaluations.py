"""Import legacy evaluation outputs into canonical research artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.eval_config import MetricResults, PartitionConfig
from research import build_run_id, create_manifest, create_result, file_sha256, write_manifest, write_result, write_results_index
from research.io import utc_now
from utils.cli_common import add_standard_cli_args

log = logging.getLogger(__name__)

CONFIG_FIELDS = {item.name for item in fields(PartitionConfig)}
METRIC_FIELDS = {item.name for item in fields(MetricResults)}


def _legacy_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*_eval.json"))
    raise ValueError(f"Input must be a legacy evaluation file or directory: {path}")


def _summary_from_legacy(data: Dict[str, Any], source_path: Path, partition_hash: Optional[str]) -> Dict[str, Any]:
    summary = {key: data.get(key) for key in CONFIG_FIELDS if key in data}
    summary["partition_hash"] = partition_hash
    summary["legacy_timestamp"] = data.get("timestamp")
    summary["source_eval_file"] = str(source_path)
    return summary


def _metrics_from_legacy(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: data.get(key) for key in METRIC_FIELDS if key in data}


def import_legacy_file(source_path: Path, output_dir: Path, force: bool = False) -> Optional[str]:
    with open(source_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Legacy evaluation file must contain a JSON object: {source_path}")
    if "partition_path" not in data or "dataset_name" not in data or "method" not in data:
        raise KeyError(f"Legacy evaluation file is missing required fields: {source_path}")

    partition_path = Path(str(data["partition_path"]))
    partition_hash = file_sha256(partition_path) if partition_path.exists() else None
    run_id = build_run_id(
        "partition-eval",
        data.get("dataset_name", "unknown"),
        source_path.stem,
        source=str(source_path.resolve()),
    )
    run_dir = output_dir / run_id
    if (run_dir / "result.json").exists() and not force:
        log.info("Skipping %s (already imported)", source_path.name)
        return None

    summary = _summary_from_legacy(data, source_path, partition_hash)
    metrics = _metrics_from_legacy(data)
    manifest = create_manifest(
        run_id=run_id,
        script_name="scripts/import_legacy_evaluations.py",
        command=list(sys.argv),
        resolved_config={"legacy_source": str(source_path)},
        dataset={
            "name": data["dataset_name"],
            "base_path": None,
            "split": "legacy_import",
        },
        partition={
            "path": str(partition_path),
            "filename": partition_path.name,
            "sha256": partition_hash,
            "method": data.get("method"),
            "n_clients": data.get("n_clients"),
            "heterogeneity": data.get("heterogeneity"),
        },
        model={
            "name": data.get("feature_model"),
            "task_type": "embedding_feature_evaluation",
        },
        seed=data.get("seed"),
        artifacts={
            "run_dir": str(run_dir),
            "manifest_path": str(run_dir / "run_manifest.json"),
            "result_path": str(run_dir / "result.json"),
        },
        extras={
            "legacy_import": {
                "source_file": str(source_path),
                "imported_at_utc": utc_now(),
            }
        },
    )
    write_manifest(run_dir, manifest)
    write_result(
        run_dir,
        create_result(
            run_id=run_id,
            result_type="partition_eval",
            status="success",
            summary=summary,
            metrics=metrics,
            artifacts={
                "source_eval_file": str(source_path),
                "run_dir": str(run_dir),
            },
        ),
    )
    return run_id


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Import legacy evaluation JSON outputs into canonical artifacts.")
    parser.add_argument("--input", type=str, required=True, help="Legacy *_eval.json file or directory")
    add_standard_cli_args(
        parser,
        include_output=True,
        output_required=True,
        output_help="Directory where canonical imported artifacts are written.",
        include_dry_run=True,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    output_dir = Path(args.output)
    files = _legacy_files(source)

    if args.dry_run:
        log.info("[dry-run] Found %s legacy evaluation files", len(files))
        log.info("[dry-run] Output directory: %s", output_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    imported = 0
    failures: List[str] = []
    for path in files:
        try:
            if import_legacy_file(path, output_dir, force=args.force):
                imported += 1
        except Exception as exc:  # noqa: BLE001 - import should continue across files
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            log.exception("Failed to import %s", path)

    write_results_index(output_dir, result_type="partition_eval")
    log.info("Imported %s legacy evaluation files into %s", imported, output_dir)

    if failures:
        preview = failures[:5]
        suffix = "" if len(failures) <= 5 else f"\n... and {len(failures) - 5} more"
        raise RuntimeError("One or more legacy imports failed:\n" + "\n".join(preview) + suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
