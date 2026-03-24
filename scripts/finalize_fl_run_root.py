#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_manifest_map(run_root: Path) -> Dict[str, Dict[str, str]]:
    candidates = sorted(run_root.glob("fl_job_manifest*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {}

    manifest_path = candidates[-1]
    mapping: Dict[str, Dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run_name = row.get("run_name", "").strip()
            if run_name:
                mapping[run_name] = row
    return mapping


def _result_paths(run_root: Path) -> List[Path]:
    fl_root = run_root / "fl"
    paths = set()
    for pattern in ("*/result.json", "*/*/result.json"):
        for path in fl_root.glob(pattern):
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def build_summary_rows(run_root: Path) -> List[Dict[str, object]]:
    manifest_map = _load_manifest_map(run_root)
    rows: List[Dict[str, object]] = []

    for result_path in _result_paths(run_root):
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        summary = payload.get("summary") or {}
        metrics = payload.get("metrics") or {}
        artifacts = payload.get("artifacts") or {}

        run_name = str(summary.get("name") or result_path.parent.name)
        manifest_row = manifest_map.get(run_name, {})

        dataset = manifest_row.get("dataset", "")
        if not dataset:
            prefix = run_name.split("_", 1)[0]
            dataset = prefix

        rows.append(
            {
                "job_name": manifest_row.get("job_name", run_name),
                "dataset": dataset,
                "run_name": run_name,
                "status": str(payload.get("status", "")),
                "rounds": _safe_int(summary.get("rounds"), default=0),
                "n_clients": _safe_int(summary.get("n_clients"), default=0),
                "status_text": str(metrics.get("status_text", "")),
                "workspace_root": str(summary.get("workspace_root") or result_path.parent),
                "nvflare_result": str(artifacts.get("nvflare_result") or result_path.parent),
            }
        )

    rows.sort(key=lambda row: (str(row["dataset"]), str(row["run_name"])))
    return rows


def write_summary(run_root: Path, rows: List[Dict[str, object]]) -> Path:
    summary_path = run_root / "fl_results_summary.csv"
    fieldnames = [
        "job_name",
        "dataset",
        "run_name",
        "status",
        "rounds",
        "n_clients",
        "status_text",
        "workspace_root",
        "nvflare_result",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _max_server_round_finished(server_log: Path) -> Optional[int]:
    if not server_log.exists():
        return None
    text = server_log.read_text(encoding="utf-8", errors="ignore")
    matches = [int(match.group(1)) for match in re.finditer(r"Round\s+(\d+)\s+finished\.", text)]
    if not matches:
        return None
    return max(matches)


def _max_tb_step_for_run(run_root: Path, run_name: str) -> Optional[int]:
    run_dir = run_root / "fl" / run_name / run_name / "server" / "simulate_job" / "tb_events"
    if not run_dir.exists():
        return None

    max_step: Optional[int] = None
    try:
        from tensorboard.backend.event_processing import event_accumulator  # type: ignore[import-not-found]
    except Exception:
        return None

    for event_file in run_dir.rglob("events.out.tfevents*"):
        try:
            accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                scalars = accumulator.Scalars(tag)
                if not scalars:
                    continue
                step = int(scalars[-1].step)
                if max_step is None or step > max_step:
                    max_step = step
        except Exception:
            continue
    return max_step


def write_completion_audit(run_root: Path, summary_rows: List[Dict[str, object]]) -> Path:
    out_path = run_root / "fl_results_completion_audit.csv"
    fieldnames = [
        "job_name",
        "dataset",
        "run_name",
        "configured_rounds",
        "expected_final_step",
        "result_status",
        "status_text",
        "max_tb_final_step",
        "max_server_round_finished",
        "completed_all_rounds",
        "completion_source",
        "workspace_root",
        "nvflare_result",
    ]

    rows: List[Dict[str, object]] = []
    for row in summary_rows:
        run_name = str(row["run_name"])
        rounds = _safe_int(row.get("rounds"), default=0)
        expected_final_step = max(rounds - 1, -1)
        max_tb_final_step = _max_tb_step_for_run(run_root, run_name)

        server_log = run_root / "fl" / run_name / run_name / "server" / "log.txt"
        max_server_round_finished = _max_server_round_finished(server_log)

        completion_by_tb = max_tb_final_step is not None and max_tb_final_step >= expected_final_step
        completion_by_server = (
            max_server_round_finished is not None and max_server_round_finished >= expected_final_step
        )
        completed = completion_by_tb or completion_by_server

        completion_source = (
            "tb_and_server"
            if completion_by_tb and completion_by_server
            else "tb"
            if completion_by_tb
            else "server"
            if completion_by_server
            else "incomplete"
        )

        rows.append(
            {
                "job_name": row.get("job_name", ""),
                "dataset": row.get("dataset", ""),
                "run_name": run_name,
                "configured_rounds": rounds,
                "expected_final_step": expected_final_step,
                "result_status": row.get("status", ""),
                "status_text": row.get("status_text", ""),
                "max_tb_final_step": "" if max_tb_final_step is None else max_tb_final_step,
                "max_server_round_finished": "" if max_server_round_finished is None else max_server_round_finished,
                "completed_all_rounds": str(bool(completed)).lower(),
                "completion_source": completion_source,
                "workspace_root": row.get("workspace_root", ""),
                "nvflare_result": row.get("nvflare_result", ""),
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def run_event_extractor(summary_csv: Path, run_root: Path) -> None:
    script = REPO_ROOT / "scripts" / "extract_fl_event_metrics.py"
    cmd_base = [sys.executable, str(script), "--summary-csv", str(summary_csv)]

    subprocess.run(
        cmd_base
        + [
            "--output-long",
            str(run_root / "fl_results_metrics_from_events_long.csv"),
            "--output-wide",
            str(run_root / "fl_results_metrics_from_events_wide.csv"),
        ],
        check=True,
    )

    subprocess.run(
        cmd_base
        + [
            "--output-long",
            str(run_root / "fl_results_metrics_from_events_long_complete_only.csv"),
            "--output-wide",
            str(run_root / "fl_results_metrics_from_events_wide_complete_only.csv"),
            "--complete-only",
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize FL artifacts for a baseline run root.")
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root

    summary_rows = build_summary_rows(run_root)
    if not summary_rows:
        raise SystemExit(f"No FL result.json files found under {run_root / 'fl'}")

    summary_csv = write_summary(run_root, summary_rows)
    audit_csv = write_completion_audit(run_root, summary_rows)
    run_event_extractor(summary_csv, run_root)

    print(f"wrote {summary_csv}")
    print(f"wrote {audit_csv}")
    print(f"runs={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
