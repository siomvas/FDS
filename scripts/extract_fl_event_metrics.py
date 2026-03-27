#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tensorboard.backend.event_processing import event_accumulator  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency presence varies by env
    event_accumulator = None


def _read_summary_rows(summary_csv: Path) -> List[Dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _max_server_round_finished(server_log: Path) -> Optional[int]:
    if not server_log.exists():
        return None
    text = server_log.read_text(encoding="utf-8", errors="ignore")
    matches = [int(match.group(1)) for match in re.finditer(r"Round\s+(\d+)\s+finished\.", text)]
    if not matches:
        return None
    return max(matches)


def _completion_fields(*, configured_rounds: int, max_tb_final_step: Optional[int], max_server_round_finished: Optional[int]) -> Dict[str, object]:
    expected_final_step = max(configured_rounds - 1, -1)
    completion_by_tb = max_tb_final_step is not None and max_tb_final_step >= expected_final_step
    completion_by_server = (
        max_server_round_finished is not None and max_server_round_finished >= expected_final_step
    )
    completed_all_rounds = completion_by_tb or completion_by_server
    completion_source = (
        "tb_and_server"
        if completion_by_tb and completion_by_server
        else "tb"
        if completion_by_tb
        else "server"
        if completion_by_server
        else "incomplete"
    )
    return {
        "configured_rounds": configured_rounds,
        "expected_final_step": expected_final_step,
        "max_tb_final_step": max_tb_final_step,
        "max_server_round_finished": max_server_round_finished,
        "completed_all_rounds": str(bool(completed_all_rounds)).lower(),
        "completion_source": completion_source,
    }


def _iter_event_files(tb_root: Path) -> Iterable[Tuple[str, Path]]:
    if not tb_root.exists():
        return []
    event_files: List[Tuple[str, Path]] = []
    for site_dir in sorted(path for path in tb_root.iterdir() if path.is_dir()):
        site = site_dir.name
        for event_path in sorted(site_dir.glob("events.out.tfevents*")):
            event_files.append((site, event_path))
    return event_files


def _extract_metrics_for_run(tb_root: Path) -> Tuple[List[Dict[str, object]], Optional[int]]:
    rows: List[Dict[str, object]] = []
    max_step: Optional[int] = None
    if event_accumulator is None:
        return rows, max_step
    for site, event_path in _iter_event_files(tb_root):
        try:
            accumulator = event_accumulator.EventAccumulator(
                str(event_path), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
        except Exception:
            continue

        for tag in sorted(accumulator.Tags().get("scalars", [])):
            scalars = accumulator.Scalars(tag)
            if not scalars:
                continue
            final = scalars[-1]
            step = int(final.step)
            value = float(final.value)
            if max_step is None or step > max_step:
                max_step = step
            rows.append(
                {
                    "tag": tag,
                    "final_step": step,
                    "final_value": value,
                    "site": site,
                }
            )
    return rows, max_step


def build_metric_rows(summary_rows: List[Dict[str, str]], complete_only: bool = False) -> List[Dict[str, object]]:
    long_rows: List[Dict[str, object]] = []

    for summary in summary_rows:
        run_name = summary["run_name"]
        run_root = Path(summary["nvflare_result"])
        tb_root = run_root / "server" / "simulate_job" / "tb_events"
        server_log = run_root / "server" / "log.txt"

        metric_rows, max_tb_step = _extract_metrics_for_run(tb_root)
        completion = _completion_fields(
            configured_rounds=_safe_int(summary.get("rounds"), default=0),
            max_tb_final_step=max_tb_step,
            max_server_round_finished=_max_server_round_finished(server_log),
        )

        if complete_only and completion["completed_all_rounds"] != "true":
            continue

        for metric in metric_rows:
            long_rows.append(
                {
                    "job_name": summary.get("job_name", ""),
                    "dataset": summary.get("dataset", ""),
                    "run_name": run_name,
                    "status": summary.get("status", ""),
                    "status_text": summary.get("status_text", ""),
                    "tag": metric["tag"],
                    "final_step": metric["final_step"],
                    "final_value": metric["final_value"],
                    "site": metric["site"],
                    "configured_rounds": completion["configured_rounds"],
                    "expected_final_step": completion["expected_final_step"],
                    "max_tb_final_step": ""
                    if completion["max_tb_final_step"] is None
                    else completion["max_tb_final_step"],
                    "max_server_round_finished": ""
                    if completion["max_server_round_finished"] is None
                    else completion["max_server_round_finished"],
                    "completed_all_rounds": completion["completed_all_rounds"],
                    "completion_source": completion["completion_source"],
                    "workspace_root": summary.get("workspace_root", ""),
                    "nvflare_result": summary.get("nvflare_result", ""),
                }
            )

    long_rows.sort(key=lambda row: (str(row["dataset"]), str(row["run_name"]), str(row["tag"]), str(row["site"])))
    return long_rows


def build_wide_rows(long_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    wide_by_run: Dict[str, Dict[str, object]] = {}

    base_columns = [
        "job_name",
        "dataset",
        "run_name",
        "status",
        "status_text",
        "configured_rounds",
        "expected_final_step",
        "max_tb_final_step",
        "max_server_round_finished",
        "completed_all_rounds",
        "completion_source",
        "workspace_root",
        "nvflare_result",
    ]

    for row in long_rows:
        run_name = str(row["run_name"])
        if run_name not in wide_by_run:
            wide_by_run[run_name] = {key: row.get(key, "") for key in base_columns}
        metric_key = f"metric::{row['site']}::{row['tag']}"
        wide_by_run[run_name][metric_key] = row["final_value"]

    return [wide_by_run[key] for key in sorted(wide_by_run)]


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract final scalar metrics from FL TensorBoard event files.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Path to fl_results_summary.csv containing run_name, rounds, and nvflare_result columns.",
    )
    parser.add_argument(
        "--output-long",
        type=Path,
        required=True,
        help="Destination CSV for long-format metrics.",
    )
    parser.add_argument(
        "--output-wide",
        type=Path,
        required=True,
        help="Destination CSV for wide-format metrics.",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Only include runs that reached configured rounds based on TB/server evidence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_rows = _read_summary_rows(args.summary_csv)
    long_rows = build_metric_rows(summary_rows, complete_only=args.complete_only)
    wide_rows = build_wide_rows(long_rows)

    long_fieldnames = [
        "job_name",
        "dataset",
        "run_name",
        "status",
        "status_text",
        "tag",
        "final_step",
        "final_value",
        "site",
        "configured_rounds",
        "expected_final_step",
        "max_tb_final_step",
        "max_server_round_finished",
        "completed_all_rounds",
        "completion_source",
        "workspace_root",
        "nvflare_result",
    ]

    wide_fieldnames = sorted({key for row in wide_rows for key in row.keys()})

    _write_csv(args.output_long, long_rows, long_fieldnames)
    _write_csv(args.output_wide, wide_rows, wide_fieldnames)

    total_runs = len({row["run_name"] for row in long_rows})
    complete_runs = len({row["run_name"] for row in long_rows if row["completed_all_rounds"] == "true"})
    print(f"wrote {args.output_long}")
    print(f"wrote {args.output_wide}")
    print(f"runs_in_output={total_runs} complete_runs={complete_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
