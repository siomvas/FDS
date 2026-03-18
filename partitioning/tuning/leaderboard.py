from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_METRICS = [
    "mmd_features_mean",
    "fid_mean",
    "centroid_dist_mean",
    "feature_variance_mean",
]


@dataclass
class RankedTrial:
    number: int
    value: float
    metrics: dict[str, Any]
    params: dict[str, Any]
    artifact_dir: Path
    metric_ranks: dict[str, int]
    rank_sum: int
    rank_mean: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc leaderboard for embedding partition Optuna studies.")
    parser.add_argument("--study_dir", required=True, type=str, help="Path to the concrete study directory.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to rank. Higher is treated as better for all metrics in this first pass.",
    )
    parser.add_argument("--top_k", default=20, type=int, help="Number of ranked rows to emit.")
    parser.add_argument("--output_dir", default=None, type=str, help="Optional output directory override.")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_rankable_trials(study_dir: Path, metrics: list[str]) -> list[dict[str, Any]]:
    trials_root = study_dir / "trials"
    if not trials_root.exists():
        raise FileNotFoundError(f"Study trials directory not found: {trials_root}")

    rows: list[dict[str, Any]] = []
    for trial_dir in sorted(trials_root.glob("trial_*")):
        metrics_path = trial_dir / "metrics.json"
        params_path = trial_dir / "params.json"
        partition_path = trial_dir / "partition.json"
        if not metrics_path.exists() or not params_path.exists() or not partition_path.exists():
            continue
        metrics_payload = _load_json(metrics_path)
        missing = [metric for metric in metrics if metric not in metrics_payload]
        if missing:
            log.warning("Skipping %s due to missing metrics: %s", trial_dir.name, ",".join(missing))
            continue
        params_payload = _load_json(params_path)
        rows.append(
            {
                "number": int(params_payload["trial_number"]),
                "value": float(metrics_payload.get("objective", metrics_payload["mmd_features_mean"])),
                "metrics": metrics_payload,
                "params": params_payload.get("params", {}),
                "artifact_dir": trial_dir,
            }
        )
    if not rows:
        raise ValueError(f"No rankable completed trial artifacts were found in {trials_root}.")
    return rows


def _compute_metric_ranks(rows: list[dict[str, Any]], metrics: list[str]) -> list[RankedTrial]:
    rank_maps: dict[str, dict[int, int]] = {}
    for metric in metrics:
        ordered = sorted(rows, key=lambda row: (float(row["metrics"][metric]), -int(row["number"])), reverse=True)
        rank_maps[metric] = {int(row["number"]): rank for rank, row in enumerate(ordered, start=1)}

    ranked_trials: list[RankedTrial] = []
    for row in rows:
        trial_number = int(row["number"])
        metric_ranks = {metric: rank_maps[metric][trial_number] for metric in metrics}
        rank_sum = sum(metric_ranks.values())
        ranked_trials.append(
            RankedTrial(
                number=trial_number,
                value=float(row["value"]),
                metrics=row["metrics"],
                params=row["params"],
                artifact_dir=row["artifact_dir"],
                metric_ranks=metric_ranks,
                rank_sum=rank_sum,
                rank_mean=rank_sum / max(1, len(metrics)),
            )
        )

    ranked_trials.sort(
        key=lambda trial: (
            trial.rank_sum,
            tuple(trial.metric_ranks[metric] for metric in metrics),
            trial.number,
        )
    )
    return ranked_trials


def _write_csv(path: Path, ranked_trials: list[RankedTrial], metrics: list[str], top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["leaderboard_rank", "trial_number", "rank_sum", "rank_mean", "objective"] + metrics + [f"{metric}_rank" for metric in metrics]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for leaderboard_rank, trial in enumerate(ranked_trials[:top_k], start=1):
            row = {
                "leaderboard_rank": leaderboard_rank,
                "trial_number": trial.number,
                "rank_sum": trial.rank_sum,
                "rank_mean": trial.rank_mean,
                "objective": trial.value,
            }
            for metric in metrics:
                row[metric] = trial.metrics[metric]
                row[f"{metric}_rank"] = trial.metric_ranks[metric]
            writer.writerow(row)


def _write_summary(path: Path, study_dir: Path, ranked_trials: list[RankedTrial], metrics: list[str], top_k: int) -> None:
    best = ranked_trials[0]
    lines = [
        f"# Embedding Partition Leaderboard: {study_dir.name}",
        "",
        f"- Trials ranked: `{len(ranked_trials)}`",
        f"- Metrics: `{metrics}`",
        f"- Top K emitted: `{min(top_k, len(ranked_trials))}`",
        "",
        "## Best By Aggregate Rank",
        "",
        f"- Trial: `{best.number}`",
        f"- Aggregate Rank Sum: `{best.rank_sum}`",
        f"- Aggregate Rank Mean: `{best.rank_mean}`",
        f"- Objective (MMD): `{best.value}`",
    ]
    for metric in metrics:
        lines.append(f"- {metric}: `{best.metrics[metric]}` (rank `{best.metric_ranks[metric]}`)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_best_payload(path: Path, best: RankedTrial, metrics: list[str]) -> None:
    payload = {
        "trial_number": best.number,
        "rank_sum": best.rank_sum,
        "rank_mean": best.rank_mean,
        "objective": best.value,
        "metrics": {metric: best.metrics[metric] for metric in metrics},
        "metric_ranks": best.metric_ranks,
        "params": best.params,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _copy_best_artifacts(output_dir: Path, best: RankedTrial) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["partition.json", "metrics.json", "params.json"]:
        source = best.artifact_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def run_cli(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    study_dir = Path(args.study_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else study_dir / "leaderboard"
    metrics = list(args.metrics)

    rows = _load_rankable_trials(study_dir, metrics)
    ranked_trials = _compute_metric_ranks(rows, metrics)

    _write_csv(output_dir / "leaderboard.csv", ranked_trials, metrics, top_k=int(args.top_k))
    _write_summary(output_dir / "summary.md", study_dir, ranked_trials, metrics, top_k=int(args.top_k))
    _write_best_payload(output_dir / "best_by_rank.json", ranked_trials[0], metrics)
    _copy_best_artifacts(output_dir / "best_by_rank", ranked_trials[0])

    log.info("Leaderboard written to %s", output_dir)
    return 0
