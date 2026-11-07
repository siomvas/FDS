from __future__ import annotations

import csv
import json
from pathlib import Path

from partitioning.tuning.leaderboard import run_cli


def _write_trial(
    study_dir: Path,
    *,
    number: int,
    mmd: float,
    fid: float,
    centroid: float,
    variance: float,
) -> None:
    trial_dir = study_dir / "trials" / f"trial_{number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "partition.json").write_text(json.dumps({"trial": number}), encoding="utf-8")
    (trial_dir / "params.json").write_text(
        json.dumps({"trial_number": number, "params": {"extractor": f"model_{number}"}}),
        encoding="utf-8",
    )
    (trial_dir / "metrics.json").write_text(
        json.dumps(
            {
                "objective": mmd,
                "mmd_features_mean": mmd,
                "fid_mean": fid,
                "centroid_dist_mean": centroid,
                "feature_variance_mean": variance,
            }
        ),
        encoding="utf-8",
    )


def test_leaderboard_ranks_trials_and_copies_best_artifacts(tmp_path):
    study_dir = tmp_path / "tiny_imagenet_fixed_embedding_optuna"
    _write_trial(study_dir, number=0, mmd=0.4, fid=1.0, centroid=0.5, variance=0.2)
    _write_trial(study_dir, number=1, mmd=0.9, fid=1.1, centroid=0.9, variance=0.9)
    _write_trial(study_dir, number=2, mmd=0.7, fid=0.8, centroid=0.7, variance=0.8)

    rc = run_cli(["--study_dir", str(study_dir), "--top_k", "2"])
    assert rc == 0

    leaderboard_dir = study_dir / "leaderboard"
    with open(leaderboard_dir / "leaderboard.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["trial_number"] == "1"
    assert rows[0]["leaderboard_rank"] == "1"

    best_payload = json.loads((leaderboard_dir / "best_by_rank.json").read_text(encoding="utf-8"))
    assert best_payload["trial_number"] == 1
    assert (leaderboard_dir / "best_by_rank" / "partition.json").exists()
    assert (leaderboard_dir / "best_by_rank" / "metrics.json").exists()
    assert (leaderboard_dir / "best_by_rank" / "params.json").exists()


def test_leaderboard_uses_requested_metrics_subset(tmp_path):
    study_dir = tmp_path / "food101_fixed_embedding_optuna"
    _write_trial(study_dir, number=0, mmd=0.95, fid=0.2, centroid=0.3, variance=0.4)
    _write_trial(study_dir, number=1, mmd=0.50, fid=0.9, centroid=0.8, variance=0.7)

    rc = run_cli(
        [
            "--study_dir",
            str(study_dir),
            "--metrics",
            "fid_mean",
            "centroid_dist_mean",
        ]
    )
    assert rc == 0

    best_payload = json.loads((study_dir / "leaderboard" / "best_by_rank.json").read_text(encoding="utf-8"))
    assert best_payload["trial_number"] == 1
    assert sorted(best_payload["metrics"].keys()) == ["centroid_dist_mean", "fid_mean"]


def test_leaderboard_tie_breaks_use_requested_metric_order_not_hidden_mmd(tmp_path):
    study_dir = tmp_path / "ade20k_fixed_embedding_optuna"
    _write_trial(study_dir, number=0, mmd=0.1, fid=0.9, centroid=0.4, variance=0.1)
    _write_trial(study_dir, number=1, mmd=0.99, fid=0.7, centroid=0.8, variance=0.1)

    rc = run_cli(
        [
            "--study_dir",
            str(study_dir),
            "--metrics",
            "fid_mean",
            "centroid_dist_mean",
        ]
    )
    assert rc == 0

    best_payload = json.loads((study_dir / "leaderboard" / "best_by_rank.json").read_text(encoding="utf-8"))
    assert best_payload["trial_number"] == 0
