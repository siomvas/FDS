from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import evaluate_partitions
from evaluation.eval_config import MetricResults
from research import create_manifest, create_result, scan_results, write_manifest, write_result, write_results_index
from research import io as research_io


def _write_canonical_partition_run(root: Path, run_id: str, *, result_type: str = "partition_eval") -> None:
    run_dir = root / run_id
    write_manifest(
        run_dir,
        create_manifest(
            run_id=run_id,
            script_name="tests.fixture",
            command=["pytest"],
            resolved_config={"fixture": True},
            dataset={"name": "tiny_imagenet", "base_path": "/tmp/data", "split": "train"},
            partition={"path": "/tmp/split.json", "filename": "split.json", "sha256": None},
            model={"name": "dinov2-small", "task_type": "embedding_feature_evaluation"},
            seed=1,
            artifacts={},
        ),
    )
    write_result(
        run_dir,
        create_result(
            run_id=run_id,
            result_type=result_type,
            status="success",
            summary={"method": "Random", "n_clients": 2, "heterogeneity": 0.1},
            metrics={"fid_mean": 1.0},
        ),
    )


def test_create_manifest_captures_git_and_environment(monkeypatch):
    monkeypatch.setattr(
        research_io,
        "git_metadata",
        lambda cwd=None: {"repo_root": "/repo", "commit": "abc123", "dirty": False},
    )
    monkeypatch.setattr(
        research_io,
        "environment_summary",
        lambda **kwargs: {"python_version": "3.11.0", "device": kwargs.get("device"), "gpu_config": kwargs.get("gpu_config")},
    )

    manifest = research_io.create_manifest(
        run_id="demo-run",
        script_name="tests.fixture",
        command=["pytest"],
        resolved_config={"seed": 7},
        dataset={"name": "tiny_imagenet"},
        seed=7,
        device="cpu",
        gpu_config="0",
    )

    assert manifest.git["commit"] == "abc123"
    assert manifest.environment["device"] == "cpu"
    assert manifest.resolved_config["seed"] == 7


def test_failed_result_requires_failure_details():
    with pytest.raises(ValueError, match="Failed results must include failure details"):
        create_result(
            run_id="demo-run",
            result_type="partition_eval",
            status="failed",
            summary={},
            metrics={},
        )


def test_write_results_index_filters_to_requested_result_type(tmp_path):
    _write_canonical_partition_run(tmp_path, "partition-run", result_type="partition_eval")
    _write_canonical_partition_run(tmp_path, "audit-run", result_type="audit_run")

    rows = write_results_index(tmp_path, result_type="partition_eval")

    assert len(rows) == 1
    assert rows[0]["result_type"] == "partition_eval"
    assert (tmp_path / "results_index.json").exists()
    assert (tmp_path / "results_index.csv").exists()


def test_evaluate_all_writes_canonical_artifacts_and_continues_after_failure(tmp_path, monkeypatch):
    partition_dir = tmp_path / "splits"
    partition_dir.mkdir(parents=True)
    valid_path = partition_dir / "random_n2_seed7.json"
    invalid_path = partition_dir / "dirichlet_n1_alpha0.1_seed8.json"

    valid_path.write_text(
        json.dumps(
            {
                "clients": {
                    "client_0": {"train": ["img_a.jpg"]},
                    "client_1": {"train": ["img_b.jpg"]},
                }
            }
        ),
        encoding="utf-8",
    )
    invalid_path.write_text(
        json.dumps({"clients": {"client_0": {"train": ["missing.jpg"]}}}),
        encoding="utf-8",
    )

    dataset_map = {
        "img_a.jpg": {"image": "img_a.jpg", "label": 0},
        "img_b.jpg": {"image": "img_b.jpg", "label": 1},
    }

    monkeypatch.setattr(
        evaluate_partitions,
        "compute_all_metrics",
        lambda **kwargs: MetricResults(
            samples_per_client_mean=1.0,
            samples_per_client_std=0.0,
            samples_per_client_min=1,
            samples_per_client_max=1,
            balance_cv=0.0,
            min_size=1,
            max_size=1,
            fid_mean=1.0,
            fid_std=0.0,
        ),
    )

    output_dir = tmp_path / "eval_out"
    with pytest.raises(RuntimeError, match="failed"):
        evaluate_partitions.evaluate_all(
            partition_dir=partition_dir,
            dataset_map=dataset_map,
            dataset_name="tiny_imagenet",
            base_path=str(tmp_path),
            output_dir=output_dir,
            feature_model="dinov2-small",
            batch_size=8,
            use_cache=False,
            cache_dir=None,
            force=False,
            subsample=0,
            feature_only=False,
            seed=7,
        )

    rows = scan_results(output_dir, result_type="partition_eval")
    assert len(rows) == 2
    assert sorted(row["status"] for row in rows) == ["failed", "success"]
    assert (output_dir / "results_index.csv").exists()
    assert (output_dir / "results_index.json").exists()
