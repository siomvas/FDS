from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from research import create_manifest, create_result, write_manifest, write_result


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_success_stage_result(stage_result_dir: Path, run_id: str) -> None:
    write_manifest(
        stage_result_dir,
        create_manifest(
            run_id=run_id,
            script_name="tests.fixture",
            command=["pytest"],
            resolved_config={"fixture": True},
            dataset={"name": "tiny_imagenet", "base_path": "/tmp/data", "split": "fixture"},
            artifacts={},
        ),
    )
    write_result(
        stage_result_dir,
        create_result(
            run_id=run_id,
            result_type="audit_run",
            status="success",
            summary={"stage": "fixture"},
            artifacts={},
        ),
    )


def test_build_matrix_deduplicates_cache_rows_and_wires_dependencies(tmp_path):
    module = _load_module("build_experiment_matrix_test", "scripts/build_experiment_matrix.py")
    spec = {
        "name": "wave1",
        "dataset": "tiny_imagenet",
        "base_path": str(tmp_path / "data"),
        "matrix_root": str(tmp_path / "matrix"),
        "resources": {
            "gpu-embed": {"partition": "gpu", "gres": "gpu:1"},
            "cpu-light": {"partition": "cpu"},
            "gpu-fl": {"partition": "gpu", "gres": "gpu:1"},
        },
        "embedding_defaults": {"batch_size": 32, "tokens": "cls", "clustering": "hdbscan", "hdbscan_metric": "euclidean"},
        "embedding_runs": [
            {"name": "dino_a", "extractor": "dinov2-small", "min_cluster_size": 10},
            {"name": "dino_b", "extractor": "dinov2-small", "min_cluster_size": 20},
        ],
        "fl_defaults": {"overrides": ["model=resnet18", "fed.rounds=2"]},
        "fl_runs": [{"name": "resnet18_short"}],
    }

    rows = module.build_rows(spec)

    stage_counts = {}
    for row in rows:
        stage_counts[row["stage"]] = stage_counts.get(row["stage"], 0) + 1

    assert stage_counts == {
        "cache_embeddings": 1,
        "generate_partition": 2,
        "preflight_fl": 2,
        "fl_train": 2,
    }
    partition_rows = [row for row in rows if row["stage"] == "generate_partition"]
    assert all(len(row["depends_on"]) == 1 for row in partition_rows)
    assert all(row["payload"]["output_partition_path"].endswith(".json") for row in partition_rows)


def test_run_matrix_task_builds_fl_command_and_writes_stage_result(tmp_path, monkeypatch):
    module = _load_module("run_matrix_task_test", "scripts/run_matrix_task.py")
    partition_path = tmp_path / "partition.json"
    partition_path.write_text(
        json.dumps({"clients": {"client_0": {"train": []}, "client_1": {"train": []}}}),
        encoding="utf-8",
    )

    dep_dir = tmp_path / "stage_results" / "preflight_dep"
    _write_success_stage_result(dep_dir, "preflight_dep")

    row = {
        "row_id": "train_row",
        "matrix_name": "wave1",
        "stage": "fl_train",
        "stage_rank": 3,
        "resource_class": "gpu-fl",
        "slurm": {"partition": "gpu"},
        "depends_on": ["preflight_dep"],
        "stage_result_dir": str(tmp_path / "stage_results" / "train_row"),
        "payload": {
            "dataset": "tiny_imagenet",
            "base_path": str(tmp_path / "data_root"),
            "partition_path": str(partition_path),
            "config_dir": str(REPO_ROOT / "fl_training" / "conf"),
            "config_name": "config",
            "overrides": ["model=resnet18", "fed.rounds=2"],
            "env": {"HF_HUB_OFFLINE": "1"},
            "run_name": "tiny_run",
            "output_dir": str(tmp_path / "fl_outputs"),
        },
    }
    dep_row = {
        "row_id": "preflight_dep",
        "stage_result_dir": str(dep_dir),
    }

    captured = {}

    class _Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    def _fake_run(cmd, cwd=None, env=None, check=False, **kwargs):
        if cmd[:2] == ["git", "rev-parse"] or cmd[:2] == ["git", "status"]:
            return _Result(returncode=1, stdout="")
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return _Result()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    rc = module.run_row(row, [dep_row, row], dry_run=False)

    assert rc == 0
    assert any(part == "fed.n_clients=2" for part in captured["cmd"])
    assert any(part == "name=tiny_run" for part in captured["cmd"])
    assert captured["env"]["DATASETS_ROOT"] == str(tmp_path / "data_root")
    assert captured["env"]["tiny_imagenet_chosen_split_path"] == str(partition_path)
    result_path = Path(row["stage_result_dir"]) / "result.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"


def test_submit_slurm_matrix_builds_afterany_stage_dependencies(tmp_path):
    module = _load_module("submit_slurm_matrix_test", "scripts/submit_slurm_matrix.py")
    rows = [
        {"row_id": "cache_1", "stage": "cache_embeddings", "resource_class": "gpu-embed", "slurm": {"partition": "gpu"}},
        {"row_id": "partition_1", "stage": "generate_partition", "resource_class": "gpu-embed", "slurm": {"partition": "gpu"}},
    ]

    plans = module.build_submission_plan(rows)
    first = module.build_sbatch_command(
        matrix_path=tmp_path / "matrix.jsonl",
        plan=plans[0],
        dependency_job_ids=[],
        dependency_mode="afterany",
    )
    second = module.build_sbatch_command(
        matrix_path=tmp_path / "matrix.jsonl",
        plan=plans[1],
        dependency_job_ids=["12345"],
        dependency_mode="afterany",
    )

    assert "--array=0-0" in first
    assert "--dependency=afterany:12345" in second


def test_preflight_hpc_validates_env_and_probes_unique_extractors(tmp_path, monkeypatch):
    module = _load_module("preflight_hpc_test", "scripts/preflight_hpc.py")
    matrix_path = tmp_path / "matrix.jsonl"
    rows = [
        {
            "row_id": "cache_a",
            "stage": "cache_embeddings",
            "resource_class": "gpu-embed",
            "stage_result_dir": str(tmp_path / "stage_results" / "cache_a"),
            "payload": {
                "base_path": str(tmp_path / "dataset"),
                "cache_dir": str(tmp_path / "cache"),
                "model": "dinov2-small",
            },
        },
        {
            "row_id": "partition_a",
            "stage": "generate_partition",
            "resource_class": "gpu-embed",
            "stage_result_dir": str(tmp_path / "stage_results" / "partition_a"),
            "payload": {
                "base_path": str(tmp_path / "dataset"),
                "output_partition_path": str(tmp_path / "partitions" / "a.json"),
                "extractor": "dinov2-small",
            },
        },
    ]
    (tmp_path / "dataset").mkdir(parents=True)
    with open(matrix_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    for name in ("FDS_CACHE_DIR", "HF_HOME", "TORCH_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    seen = []
    monkeypatch.setattr(module.ModelRegistry, "create", lambda extractor, **kwargs: seen.append((extractor, kwargs)))

    summary = module.run_checks(
        matrix_path=matrix_path,
        probe_extractors=True,
        device="cpu",
        require_env=[],
        skip_offline_env_check=False,
    )

    assert summary["rows"] == 2
    assert seen == [("dinov2-small", {"device": "cpu", "proxy": ""})]
    assert (tmp_path / "cache").exists()


def test_preflight_hpc_rejects_missing_offline_env(tmp_path, monkeypatch):
    module = _load_module("preflight_hpc_env_fail_test", "scripts/preflight_hpc.py")
    matrix_path = tmp_path / "matrix.jsonl"
    matrix_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required HPC environment variables"):
        module.run_checks(
            matrix_path=matrix_path,
            probe_extractors=False,
            device="cpu",
            require_env=[],
            skip_offline_env_check=False,
        )
