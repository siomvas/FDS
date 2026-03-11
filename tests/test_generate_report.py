from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from research import create_manifest, create_result, write_manifest, write_result, write_results_index


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_partition_result(
    root: Path,
    run_id: str,
    *,
    status: str = "success",
    method: str = "Random",
    fid_mean: float = 1.0,
) -> None:
    run_dir = root / run_id
    write_manifest(
        run_dir,
        create_manifest(
            run_id=run_id,
            script_name="tests.fixture",
            command=["pytest"],
            resolved_config={"fixture": True},
            dataset={"name": "tiny_imagenet", "base_path": "/tmp/data", "split": "train"},
            partition={"path": "/tmp/partition.json", "filename": "partition.json", "sha256": None},
            model={"name": "dinov2-small", "task_type": "embedding_feature_evaluation"},
            seed=7,
            artifacts={},
        ),
    )
    write_result(
        run_dir,
        create_result(
            run_id=run_id,
            result_type="partition_eval",
            status=status,
            summary={
                "method": method,
                "n_clients": 3,
                "heterogeneity": 0.1,
                "samples_per_client_min": 9,
                "samples_per_client_max": 11,
                "balance_cv": 0.02,
            },
            metrics={
                "fid_mean": fid_mean,
                "mmd_features_mean": 0.1,
                "wasserstein_mean": 0.2,
                "vendi_score_mean": 10.0,
            },
            failure=None
            if status == "success"
            else {
                "error_type": "RuntimeError",
                "error_message": "fixture failure",
                "stage": "test",
            },
        ),
    )


def test_load_results_reads_canonical_directory_and_filters_failures(tmp_path):
    module = _load_module("generate_report_schema_test", "scripts/generate_report.py")
    _write_partition_result(tmp_path, "success_run", status="success", method="random")
    _write_partition_result(tmp_path, "failed_run", status="failed", method="vlm")

    df = module.load_results(tmp_path)

    assert df["method"].tolist() == ["Random"]
    assert df["status"].tolist() == ["success"]
    assert "fid_mean" in df.columns


def test_load_results_accepts_results_index_json(tmp_path):
    module = _load_module("generate_report_index_test", "scripts/generate_report.py")
    _write_partition_result(tmp_path, "indexed_run", method="vlm")
    write_results_index(tmp_path, result_type="partition_eval")

    df = module.load_results(tmp_path / "results_index.json")

    assert df["method"].tolist() == ["VLM"]


def test_load_results_rejects_malformed_canonical_artifact(tmp_path):
    module = _load_module("generate_report_malformed_test", "scripts/generate_report.py")
    bad_dir = tmp_path / "bad_run"
    bad_dir.mkdir(parents=True)
    with open(bad_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": "bad"}, f)

    with pytest.raises(FileNotFoundError, match="run_manifest.json"):
        module.load_results(tmp_path)


def test_generate_report_falls_back_to_appendix_metrics_when_primary_missing(tmp_path):
    module = _load_module("generate_report_appendix_fallback_test", "scripts/generate_report.py")
    df = pd.DataFrame(
        {
            "method": ["random", "dirichlet"],
            "n_clients": [3, 3],
            "heterogeneity": [0.1, 0.1],
            "balance_cv": [0.01, 0.05],
            "samples_per_client_min": [10, 8],
            "samples_per_client_max": [10, 12],
        }
    )

    output_path = tmp_path / "appendix_report.html"
    module.generate_report(
        df=module.normalize_results_schema(df),
        output_path=output_path,
        dataset_name="appendix_fixture",
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Balance (CV)" in html
    assert "No comparable summary metrics available for ranking." not in html
    assert "resizeVisiblePlots" in html


def test_generate_report_scopes_summary_and_raw_table_per_config(tmp_path):
    module = _load_module("generate_report_config_scope_test", "scripts/generate_report.py")
    df = pd.DataFrame(
        {
            "method": ["Random", "VLM", "Random", "VLM"],
            "n_clients": [3, 3, 6, 6],
            "heterogeneity": [0.1, 0.1, 0.1, 0.1],
            "fid_mean": [1.0, 2.0, 3.0, 4.0],
            "mmd_features_mean": [0.1, 0.2, 0.3, 0.4],
            "wasserstein_mean": [0.5, 0.6, 0.7, 0.8],
            "vendi_score_mean": [10.0, 11.0, 12.0, 13.0],
            "balance_cv": [0.01, 0.03, 0.02, 0.04],
            "samples_per_client_min": [9, 8, 7, 6],
            "samples_per_client_max": [11, 12, 13, 14],
        }
    )

    output_path = tmp_path / "config_scoped_report.html"
    module.generate_report(
        df=df,
        output_path=output_path,
        dataset_name="config_scope_fixture",
    )

    html = output_path.read_text(encoding="utf-8")
    assert html.count('<h3>Method Rankings</h3>') == 3
    assert html.count('ranking-table sortable') == 3


def test_generate_summary_findings_handles_empty_metric_list():
    module = _load_module("generate_report_findings_test", "scripts/generate_report.py")
    df = pd.DataFrame({"method": ["Random"]})

    assert module.generate_summary_findings(df, []) == "<p>No summary metrics available.</p>"


def test_get_method_color_is_case_insensitive():
    module = _load_module("generate_report_colors_test", "scripts/generate_report.py")

    assert module.get_method_color("Vlm") == module.METHOD_COLORS["VLM"]
