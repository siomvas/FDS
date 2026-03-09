from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_results_normalizes_legacy_columns_and_methods(tmp_path):
    module = _load_module("generate_report_schema_test", "scripts/generate_report.py")
    input_csv = tmp_path / "legacy_results.csv"
    pd.DataFrame(
        {
            "method": ["random", "vlm"],
            "n_clients": [3, 3],
            "heterogeneity": [0.1, 0.1],
            "vendi_score_features": [28.1, 29.4],
            "coefficient_of_variation": [0.01, 0.02],
            "samples_per_client_min": [10, 9],
            "samples_per_client_max": [10, 11],
        }
    ).to_csv(input_csv, index=False)

    df = module.load_results(input_csv)

    assert "vendi_score_mean" in df.columns
    assert "balance_cv" in df.columns
    assert df["method"].tolist() == ["Random", "VLM"]


def test_generate_report_handles_legacy_summary_metrics(tmp_path):
    module = _load_module("generate_report_legacy_test", "scripts/generate_report.py")
    df = pd.DataFrame(
        {
            "method": ["random", "vlm"],
            "n_clients": [3, 3],
            "heterogeneity": [0.1, 0.1],
            "vendi_score_features": [28.1, 29.4],
            "coefficient_of_variation": [0.01, 0.02],
            "samples_per_client_min": [10, 9],
            "samples_per_client_max": [10, 11],
        }
    )

    output_path = tmp_path / "report.html"
    module.generate_report(
        df=module.normalize_results_schema(df),
        output_path=output_path,
        dataset_name="legacy_fixture",
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Vendi Score" in html
    assert "VLM" in html
    assert "No summary metrics available." not in html
    assert '<script src="https://cdn.plot.ly/' not in html
    assert "resizeVisiblePlots" in html


def test_generate_report_falls_back_to_appendix_metrics_when_primary_missing(tmp_path):
    module = _load_module("generate_report_appendix_fallback_test", "scripts/generate_report.py")
    df = pd.DataFrame(
        {
            "method": ["random", "dirichlet"],
            "n_clients": [3, 3],
            "heterogeneity": [0.1, 0.1],
            "coefficient_of_variation": [0.01, 0.05],
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
