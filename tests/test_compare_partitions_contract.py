from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from research import scan_results


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


class _FakeSplitter:
    def __init__(self, payload):
        self.payload = payload

    def get_client_splits(self):
        return self.payload

    def validate_splits(self, splits):
        assert splits == self.payload

    def store_client_splits(self, output_path: str, format: str = "json"):
        assert format == "json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle)


def test_statistics_helpers_rank_by_metric_direction():
    module = _load_module("evaluation_statistics_test", "evaluation/statistics.py")
    df = pd.DataFrame(
        {
            "method": ["Random", "Dirichlet", "Random", "Dirichlet"],
            "dataset_name": ["tiny_imagenet"] * 4,
            "n_clients": [3, 3, 6, 6],
            "heterogeneity": [0.1, 0.1, 0.1, 0.1],
            "seed": [7, 7, 7, 7],
            "fid_mean": [4.0, 2.0, 5.0, 1.0],
            "JSD_mean": [0.1, 0.4, 0.2, 0.5],
        }
    )

    metrics = module.available_metrics(df)
    rankings = module.rank_methods(df, metrics)
    pairwise = module.pairwise_method_deltas(df, metrics)

    assert metrics == ["JSD_mean", "fid_mean"]
    assert rankings[(rankings["metric"] == "fid_mean") & (rankings["rank"] == 1)]["method"].iloc[0] == "Dirichlet"
    assert rankings[(rankings["metric"] == "JSD_mean") & (rankings["rank"] == 1)]["method"].iloc[0] == "Dirichlet"
    fid_delta = pairwise[pairwise["metric"] == "fid_mean"].iloc[0]
    assert fid_delta["matched_runs"] == 2
    assert fid_delta["better_method"] == "Dirichlet"


def test_run_experiment_writes_canonical_comparison_artifacts(tmp_path, monkeypatch):
    module = _load_module("compare_partitions_contract_test", "evaluation/_compare_partitions.py")
    payload = {
        "clients": {
            "client_0": {"train": [{"image": "img_a.jpg", "label": 0}]},
            "client_1": {"train": [{"image": "img_b.jpg", "label": 1}]},
        }
    }
    monkeypatch.setattr(module, "create_splitter", lambda *args, **kwargs: _FakeSplitter(payload))
    monkeypatch.setattr(
        module,
        "compute_label_metrics",
        lambda client_datasets, num_classes: {
            "JSD_mean": 0.2,
            "JSD_std": 0.0,
            "JSD_global_mean": 0.1,
            "JSD_global_std": 0.0,
            "TV_mean": 0.3,
            "TV_std": 0.0,
        },
    )

    exp_config = module.ExperimentConfig(method="random", n_clients=2, heterogeneity=0.1, seed=7)
    run_id = module.run_experiment(
        exp_config=exp_config,
        dataset_name="tiny_imagenet",
        train_dataset=None,
        val_dataset=None,
        base_path=str(tmp_path),
        num_classes=2,
        metrics_mode="label",
        subsample=0,
        label_mapping=None,
        feature_model="dinov2-small",
        use_cache=False,
        cache_dir=str(tmp_path / "cache"),
        output_dir=tmp_path,
    )

    rows = scan_results(tmp_path, result_type="partition_eval")
    assert [row["run_id"] for row in rows] == [run_id]
    assert rows[0]["status"] == "success"
    assert rows[0]["method"] == "Random"
    split_path = tmp_path / run_id / "split.json"
    assert split_path.exists()


def test_validate_experiment_request_rejects_vlm_without_cluster_labels():
    module = _load_module("compare_partitions_validate_test", "evaluation/_compare_partitions.py")

    with pytest.raises(ValueError, match="--cluster_labels_dir"):
        module.validate_experiment_request(
            methods=["random", "vlm"],
            embedding_presets=["default"],
            cluster_labels_dir=None,
        )
