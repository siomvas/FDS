from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from evaluation._natural_partition_compare import NaturalComparisonConfig, build_natural_splits, run_comparison


class _DummyEmbeddingSplitter:
    def __init__(self, *args, **kwargs):
        self._splits = {
            "clients": {
                "client_0": {"train": [{"image": kwargs["train_dataset"].data[0], "label": kwargs["train_dataset"].labels[0]}], "val": []},
                "client_1": {"train": [{"image": kwargs["train_dataset"].data[1], "label": kwargs["train_dataset"].labels[1]}], "val": []},
            },
            "global_val": None,
        }

    def get_client_splits(self):
        return self._splits

    def store_client_splits(self, filename: str, format: str = "json"):
        Path(filename).write_text(json.dumps(self._splits), encoding="utf-8")


def test_build_natural_splits_uses_dataset_client_metadata(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for idx in range(2):
        Image.new("RGB", (8, 8), color=(idx * 40, 0, 0)).save(images_dir / f"img_{idx}.jpg")
    (tmp_path / "metadata.csv").write_text(
        "\n".join(
            [
                "image,label,center,split",
                "images/img_0.jpg,0,site_a,train",
                "images/img_1.jpg,1,site_b,train",
            ]
        ),
        encoding="utf-8",
    )

    from data_loading.dataset import BaseDataset

    dataset = BaseDataset("fed_isic2019", str(tmp_path), "train")
    splits = build_natural_splits(dataset)
    assert set(splits["clients"]) == {"site_a", "site_b"}
    assert len(splits["clients"]["site_a"]["train"]) == 1


def test_run_comparison_writes_expected_artifacts(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for idx in range(2):
        Image.new("RGB", (8, 8), color=(idx * 40, 0, 0)).save(images_dir / f"img_{idx}.jpg")
    (tmp_path / "metadata.csv").write_text(
        "\n".join(
            [
                "image,label,center,split",
                "images/img_0.jpg,0,site_a,train",
                "images/img_1.jpg,1,site_b,train",
            ]
        ),
        encoding="utf-8",
    )

    import evaluation._natural_partition_compare as module

    monkeypatch.setattr(module, "EmbeddingSplitter", _DummyEmbeddingSplitter)
    monkeypatch.setattr(module, "get_preset_config", lambda preset, n_clients: object())
    monkeypatch.setattr(
        module,
        "compute_feature_metrics",
        lambda *args, **kwargs: {"mmd_features_mean": 0.5, "fid_mean": 1.0, "centroid_dist_mean": 0.2, "feature_variance_mean": 0.1},
    )
    monkeypatch.setattr(
        module,
        "generate_report",
        lambda df, output_path, dataset_name: Path(output_path).write_text("<html></html>", encoding="utf-8"),
    )
    monkeypatch.setattr(module, "write_statistics_bundle", lambda output_dir, df: None)

    cfg = NaturalComparisonConfig(
        dataset="fed_isic2019",
        base_path=str(tmp_path),
        embedding_preset="default",
        target_n_clients=None,
        heterogeneity=1.0,
        metrics_mode="feature",
        feature_model="dinov2-small",
        subsample=0,
        use_cache=False,
        cache_dir=str(tmp_path / "cache"),
        seed=7,
    )
    output_dir = tmp_path / "results"
    run_comparison(cfg, output_dir)

    assert (output_dir / "natural_partition.json").exists()
    assert (output_dir / "embedding_partition.json").exists()
    assert (output_dir / "comparison.csv").exists()
    assert (output_dir / "report.html").exists()

    comparison = pd.read_csv(output_dir / "comparison.csv")
    assert "delta_embedding_minus_natural" in comparison.columns
