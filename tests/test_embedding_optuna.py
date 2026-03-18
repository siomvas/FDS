from __future__ import annotations

import json
import sys

import pytest

from partitioning.tuning.embedding_optuna import _parse_stage_configs, build_arg_parser, run_cli


class _DummyDataset:
    def __init__(self, length: int):
        self._length = length
        self.dataset_name = "tiny_imagenet"
        self.requested_dataset_name = "tiny_imagenet"

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return {"image": f"/tmp/img_{idx}.jpg", "label": idx % 2}


def test_parse_stage_configs_requires_matching_lengths():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--stage_train_fractions",
            "0.1",
            "1.0",
            "--stage_metric_subsamples",
            "64",
        ]
    )

    with pytest.raises(ValueError, match="same length"):
        _parse_stage_configs(args)


def test_run_cli_dry_run_writes_stage_manifest(tmp_path, monkeypatch):
    from partitioning.tuning import embedding_optuna

    monkeypatch.setattr(embedding_optuna, "BaseDataset", lambda dataset, base_path, split: _DummyDataset(40))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_embedding_partitions.py",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            str(tmp_path),
            "--study_dir",
            str(tmp_path / "study"),
            "--target_n_clients",
            "5",
            "--dry-run",
        ],
    )

    rc = run_cli()
    assert rc == 0

    manifest_path = tmp_path / "study" / "tiny_imagenet_fixed_embedding_optuna" / "stage_manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "tiny_imagenet"
    assert payload["train_total"] == 40
    assert [stage["train_selected"] for stage in payload["stages"]] == [4, 14, 40]


def test_run_cli_fixed_mode_requires_target_n_clients(tmp_path, monkeypatch):
    from partitioning.tuning import embedding_optuna

    monkeypatch.setattr(embedding_optuna, "BaseDataset", lambda dataset, base_path, split: _DummyDataset(10))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_embedding_partitions.py",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            str(tmp_path),
            "--study_dir",
            str(tmp_path / "study"),
            "--dry-run",
        ],
    )

    with pytest.raises(ValueError, match="target_n_clients"):
        run_cli()
