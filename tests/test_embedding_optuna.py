from __future__ import annotations

import json
import sys

import pytest

from partitioning.tuning.embedding_optuna import (
    DEFAULT_EXTRACTORS,
    _build_clustering_config,
    _build_token_and_pooling_params,
    _extractor_family,
    _parse_stage_configs,
    _resolve_hdbscan_min_cluster_size_choices,
    _resolve_hdbscan_min_samples_choices,
    _study_extractor_choices,
    _study_pooling_choices,
    _study_token_choices,
    build_arg_parser,
    run_cli,
)


class _DummyDataset:
    def __init__(self, length: int):
        self._length = length
        self.dataset_name = "tiny_imagenet"
        self.requested_dataset_name = "tiny_imagenet"

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return {"image": f"/tmp/img_{idx}.jpg", "label": idx % 2}


class _DummyTrial:
    def __init__(self, categorical: dict[str, object] | None = None):
        self._categorical = categorical or {}
        self.calls: list[tuple[str, list[object]]] = []
        self.int_calls: list[tuple[str, int, int, bool]] = []

    def suggest_categorical(self, name, choices):
        choice_list = list(choices)
        self.calls.append((name, choice_list))
        value = self._categorical.get(name, choice_list[0])
        if value not in choice_list:
            raise AssertionError(f"Choice {value!r} is not valid for {name}: {choice_list}")
        return value

    def suggest_int(self, name, low, high, log=False):
        self.int_calls.append((name, low, high, log))
        value = self._categorical.get(name, low)
        if not (low <= value <= high):
            raise AssertionError(f"Value {value!r} is not valid for {name}: [{low}, {high}]")
        return value


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


def test_fixed_mode_default_extractors_cover_all_distinct_model_families():
    families = {_extractor_family(name) for name in DEFAULT_EXTRACTORS}
    assert families == {
        "clip",
        "dinov2",
        "siglip",
        "mae",
        "inception_v3",
        "raw_pixel_resizer",
        "sparse_random_projection_resizer",
        "uae",
        "wavelet_scattering",
    }


def test_discovery_mode_defaults_use_narrowed_extractors_tokens_and_pooling():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--study_mode",
            "discovery",
        ]
    )

    assert _study_extractor_choices(args) == [
        "dinov2-base",
        "clip-base-patch16",
        "siglip-base-patch16-224",
    ]
    assert _study_token_choices(args) == ["cls", "all_last"]
    assert _study_pooling_choices(args) == ["mean", "gem"]


def test_non_transformer_extractors_force_cls_only_search_space():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--target_n_clients",
            "5",
        ]
    )
    trial = _DummyTrial()

    token_choice, token_params, pooling_strategy, pooling_params = _build_token_and_pooling_params(
        trial,
        args,
        extractor_name="wavelet_scattering",
    )

    assert token_choice == "cls"
    assert token_params == {}
    assert pooling_strategy == "cls"
    assert pooling_params == {}
    assert trial.calls == []


def test_transformer_cls_tokens_disable_pooling_search():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--target_n_clients",
            "5",
        ]
    )
    trial = _DummyTrial({"tokens": "cls"})

    token_choice, token_params, pooling_strategy, pooling_params = _build_token_and_pooling_params(
        trial,
        args,
        extractor_name="dinov2-base",
    )

    assert token_choice == "cls"
    assert token_params == {}
    assert pooling_strategy == "cls"
    assert pooling_params == {}
    assert [name for name, _ in trial.calls] == ["tokens"]


def test_transformer_all_last_uses_non_cls_pooling_choices():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--target_n_clients",
            "5",
        ]
    )
    trial = _DummyTrial({"tokens": "all_last", "pooling": "gem", "gem_p": 2.0})

    token_choice, token_params, pooling_strategy, pooling_params = _build_token_and_pooling_params(
        trial,
        args,
        extractor_name="clip-base-patch16",
    )

    assert token_choice == "all_last"
    assert token_params == {}
    assert pooling_strategy == "gem"
    assert pooling_params == {"p": 2.0}
    assert ("pooling", ["mean", "max", "gem"]) in trial.calls


def test_transformer_k_last_cls_forces_cls_pooling():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--target_n_clients",
            "5",
        ]
    )
    trial = _DummyTrial({"tokens": "k_last_cls", "lastk_k": 4})

    token_choice, token_params, pooling_strategy, pooling_params = _build_token_and_pooling_params(
        trial,
        args,
        extractor_name="mae-base",
    )

    assert token_choice == "k_last_cls"
    assert token_params == {"lastk_k": 4, "layer_aggregation": "mean"}
    assert pooling_strategy == "cls"
    assert pooling_params == {}
    assert [name for name, _ in trial.calls] == ["tokens", "lastk_k"]


def test_transformer_k_last_all_uses_non_cls_pooling_choices():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--target_n_clients",
            "5",
        ]
    )
    trial = _DummyTrial({"tokens": "k_last_all", "lastk_k": 2, "pooling": "max"})

    token_choice, token_params, pooling_strategy, pooling_params = _build_token_and_pooling_params(
        trial,
        args,
        extractor_name="siglip-base-patch16-224",
    )

    assert token_choice == "k_last_all"
    assert token_params == {"lastk_k": 2, "layer_aggregation": "mean"}
    assert pooling_strategy == "max"
    assert pooling_params == {}
    assert [name for name, _ in trial.calls] == ["tokens", "lastk_k", "pooling"]


def test_run_cli_rejects_unsupported_study_pooling_choices(tmp_path, monkeypatch):
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
            "--target_n_clients",
            "5",
            "--pooling_choices",
            "lastk",
            "--dry-run",
        ],
    )

    with pytest.raises(ValueError, match="--pooling_choices contains unsupported values"):
        run_cli()


def test_run_cli_rejects_spectral_fixed_clustering_choices(tmp_path, monkeypatch):
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
            "--target_n_clients",
            "5",
            "--fixed_clustering_choices",
            "kmeans,spectral",
            "--dry-run",
        ],
    )

    with pytest.raises(ValueError, match="--fixed_clustering_choices contains unsupported values"):
        run_cli()


def test_run_cli_rejects_knn_fixed_clustering_choices(tmp_path, monkeypatch):
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
            "--target_n_clients",
            "5",
            "--fixed_clustering_choices",
            "kmeans,knn",
            "--dry-run",
        ],
    )

    with pytest.raises(ValueError, match="--fixed_clustering_choices contains unsupported values"):
        run_cli()


def test_hdbscan_metric_is_not_sampled_when_normalized():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--study_mode",
            "discovery",
        ]
    )
    trial = _DummyTrial(
        {
            "clustering": "hdbscan",
            "min_cluster_size": 20,
            "min_samples": 5,
            "hdbscan_method": "eom",
        }
    )

    clustering_method, params = _build_clustering_config(
        trial,
        args,
        preprocessing_params={"normalization": "l2"},
        dataset_len=200,
    )

    assert clustering_method == "hdbscan"
    assert params["hdbscan_metric"] == "euclidean"
    assert [name for name, _ in trial.calls] == ["clustering", "min_cluster_size", "min_samples", "hdbscan_method"]


def test_default_hdbscan_choices_are_small_and_interpretable():
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
            "--study_mode",
            "discovery",
        ]
    )

    min_cluster_size_choices = _resolve_hdbscan_min_cluster_size_choices(args, dataset_len=500)
    min_samples_choices = _resolve_hdbscan_min_samples_choices(args, min_cluster_size_choices)

    assert min_cluster_size_choices == [20, 40, 80]
    assert min_samples_choices == [1, 5, 10]
