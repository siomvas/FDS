from __future__ import annotations

from pathlib import Path

import pytest

from partitioning.tuning.optuna_proxy import (
    EmbeddingStudyPreset,
    build_arg_parser,
    build_embedding_optuna_argv,
    run_preset_cli,
)


def test_build_embedding_optuna_argv_for_fixed_preset_includes_target_n_clients():
    preset = EmbeddingStudyPreset(
        key="tiny_imagenet_embedding_fixed",
        dataset_name="tiny_imagenet",
        study_mode="fixed",
        default_target_n_clients=10,
    )

    argv = build_embedding_optuna_argv(
        preset,
        base_path=Path("/tmp/data"),
        study_dir=Path("/tmp/study"),
        study_name=None,
        n_trials=25,
        timeout=3600,
        sampler="tpe",
        bruteforce_avoid_premature_stop=False,
        seed=7,
        num_workers=2,
        device="cuda:0",
        cache_dir=Path("/tmp/cache"),
        proxy=None,
        embed_batch_size=64,
        metric_batch_size=32,
        artifact_top_k=5,
        target_n_clients=6,
        min_client_size=20,
        min_discovered_clients=2,
        max_discovered_clients=50,
        no_cache=True,
        dry_run=False,
    )

    assert "--dataset" in argv and "tiny_imagenet" in argv
    assert "--study_mode" in argv and "fixed" in argv
    assert "--target_n_clients" in argv and "6" in argv
    assert "--no_cache" in argv


def test_build_embedding_optuna_argv_for_discovery_preset_omits_target_n_clients():
    preset = EmbeddingStudyPreset(
        key="tiny_imagenet_embedding_discovery",
        dataset_name="tiny_imagenet",
        study_mode="discovery",
    )

    argv = build_embedding_optuna_argv(
        preset,
        base_path=Path("/tmp/data"),
        study_dir=Path("/tmp/study"),
        study_name=None,
        n_trials=25,
        timeout=None,
        sampler="tpe",
        bruteforce_avoid_premature_stop=False,
        seed=7,
        num_workers=2,
        device=None,
        cache_dir=None,
        proxy=None,
        embed_batch_size=64,
        metric_batch_size=32,
        artifact_top_k=5,
        target_n_clients=None,
        min_client_size=20,
        min_discovered_clients=2,
        max_discovered_clients=50,
        no_cache=False,
        dry_run=True,
    )

    assert "--study_mode" in argv and "discovery" in argv
    assert "--target_n_clients" not in argv
    assert "--dry-run" in argv


def test_discovery_preset_defaults_to_bruteforce_sampler():
    preset = EmbeddingStudyPreset(
        key="tiny_imagenet_embedding_discovery",
        dataset_name="tiny_imagenet",
        study_mode="discovery",
    )

    parser = build_arg_parser(preset)
    args = parser.parse_args(
        [
            "--base_path",
            "/tmp/data",
            "--study_dir",
            "/tmp/study",
        ]
    )

    assert args.sampler == "brute_force"


def test_build_embedding_optuna_argv_includes_preset_extra_args():
    preset = EmbeddingStudyPreset(
        key="tiny_imagenet_embedding_fixed",
        dataset_name="tiny_imagenet",
        study_mode="fixed",
        default_target_n_clients=10,
        extra_args={
            "stage_train_fractions": [1.0],
            "stage_metric_subsamples": [0],
        },
    )

    argv = build_embedding_optuna_argv(
        preset,
        base_path=Path("/tmp/data"),
        study_dir=Path("/tmp/study"),
        study_name=None,
        n_trials=25,
        timeout=None,
        sampler="tpe",
        bruteforce_avoid_premature_stop=False,
        seed=7,
        num_workers=2,
        device=None,
        cache_dir=None,
        proxy=None,
        embed_batch_size=64,
        metric_batch_size=32,
        artifact_top_k=5,
        target_n_clients=10,
        min_client_size=20,
        min_discovered_clients=2,
        max_discovered_clients=50,
        no_cache=False,
        dry_run=False,
    )

    assert "--stage_train_fractions" in argv
    assert "--stage_metric_subsamples" in argv
    assert argv[argv.index("--stage_train_fractions") + 1] == "1.0"
    assert argv[argv.index("--stage_metric_subsamples") + 1] == "0"


def test_build_embedding_optuna_argv_includes_bruteforce_sampler_flags():
    preset = EmbeddingStudyPreset(
        key="tiny_imagenet_embedding_fixed",
        dataset_name="tiny_imagenet",
        study_mode="fixed",
        default_target_n_clients=10,
    )

    argv = build_embedding_optuna_argv(
        preset,
        base_path=Path("/tmp/data"),
        study_dir=Path("/tmp/study"),
        study_name=None,
        n_trials=25,
        timeout=None,
        sampler="brute_force",
        bruteforce_avoid_premature_stop=True,
        seed=7,
        num_workers=2,
        device=None,
        cache_dir=None,
        proxy=None,
        embed_batch_size=64,
        metric_batch_size=32,
        artifact_top_k=5,
        target_n_clients=10,
        min_client_size=20,
        min_discovered_clients=2,
        max_discovered_clients=50,
        no_cache=False,
        dry_run=False,
    )

    assert "--sampler" in argv
    assert argv[argv.index("--sampler") + 1] == "brute_force"
    assert "--bruteforce_avoid_premature_stop" in argv


def test_run_preset_cli_requires_target_n_clients_for_fixed_mode(tmp_path):
    preset = EmbeddingStudyPreset(
        key="tiny_imagenet_embedding_fixed",
        dataset_name="tiny_imagenet",
        study_mode="fixed",
        default_target_n_clients=None,
    )

    with pytest.raises(ValueError, match="requires target_n_clients"):
        run_preset_cli(
            preset,
            [
                "--base_path",
                str(tmp_path),
                "--study_dir",
                str(tmp_path / "study"),
                "--dry-run",
            ],
        )
