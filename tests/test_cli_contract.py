"""CLI contract tests for shared canonical flags across script entrypoints."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from utils.cli_common import add_standard_cli_args

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_common_output_alias_maps_to_output():
    parser = argparse.ArgumentParser()
    add_standard_cli_args(parser, include_output=True, output_required=True)

    args = parser.parse_args(["--output_dir", "result.json"])
    assert args.output == "result.json"


def test_generate_baselines_parse_args_accepts_output_alias(monkeypatch):
    module = _load_module("generate_baselines_for_test", "scripts/generate_baselines.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_baselines.py",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/fake-dataset-root",
            "--output_dir",
            "/tmp/out",
        ],
    )
    args = module.parse_args()
    assert args.output == "/tmp/out"
    assert args.dataset == "tiny_imagenet"


def test_generate_embedding_partitions_parse_args_accepts_output_alias(monkeypatch):
    module = _load_module(
        "generate_embedding_partitions_for_test",
        "scripts/generate_embedding_partitions.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_embedding_partitions.py",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/fake-dataset-root",
            "--extractor",
            "dinov2-small",
            "--clustering",
            "hdbscan",
            "--hdbscan_metric",
            "euclidean",
            "--output_dir",
            "/tmp/out.json",
        ],
    )
    args = module.parse_args()
    assert args.output == "/tmp/out.json"
    assert args.dataset == "tiny_imagenet"


def test_generate_embedding_partitions_dry_run_allows_missing_base_path(monkeypatch):
    module = _load_module(
        "generate_embedding_partitions_dry_run_for_test",
        "scripts/generate_embedding_partitions.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_embedding_partitions.py",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/does-not-exist-base-path",
            "--extractor",
            "dinov2-small",
            "--clustering",
            "hdbscan",
            "--hdbscan_metric",
            "euclidean",
            "--output",
            "/tmp/out.json",
            "--dry-run",
        ],
    )
    assert module.main() == 0


def test_generate_embedding_partitions_rejects_inconsistent_lastk_pooling(monkeypatch):
    module = _load_module(
        "generate_embedding_partitions_lastk_for_test",
        "scripts/generate_embedding_partitions.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_embedding_partitions.py",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/fake-dataset-root",
            "--extractor",
            "dinov2-small",
            "--clustering",
            "hdbscan",
            "--hdbscan_metric",
            "euclidean",
            "--tokens",
            "k_last_cls",
            "--lastk_k",
            "4",
            "--layer_aggregation",
            "mean",
            "--pooling",
            "mean",
            "--dry-run",
        ],
    )
    try:
        module.main()
    except ValueError as exc:
        assert "--pooling must be 'lastk'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for inconsistent last-k pooling configuration.")


def test_generate_report_dry_run_allows_missing_input(monkeypatch):
    module = _load_module("generate_report_for_test", "scripts/generate_report.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_report.py",
            "--input",
            "/tmp/does-not-exist.csv",
            "--output_dir",
            "/tmp/report.html",
            "--dry-run",
        ],
    )
    assert module.main() == 0


def test_evaluate_partitions_dry_run_allows_missing_base_path(monkeypatch):
    from evaluation import evaluate_partitions

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_partitions.py",
            "--partition",
            "/tmp/does-not-exist-partition.json",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            "/tmp/does-not-exist-base-path",
            "--output_dir",
            "/tmp/results",
            "--dry-run",
        ],
    )
    assert evaluate_partitions.main() == 0


class _StringLabelDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_evaluate_partitions_label_mapping_not_persisted_without_dir(tmp_path, monkeypatch):
    from evaluation import evaluate_partitions

    monkeypatch.chdir(tmp_path)
    dataset = _StringLabelDataset(
        [
            ("img_a.jpg", "dog"),
            ("img_b.jpg", "cat"),
            ("img_c.jpg", "dog"),
        ]
    )

    mapping = evaluate_partitions.get_or_create_label_mapping(
        "tiny_imagenet",
        dataset,
        mapping_dir=None,
    )

    assert mapping == {"cat": 0, "dog": 1}
    assert not (tmp_path / "evaluation" / "tiny_imagenet_label_mapping.json").exists()


def test_evaluate_partitions_label_mapping_persists_in_explicit_dir(tmp_path):
    from evaluation import evaluate_partitions

    dataset = _StringLabelDataset(
        [
            ("img_a.jpg", "dog"),
            ("img_b.jpg", "cat"),
            ("img_c.jpg", "dog"),
        ]
    )
    mapping_dir = tmp_path / "cache" / "label_mappings"
    mapping_file = mapping_dir / "tiny_imagenet_label_mapping.json"

    mapping = evaluate_partitions.get_or_create_label_mapping(
        "tiny_imagenet",
        dataset,
        mapping_dir=mapping_dir,
    )
    assert mapping == {"cat": 0, "dog": 1}
    assert mapping_file.exists()

    mapping_file.write_text('{"dog": 9, "cat": 3}')
    reloaded = evaluate_partitions.get_or_create_label_mapping(
        "tiny_imagenet",
        dataset,
        mapping_dir=mapping_dir,
    )
    assert reloaded == {"dog": 9, "cat": 3}
