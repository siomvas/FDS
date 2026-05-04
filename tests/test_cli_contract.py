"""CLI contract tests for the anonymous reviewer entrypoints."""

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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_common_output_alias_maps_to_output():
    parser = argparse.ArgumentParser()
    add_standard_cli_args(parser, include_output=True, output_required=True)

    args = parser.parse_args(["--output_dir", "result.json"])
    assert args.output == "result.json"


def test_download_datasets_all_dry_run_allows_missing_ade20k_archive(tmp_path):
    module = _load_module("download_datasets_contract_test", "scripts/download_datasets.py")

    status = module.main(["--dataset", "all", "--output-root", str(tmp_path), "--dry-run"])

    assert status == 0


def test_download_datasets_explicit_ade20k_requires_archive(tmp_path):
    module = _load_module("download_datasets_ade_contract_test", "scripts/download_datasets.py")

    status = module.main(["--dataset", "ade20k", "--output-root", str(tmp_path)])

    assert status == 2


def test_reproduce_subcommands_dry_run_without_datasets(tmp_path):
    module = _load_module("reproduce_contract_test", "scripts/reproduce.py")

    assert module.main(["partitions", "--datasets", "tiny_imagenet", "--output-root", str(tmp_path), "--dry-run"]) == 0
    assert module.main(["metrics", "--datasets", "tiny_imagenet", "--output-root", str(tmp_path), "--dry-run"]) == 0
    assert module.main(["fl", "--datasets", "tiny_imagenet", "--split-root", str(tmp_path / "splits"), "--dry-run"]) == 0


def test_audit_detects_private_absolute_path(tmp_path):
    module = _load_module("audit_submission_contract_test", "scripts/audit_submission.py")
    sample = tmp_path / "sample.md"
    private_path = "/" + "Users" + "/privateuser/project/data"
    sample.write_text(f"dataset root: {private_path}\n", encoding="utf-8")

    findings = module.scan_paths([sample], root=tmp_path)

    assert any("absolute private user path" in finding.reason for finding in findings)


def test_audit_allows_placeholder_absolute_path(tmp_path):
    module = _load_module("audit_submission_placeholder_test", "scripts/audit_submission.py")
    sample = tmp_path / "sample.md"
    placeholder_path = "/" + "Users" + "/example/project/data"
    sample.write_text(f"dataset root: {placeholder_path}\n", encoding="utf-8")

    findings = module.scan_paths([sample], root=tmp_path)

    assert findings == []
