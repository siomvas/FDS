"""Tests for scripts/preflight_fl.py fail-fast validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rgb(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(128, 64, 32)).save(path)
    return str(path)


def _write_json(path: Path, payload: dict) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return str(path)


def _base_cfg(split_file: str, data_root: str) -> dict:
    return {
        "seed": 42,
        "dataset": {
            "class_path": "data_loading.data_modules.TinyImageNetDataModule",
            "dataset_splits": split_file,
            "args": {"root": data_root},
        },
        "training": {
            "task_type": "classification",
            "batch_size": 2,
            "num_classes": 2,
        },
        "model": {
            "name": "resnet18",
            "provider": "torchvision",
            "args": {"weights": None},
        },
        "fed": {"rounds": 1},
    }


def test_preflight_fl_valid_config_and_split_passes(tmp_path, monkeypatch):
    module = _load_module("preflight_fl_smoke_valid", "scripts/preflight_fl.py")

    img0 = _write_rgb(tmp_path / "img0.jpg")
    img1 = _write_rgb(tmp_path / "img1.jpg")

    split_file = _write_json(
        tmp_path / "splits.json",
        {
            "clients": {
                "client_0": {
                    "train": [{"image": img0, "label": 0}],
                    "val": [],
                }
            },
            "global_val": [{"image": img1, "label": 1}],
        },
    )

    cfg_file = _write_json(tmp_path / "cfg.json", _base_cfg(split_file=split_file, data_root=str(tmp_path)))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_fl.py",
            "--config-file",
            cfg_file,
            "--check-paths",
        ],
    )

    assert module.main() == 0


def test_preflight_fl_rejects_provider_task_mismatch(tmp_path, monkeypatch):
    module = _load_module("preflight_fl_mismatch", "scripts/preflight_fl.py")

    img0 = _write_rgb(tmp_path / "img0.jpg")
    split_file = _write_json(
        tmp_path / "splits.json",
        {
            "clients": {
                "client_0": {
                    "train": [{"image": img0, "label": 0}],
                    "val": [],
                }
            },
            "global_val": None,
        },
    )

    cfg = _base_cfg(split_file=split_file, data_root=str(tmp_path))
    cfg["training"]["task_type"] = "segmentation"
    cfg["model"]["provider"] = "timm"
    cfg_file = _write_json(tmp_path / "cfg_bad_model.json", cfg)

    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight_fl.py", "--config-file", cfg_file],
    )

    assert module.main() == 1


def test_preflight_fl_requires_split_path(tmp_path, monkeypatch):
    module = _load_module("preflight_fl_missing_split", "scripts/preflight_fl.py")

    cfg = _base_cfg(split_file="/tmp/placeholder.json", data_root=str(tmp_path))
    del cfg["dataset"]["dataset_splits"]

    cfg_file = _write_json(tmp_path / "cfg_no_split.json", cfg)

    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight_fl.py", "--config-file", cfg_file],
    )

    assert module.main() == 1


def test_preflight_fl_rejects_path_only_split_schema(tmp_path, monkeypatch):
    module = _load_module("preflight_fl_bad_schema", "scripts/preflight_fl.py")

    split_file = _write_json(
        tmp_path / "splits_bad.json",
        {
            "clients": {
                "client_0": {
                    "train": ["/tmp/img0.jpg"],
                    "val": [],
                }
            },
            "global_val": None,
        },
    )
    cfg_file = _write_json(tmp_path / "cfg_bad_split.json", _base_cfg(split_file=split_file, data_root=str(tmp_path)))

    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight_fl.py", "--config-file", cfg_file],
    )

    assert module.main() == 1
