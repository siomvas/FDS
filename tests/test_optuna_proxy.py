from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from fl_training.tuning.optuna_proxy import build_slice_manifest, run_preset_cli
from fl_training.tuning.presets import PRESETS


def _write_rgb(path: Path, color=(128, 128, 128), size=(64, 64)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return str(path)


def _build_tiny_imagenet_minimal(base_path: Path) -> None:
    root = base_path / "tiny-imagenet-200"
    train_dir = root / "train"
    val_img_dir = root / "val" / "images"

    classes = ["n000", "n001"]
    for class_idx, class_name in enumerate(classes):
        class_img_dir = train_dir / class_name / "images"
        for img_idx in range(10):
            _write_rgb(
                class_img_dir / f"{class_name}_{img_idx}.JPEG",
                color=(40 * class_idx, 30 * img_idx, 100),
            )

    val_img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(2):
        img_name = f"val_{idx}.JPEG"
        _write_rgb(val_img_dir / img_name, color=(20 * idx, 10, 200))
        rows.append(f"{img_name}\t{classes[idx % len(classes)]}\t0\t0\t64\t64")

    with open(root / "val" / "val_annotations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def test_build_slice_manifest_for_tiny_imagenet(tmp_path):
    _build_tiny_imagenet_minimal(tmp_path)
    preset = PRESETS["tiny_imagenet_resnet18"]
    manifest_path = build_slice_manifest(
        preset,
        base_path=tmp_path,
        output_path=tmp_path / "slices" / "tiny_manifest.json",
        seed=7,
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["metadata"]["train_total"] == 20
    assert payload["metadata"]["train_selected"] == 4
    assert len(payload["clients"]["client_0"]["train"]) == 4
    assert len(payload["global_val"]) == 2


def test_run_preset_cli_dry_run_writes_slice_manifest(tmp_path, monkeypatch):
    _build_tiny_imagenet_minimal(tmp_path)
    preset = PRESETS["tiny_imagenet_resnet18"]
    study_dir = tmp_path / "study"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_tiny_imagenet_resnet18.py",
            "--base_path",
            str(tmp_path),
            "--study_dir",
            str(study_dir),
            "--dry-run",
        ],
    )

    rc = run_preset_cli(preset)
    assert rc == 0
    manifest_path = study_dir / "slices" / "tiny_imagenet_1of5_seed42.json"
    assert manifest_path.exists()
