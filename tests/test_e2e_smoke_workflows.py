"""End-to-end smoke workflows for partitioning and FL training wiring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from data_loading.data_modules import get_datamodule
from fl_training.engines.classification import ClassificationEngine
from fl_training.engines.detection import DetectionEngine
from fl_training.engines.segmentation import SegmentationEngine
from partitioning import DirichletSplitter, EmbeddingSplitter, RandomSplitter


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class _DictSegModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x):
        return {"out": self.head(x)}


class _DummyDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if targets is not None:
            loss = torch.stack([img.float().mean() for img in images]).abs().mean()
            return {"loss_total": (self.scale * 0.0) + loss}

        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]], device=images[0].device),
                "scores": torch.tensor([0.95], device=images[0].device),
                "labels": torch.tensor([1], device=images[0].device),
            }
            for _ in images
        ]


def _write_rgb(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return str(path)


def _write_mask(path: Path, cls_id: int, size: tuple[int, int] = (32, 32)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=cls_id).save(path)
    return str(path)


def _write_ann(path: Path, rows: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")
    return str(path)


def _write_split_file(path: Path, payload: dict) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return str(path)


def _classification_config() -> dict:
    return {
        "training": {
            "steps_or_epochs": "steps",
            "loss": "cross_entropy",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 2,
            "lr": 1e-3,
            "lr_min": 0.0,
            "scheduler": "none",
        },
        "optimizer": {"class_name": "Adam", "args": {}},
        "fed": {"rounds": 1},
    }


def _segmentation_config(num_classes: int) -> dict:
    return {
        "training": {
            "steps_or_epochs": "steps",
            "num_classes": num_classes,
            "ignore_index": 255,
            "loss": "cross_entropy",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 2,
            "lr": 1e-3,
            "lr_min": 0.0,
            "scheduler": "none",
        },
        "optimizer": {"class_name": "Adam", "args": {}},
        "fed": {"rounds": 1},
    }


def _detection_config() -> dict:
    return {
        "training": {
            "steps_or_epochs": "steps",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 2,
            "lr": 1e-3,
            "lr_min": 0.0,
            "scheduler": "none",
            "detection_score_threshold": 0.5,
        },
        "optimizer": {"class_name": "Adam", "args": {}},
        "fed": {"rounds": 1},
    }


def test_partitioning_random_smoke(mock_train_dataset):
    splitter = RandomSplitter(
        train_dataset=mock_train_dataset,
        val_dataset=None,
        n_clients=3,
        heterogeneity=0.0,
        valset_type="none",
        seed=42,
    )

    splits = splitter.get_client_splits()

    assert len(splits["clients"]) == 3
    assert splits["global_val"] is None
    assert splitter.validate_splits(splits) is True


def test_partitioning_dirichlet_smoke(mock_train_dataset):
    splitter = DirichletSplitter(
        train_dataset=mock_train_dataset,
        val_dataset=None,
        n_clients=3,
        heterogeneity=0.5,
        valset_type="none",
        seed=42,
    )

    splits = splitter.get_client_splits()

    assert len(splits["clients"]) == 3
    assert splitter.validate_splits(splits) is True


def test_partitioning_embedding_smoke(mock_train_dataset, mock_val_dataset, tmp_path):
    splitter = EmbeddingSplitter(
        train_dataset=mock_train_dataset,
        val_dataset=mock_val_dataset,
        extractor_name="dinov2-base",
        clustering_method="kmeans",
        clustering_params={"n_clusters": 3, "random_state": 42},
        token_extraction_strategy="cls",
        token_params={},
        pooling_strategy="mean",
        pooling_params={},
        preprocessing_params={},
        valset_type="global",
        seed=42,
        cache_dir=str(tmp_path / "cache"),
        dataset_name="tiny_imagenet",
        base_path="/tmp",
        device="cpu",
        batch_size=4,
        use_cache=False,
        num_workers=0,
    )

    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    train_embeddings = np.random.randn(len(mock_train_dataset), 8)

    with patch.object(splitter, "_get_embeddings", return_value=train_embeddings):
        with patch.object(splitter, "_cluster", return_value=labels):
            splits = splitter.get_client_splits()

    assert len(splits["clients"]) == 3
    assert splits["global_val"] is not None
    assert splitter.validate_splits(splits) is True


def test_fl_classification_one_round_smoke(tmp_path):
    train_samples = []
    val_samples = []

    for i in range(4):
        image_path = _write_rgb(tmp_path / "classification" / f"train_{i}.jpg", color=(20 * i, 40, 60))
        train_samples.append({"image": image_path, "label": i % 3})

    for i in range(2):
        image_path = _write_rgb(tmp_path / "classification" / f"val_{i}.jpg", color=(80, 10 * i, 30))
        val_samples.append({"image": image_path, "label": i % 3})

    split_file = _write_split_file(
        tmp_path / "classification_splits.json",
        {
            "clients": {
                "client_0": {"train": train_samples, "val": []},
                "client_1": {"train": train_samples[:1], "val": []},
            },
            "global_val": val_samples,
        },
    )

    dm = get_datamodule(
        dataset_name="tiny_imagenet",
        data_dir=str(tmp_path),
        split_file=split_file,
        client_idx=0,
        batch_size=2,
        num_workers=0,
        seed=42,
    )
    dm.setup()

    engine = ClassificationEngine(
        model=_TinyClassifier(num_classes=3),
        config=_classification_config(),
        device=torch.device("cpu"),
    )

    train_metrics = engine.train_one_round(dm.train_dataloader)
    val_metrics = engine.evaluate(dm.val_dataloader)

    assert train_metrics["num_steps_current_round"] == 2.0
    assert "train_loss" in train_metrics
    assert "val_loss" in val_metrics


def test_fl_segmentation_one_round_smoke(tmp_path):
    train_samples = []
    val_samples = []

    for i in range(4):
        image_path = _write_rgb(tmp_path / "segmentation" / f"train_{i}.jpg", color=(10 * i, 30, 90), size=(32, 32))
        mask_path = _write_mask(tmp_path / "segmentation" / f"train_{i}.png", cls_id=i % 3, size=(32, 32))
        train_samples.append({"image": image_path, "mask_path": mask_path})

    for i in range(2):
        image_path = _write_rgb(tmp_path / "segmentation" / f"val_{i}.jpg", color=(60, 20 * i, 40), size=(32, 32))
        mask_path = _write_mask(tmp_path / "segmentation" / f"val_{i}.png", cls_id=i % 3, size=(32, 32))
        val_samples.append({"image": image_path, "mask_path": mask_path})

    split_file = _write_split_file(
        tmp_path / "segmentation_splits.json",
        {
            "clients": {
                "client_0": {"train": train_samples, "val": []},
                "client_1": {"train": train_samples[:1], "val": []},
            },
            "global_val": val_samples,
        },
    )

    dm = get_datamodule(
        dataset_name="ade20k",
        data_dir=str(tmp_path),
        split_file=split_file,
        client_idx=0,
        image_size=32,
        batch_size=2,
        num_workers=0,
        seed=42,
    )
    dm.setup()

    engine = SegmentationEngine(
        model=_DictSegModel(num_classes=3),
        config=_segmentation_config(num_classes=3),
        device=torch.device("cpu"),
    )

    train_metrics = engine.train_one_round(dm.train_dataloader)
    val_metrics = engine.evaluate(dm.val_dataloader)

    assert train_metrics["num_steps_current_round"] == 2.0
    assert "train_loss" in train_metrics
    assert "val_loss" in val_metrics


def test_fl_detection_one_round_smoke_pascal_voc(tmp_path):
    train_samples = []
    val_samples = []

    for i in range(4):
        image_path = _write_rgb(tmp_path / "detection" / f"train_{i}.jpg", color=(40, 20 * i, 20), size=(64, 64))
        ann_path = _write_ann(tmp_path / "detection" / f"train_{i}.txt", rows=["0 0.5 0.5 0.4 0.4"])
        train_samples.append({"image": image_path, "annotation_path": ann_path})

    for i in range(2):
        image_path = _write_rgb(tmp_path / "detection" / f"val_{i}.jpg", color=(25, 80, 30 * i), size=(64, 64))
        ann_path = _write_ann(tmp_path / "detection" / f"val_{i}.txt", rows=["person 0.5 0.5 0.3 0.3"])
        val_samples.append({"image": image_path, "annotation_path": ann_path})

    split_file = _write_split_file(
        tmp_path / "detection_splits.json",
        {
            "clients": {
                "client_0": {"train": train_samples, "val": []},
                "client_1": {"train": train_samples[:1], "val": []},
            },
            "global_val": val_samples,
        },
    )

    dm = get_datamodule(
        dataset_name="pascal_voc",
        data_dir=str(tmp_path),
        split_file=split_file,
        client_idx=0,
        batch_size=2,
        num_workers=0,
        seed=42,
        detection_label_offset=1,
    )
    dm.setup()

    engine = DetectionEngine(
        model=_DummyDetectionModel(),
        config=_detection_config(),
        device=torch.device("cpu"),
    )

    train_metrics = engine.train_one_round(dm.train_dataloader)
    val_metrics = engine.evaluate(dm.val_dataloader)

    assert train_metrics["num_steps_current_round"] == 2.0
    assert "train_loss" in train_metrics
    assert "avg_detections" in val_metrics
