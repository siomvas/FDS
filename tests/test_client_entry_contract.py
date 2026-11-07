"""Functional contract tests for FL client entry wiring."""

from __future__ import annotations

import base64
import json
import sys
from types import SimpleNamespace

import pytest
import torch.nn as nn

from fl_training import client_entry


def test_extract_client_idx_supported_formats():
    assert client_entry.extract_client_idx("site-1") == 0
    assert client_entry.extract_client_idx("site-3") == 2
    assert client_entry.extract_client_idx("client_7") == 7
    assert client_entry.extract_client_idx("worker42") == 42


def test_decode_payload_invalid_base64_raises():
    with pytest.raises(ValueError, match="Invalid config payload"):
        client_entry.decode_payload("not-valid-base64")


def test_validate_client_config_rejects_task_mismatch():
    cfg = {
        "dataset": {
            "class_path": "data_loading.data_modules.TinyImageNetDataModule",
            "args": {"split_file": "/tmp/splits.json", "data_dir": "/tmp/data"},
        },
        "training": {"task_type": "classification", "batch_size": 4, "num_classes": 2},
        "model": {"name": "resnet18", "provider": "torchvision", "args": {"weights": None}},
        "fed": {"rounds": 1},
        "seed": 42,
    }
    with pytest.raises(ValueError, match="Task mismatch"):
        client_entry.validate_client_config(cfg, task_type="detection")


def test_validate_client_config_requires_split_path():
    cfg = {
        "dataset": {
            "class_path": "data_loading.data_modules.TinyImageNetDataModule",
            "args": {"data_dir": "/tmp/data"},
        },
        "training": {"task_type": "classification", "batch_size": 4, "num_classes": 2},
        "model": {"name": "resnet18", "provider": "torchvision", "args": {"weights": None}},
        "fed": {"rounds": 1},
        "seed": 42,
    }
    with pytest.raises(ValueError, match="Missing dataset split path"):
        client_entry.validate_client_config(cfg, task_type="classification")


def test_main_maps_dataset_arguments_to_datamodule(monkeypatch):
    captured = {}

    class DummyDataModule:
        def __init__(self):
            self.train_dataloader = []
            self.val_dataloader = None

        def setup(self):
            return None

    class DummyEngine:
        def __init__(self, model, config, device):
            self.model = model

    def fake_get_data_module(class_path, args):
        captured["class_path"] = class_path
        captured["args"] = args
        return DummyDataModule()

    cfg = {
        "dataset": {
            "class_path": "data_loading.data_modules.TinyImageNetDataModule",
            "args": {"root": "/tmp/data"},
            "dataset_splits": "/tmp/splits.json",
            "label_map": {"cat": 0, "dog": 1},
        },
        "training": {"batch_size": 8, "num_classes": 2},
        "model": {"name": "resnet18", "provider": "torchvision", "args": {"weights": None}},
        "fed": {"rounds": 1},
        "seed": 123,
    }
    payload = base64.b64encode(json.dumps(cfg).encode("utf-8")).decode("utf-8")

    monkeypatch.setattr(client_entry, "get_data_module", fake_get_data_module)
    monkeypatch.setattr(client_entry, "create_model", lambda **_: nn.Linear(1, 1))
    monkeypatch.setattr(client_entry, "ClassificationEngine", DummyEngine)
    monkeypatch.setattr(client_entry, "SummaryWriter", lambda: SimpleNamespace(add_scalar=lambda *a, **k: None))
    monkeypatch.setattr(client_entry.flare, "init", lambda: object())
    monkeypatch.setattr(client_entry.flare, "get_site_name", lambda _: "site-2")
    monkeypatch.setattr(client_entry.flare, "is_running", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["client_entry.py", "--task_type", "classification", "--config_payload", payload],
    )

    client_entry.main()

    assert captured["class_path"] == "data_loading.data_modules.TinyImageNetDataModule"
    assert captured["args"]["client_idx"] == 1
    assert captured["args"]["seed"] == 123
    assert captured["args"]["batch_size"] == 8
    assert captured["args"]["data_dir"] == "/tmp/data"
    assert captured["args"]["split_file"] == "/tmp/splits.json"
    assert captured["args"]["label_map"] == {"cat": 0, "dog": 1}


def test_metric_callback_groups_tags_by_phase_and_filters_internal_keys():
    events = []

    writer = SimpleNamespace(add_scalar=lambda name, value, global_step: events.append((name, value, global_step)))

    client_entry.metric_callback(
        {
            "round": 7,
            "accuracy": 0.75,
            "val_loss": 1.25,
            "num_steps_current_round": 4,
        },
        writer,
        prefix="pre",
        step_override=3,
    )

    assert events == [
        ("pre/accuracy", 0.75, 3),
        ("pre/loss", 1.25, 3),
    ]


def test_metric_callback_normalizes_train_metric_names():
    events = []

    writer = SimpleNamespace(add_scalar=lambda name, value, global_step: events.append((name, value, global_step)))

    client_entry.metric_callback(
        {
            "round": 5,
            "train_loss": 0.5,
            "train_accuracy": 0.8,
            "mean_iou": 0.3,
        },
        writer,
        prefix="train",
    )

    assert events == [
        ("train/loss", 0.5, 5),
        ("train/accuracy", 0.8, 5),
        ("train/mean_iou", 0.3, 5),
    ]


def test_build_datamodule_args_propagates_detection_provider():
    cfg = {
        "dataset": {
            "class_path": "data_loading.data_modules.PascalVOCDataModule",
            "args": {"data_dir": "/tmp/data", "split_file": "/tmp/splits.json"},
        },
        "training": {"batch_size": 2, "num_classes": 21},
        "model": {"name": "facebook/detr-resnet-50", "provider": "hf", "args": {}},
        "fed": {"rounds": 1},
        "seed": 5,
    }

    args = client_entry.build_datamodule_args(cfg, client_name="site-1")

    assert args["client_idx"] == 0
    assert args["batch_size"] == 2
    assert args["detection_provider"] == "hf"


def test_build_datamodule_args_does_not_inject_detection_provider_for_classification():
    cfg = {
        "dataset": {
            "class_path": "data_loading.data_modules.TinyImageNetDataModule",
            "args": {"data_dir": "/tmp/data", "split_file": "/tmp/splits.json"},
        },
        "training": {"task_type": "classification", "batch_size": 8, "num_classes": 200},
        "model": {"name": "resnet18", "provider": "timm", "args": {}},
        "fed": {"rounds": 1},
        "seed": 5,
    }

    args = client_entry.build_datamodule_args(cfg, client_name="site-1")

    assert args["client_idx"] == 0
    assert args["batch_size"] == 8
    assert "detection_provider" not in args
