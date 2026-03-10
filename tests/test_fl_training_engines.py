import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset

from fl_training.engines.segmentation import SegmentationEngine
from fl_training.engines.detection import DetectionEngine
from fl_training.models.factory import create_model


class _DictSegModel(nn.Module):
    """Torchvision-style segmentation output: {'out': logits}."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x):
        return {"out": self.head(x)}


class _DummyDetectionModel(nn.Module):
    """Lightweight detection model compatible with torchvision-style calls."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if isinstance(images, list):
            if targets is not None:
                loss = torch.stack([img.float().mean() for img in images]).abs().mean()
                return {"loss_total": (self.scale * 0.0) + loss}
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=images[0].device),
                    "scores": torch.tensor([0.95], device=images[0].device),
                    "labels": torch.tensor([1], device=images[0].device),
                }
                for _ in images
            ]
        raise TypeError("Unexpected input format")


class _DummyHFDetectionModel(nn.Module):
    """Lightweight HF-style detection model expecting pixel_values + labels."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_values=None, labels=None):
        if labels is not None:
            loss = pixel_values.float().mean().abs()
            return {"loss_total": (self.scale * 0.0) + loss}

        batch_size = int(pixel_values.shape[0])
        logits = torch.randn(batch_size, 5, 3, device=pixel_values.device)
        return SimpleNamespace(logits=logits)


def test_segmentation_engine_supports_dict_outputs():
    cfg = {
        "training": {
            "steps_or_epochs": "steps",
            "num_classes": 3,
            "ignore_index": 255,
            "loss": "cross_entropy",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 2,
            "lr": 1e-3,
            "scheduler": "none",
            "optimizer": {"class_name": "Adam", "args": {}},
        },
        "fed": {"rounds": 1},
    }

    engine = SegmentationEngine(_DictSegModel(num_classes=3), cfg, torch.device("cpu"))
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 16, 16), torch.randint(0, 3, (4, 16, 16))),
        batch_size=2,
    )

    train_metrics = engine.train_one_round(loader)
    val_metrics = engine.evaluate(loader)

    assert train_metrics["num_steps_current_round"] == 2.0
    assert "train_loss" in train_metrics
    assert "val_loss" in val_metrics


def test_segmentation_engine_rejects_out_of_range_masks_by_default():
    cfg = {
        "training": {
            "steps_or_epochs": "steps",
            "num_classes": 3,
            "ignore_index": 255,
            "loss": "cross_entropy",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 1,
            "lr": 1e-3,
            "scheduler": "none",
            "optimizer": {"class_name": "Adam", "args": {}},
        },
        "fed": {"rounds": 1},
    }

    engine = SegmentationEngine(_DictSegModel(num_classes=3), cfg, torch.device("cpu"))
    masks = torch.tensor([[[[0, 2], [5, 255]]]], dtype=torch.long)
    with pytest.raises(ValueError, match="out-of-range segmentation labels"):
        engine._prepare_masks(masks)


def test_segmentation_engine_can_map_out_of_range_masks_when_explicitly_enabled():
    cfg = {
        "training": {
            "steps_or_epochs": "steps",
            "num_classes": 3,
            "ignore_index": 255,
            "loss": "cross_entropy",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 1,
            "lr": 1e-3,
            "scheduler": "none",
            "segmentation_out_of_range_policy": "ignore",
            "optimizer": {"class_name": "Adam", "args": {}},
        },
        "fed": {"rounds": 1},
    }

    engine = SegmentationEngine(_DictSegModel(num_classes=3), cfg, torch.device("cpu"))
    masks = torch.tensor([[[[0, 2], [5, 255]]]], dtype=torch.long)
    prepared = engine._prepare_masks(masks)

    assert prepared.shape == (1, 2, 2)
    assert int(prepared[0, 0, 0]) == 0
    assert int(prepared[0, 0, 1]) == 2
    assert int(prepared[0, 1, 0]) == 255
    assert int(prepared[0, 1, 1]) == 255


def test_detection_engine_smoke():
    cfg = {
        "training": {
            "steps_or_epochs": "steps",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 2,
            "lr": 1e-3,
            "scheduler": "none",
            "detection_score_threshold": 0.5,
            "optimizer": {"class_name": "Adam", "args": {}},
        },
        "fed": {"rounds": 1},
    }

    engine = DetectionEngine(_DummyDetectionModel(), cfg, torch.device("cpu"))
    base_loader = DataLoader(
        TensorDataset(
            torch.randn(4, 3, 16, 16),
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]] * 4, dtype=torch.float32),
            torch.tensor([[1]] * 4, dtype=torch.int64),
        ),
        batch_size=2,
    )

    class WrappedLoader:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __len__(self):
            return len(self.wrapped)

        def __iter__(self):
            for images, boxes, labels in self.wrapped:
                yield images, {"boxes": boxes, "labels": labels}

    loader = WrappedLoader(base_loader)
    train_metrics = engine.train_one_round(loader)
    val_metrics = engine.evaluate(loader)

    assert train_metrics["num_steps_current_round"] == 2.0
    assert "train_loss" in train_metrics
    assert "avg_detections" in val_metrics
    assert "map_50" in val_metrics


def test_detection_engine_supports_variable_image_shapes_in_eval():
    cfg = {
        "training": {
            "steps_or_epochs": "steps",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 1,
            "lr": 1e-3,
            "scheduler": "none",
            "detection_score_threshold": 0.5,
            "optimizer": {"class_name": "Adam", "args": {}},
        },
        "fed": {"rounds": 1},
    }

    engine = DetectionEngine(_DummyDetectionModel(), cfg, torch.device("cpu"))

    class VariableShapeLoader:
        def __len__(self):
            return 1

        def __iter__(self):
            images = [
                torch.randn(3, 375, 500),
                torch.randn(3, 333, 500),
            ]
            targets = [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                },
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                },
            ]
            yield images, targets

    loader = VariableShapeLoader()
    train_metrics = engine.train_one_round(loader)
    val_metrics = engine.evaluate(loader)

    assert train_metrics["num_steps_current_round"] == 1.0
    assert "train_loss" in train_metrics
    assert "avg_detections" in val_metrics
    assert "map_50" in val_metrics


def test_detection_engine_hf_adapter_with_batched_tensor_inputs():
    cfg = {
        "model": {"provider": "hf"},
        "training": {
            "steps_or_epochs": "steps",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 1,
            "lr": 1e-3,
            "scheduler": "none",
            "detection_score_threshold": 0.5,
            "optimizer": {"class_name": "Adam", "args": {}},
        },
        "fed": {"rounds": 1},
    }

    engine = DetectionEngine(_DummyHFDetectionModel(), cfg, torch.device("cpu"))
    base_loader = DataLoader(
        TensorDataset(
            torch.randn(4, 3, 32, 32),
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]] * 4, dtype=torch.float32),
            torch.tensor([[1]] * 4, dtype=torch.int64),
        ),
        batch_size=2,
    )

    class WrappedLoader:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __len__(self):
            return len(self.wrapped)

        def __iter__(self):
            for images, boxes, labels in self.wrapped:
                yield images, {"boxes": boxes, "labels": labels}

    loader = WrappedLoader(base_loader)
    train_metrics = engine.train_one_round(loader)
    val_metrics = engine.evaluate(loader)

    assert train_metrics["num_steps_current_round"] == 1.0
    assert "train_loss" in train_metrics
    assert "avg_detections" in val_metrics


def test_detection_engine_hf_adapter_rejects_variable_shape_batches():
    cfg = {
        "model": {"provider": "hf"},
        "training": {
            "steps_or_epochs": "steps",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 1,
            "lr": 1e-3,
            "scheduler": "none",
            "detection_score_threshold": 0.5,
            "optimizer": {"class_name": "Adam", "args": {}},
        },
        "fed": {"rounds": 1},
    }

    engine = DetectionEngine(_DummyHFDetectionModel(), cfg, torch.device("cpu"))

    class VariableShapeLoader:
        def __len__(self):
            return 1

        def __iter__(self):
            images = [
                torch.randn(3, 375, 500),
                torch.randn(3, 333, 500),
            ]
            targets = [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                },
                {
                    "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                },
            ]
            yield images, targets

    with pytest.raises(RuntimeError, match="HF detection adapter requires batched tensor inputs"):
        engine.train_one_round(VariableShapeLoader())


def test_model_factory_torchvision_path():
    model = create_model(
        model_cfg={
            "name": "resnet18",
            "provider": "torchvision",
            "args": {"weights": None},
        },
        task_type="classification",
        num_classes=5,
    )
    assert isinstance(model, nn.Module)


def test_model_factory_torchvision_detection_path():
    model = create_model(
        model_cfg={
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "provider": "torchvision",
            "args": {"weights": None, "weights_backbone": None},
        },
        task_type="detection",
        num_classes=3,
    )
    assert isinstance(model, nn.Module)


def test_model_factory_torchvision_segmentation_path():
    model = create_model(
        model_cfg={
            "name": "fcn_resnet50",
            "provider": "torchvision",
            "args": {"weights": None, "weights_backbone": None},
        },
        task_type="segmentation",
        num_classes=4,
    )
    assert isinstance(model, nn.Module)


def test_model_factory_torchvision_deeplab_segmentation_path():
    model = create_model(
        model_cfg={
            "name": "deeplabv3_resnet50",
            "provider": "torchvision",
            "args": {"weights": None, "weights_backbone": None},
        },
        task_type="segmentation",
        num_classes=4,
    )
    assert isinstance(model, nn.Module)


def test_model_factory_rejects_provider_task_mismatch():
    with pytest.raises(ValueError, match="does not support task 'segmentation'"):
        create_model(
            model_cfg={
                "name": "resnet18",
                "provider": "timm",
                "args": {"pretrained": False},
            },
            task_type="segmentation",
            num_classes=3,
        )


def test_model_factory_rejects_non_mapping_args():
    with pytest.raises(TypeError, match="model_cfg.args must be a mapping"):
        create_model(
            model_cfg={
                "name": "resnet18",
                "provider": "torchvision",
                "args": ["not", "a", "mapping"],
            },
            task_type="classification",
            num_classes=3,
        )


def test_model_factory_rejects_classification_arch_for_detection_task():
    with pytest.raises(RuntimeError, match="model 'resnet18'"):
        create_model(
            model_cfg={
                "name": "resnet18",
                "provider": "torchvision",
                "args": {"weights": None},
            },
            task_type="detection",
            num_classes=3,
        )


def test_model_factory_wraps_adapter_oserror(monkeypatch):
    from fl_training.models.adapters import timm as timm_adapter

    def _raise_oserror(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(timm_adapter, "create_model", _raise_oserror)

    with pytest.raises(RuntimeError, match="Model factory failed"):
        create_model(
            model_cfg={
                "name": "resnet18",
                "provider": "timm",
                "args": {"pretrained": False},
            },
            task_type="classification",
            num_classes=3,
        )


def test_hf_adapter_normalizes_load_error(monkeypatch):
    hf_adapter = pytest.importorskip("fl_training.models.adapters.hf")

    def _raise_oserror(*args, **kwargs):
        raise OSError("temporary network failure")

    monkeypatch.setattr(
        hf_adapter.AutoModelForImageClassification,
        "from_pretrained",
        _raise_oserror,
    )

    with pytest.raises(RuntimeError, match="Unable to create Hugging Face model"):
        hf_adapter.create_model(
            name="google/vit-base-patch16-224",
            num_classes=3,
            task_type="classification",
        )
