import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset

from fl_training.engines.segmentation import SegmentationEngine
from fl_training.engines.detection import DetectionEngine
from fl_training.engines.classification import ClassificationEngine
from fl_training.models.factory import create_model
from fl_training.models.adapters import torchvision as torchvision_adapter


class _DictSegModel(nn.Module):
    """Torchvision-style segmentation output: {'out': logits}."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x):
        return {"out": self.head(x)}


class _TorchvisionOnlySegModel(nn.Module):
    """Torchvision-style segmentation model that rejects HF-only kwargs."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return {"out": self.head(x)}


class _HFStyleSegModel(nn.Module):
    """HF-style segmentation model expecting pixel_values + labels."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, pixel_values=None, labels=None):
        logits = self.head(pixel_values)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


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


class _DummyFCOSDetectionModel(nn.Module):
    """Torchvision FCOS-style training losses without `loss_*` key names."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if isinstance(images, list):
            if targets is not None:
                base = (self.scale * 0.0) + torch.stack([img.float().mean() for img in images]).abs().mean()
                return {
                    "classification": base,
                    "bbox_regression": base * 0.5,
                    "bbox_ctrness": base * 0.25,
                }
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


class _DummyHFDetrModel(nn.Module):
    """HF-style DETR-like model returning logits and normalized boxes."""

    def __init__(self, logits: torch.Tensor, pred_boxes: torch.Tensor, use_focal_loss: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self._logits = logits
        self._pred_boxes = pred_boxes
        self.config = SimpleNamespace(use_focal_loss=use_focal_loss)

    def forward(self, pixel_values=None, labels=None):
        batch_size = int(pixel_values.shape[0])
        logits = self._logits.to(pixel_values.device)
        pred_boxes = self._pred_boxes.to(pixel_values.device)
        if logits.shape[0] != batch_size:
            logits = logits.expand(batch_size, -1, -1)
        if pred_boxes.shape[0] != batch_size:
            pred_boxes = pred_boxes.expand(batch_size, -1, -1)

        loss = (self.scale * 0.0) + pixel_values.float().mean().abs()
        if labels is not None:
            return SimpleNamespace(loss=loss, logits=logits, pred_boxes=pred_boxes)
        return SimpleNamespace(logits=logits, pred_boxes=pred_boxes)


class _AutoDetectedHFDetrModel(_DummyHFDetrModel):
    pass


_AutoDetectedHFDetrModel.__module__ = "transformers.models.detr.modeling_detr"


class _PerfectClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        del x
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float32, device=self.scale.device)
        return logits * self.scale


def test_classification_engine_reports_fractional_accuracy():
    cfg = {
        "training": {
            "steps_or_epochs": "steps",
            "loss": "cross_entropy",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 1,
            "lr": 1e-3,
            "lr_min": 0.0,
            "scheduler": "none",
            "optimizer": {"class_name": "Adam", "args": {}},
        },
        "fed": {"rounds": 1},
    }

    engine = ClassificationEngine(_PerfectClassificationModel(), cfg, torch.device("cpu"))
    loader = DataLoader(
        TensorDataset(
            torch.randn(2, 3, 8, 8),
            torch.tensor([0, 1], dtype=torch.long),
        ),
        batch_size=2,
    )

    train_metrics = engine.train_one_round(loader)
    val_metrics = engine.evaluate(loader)

    assert train_metrics["train_accuracy"] == pytest.approx(1.0)
    assert val_metrics["accuracy"] == pytest.approx(1.0)


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


def test_segmentation_engine_torchvision_provider_uses_single_plain_forward():
    cfg = {
        "model": {"provider": "torchvision"},
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

    model = _TorchvisionOnlySegModel(num_classes=3)
    engine = SegmentationEngine(model, cfg, torch.device("cpu"))
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16))),
        batch_size=2,
    )

    train_metrics = engine.train_one_round(loader)

    assert train_metrics["num_steps_current_round"] == 1.0
    assert model.calls == 1


def test_segmentation_engine_hf_provider_uses_pixel_values_and_labels():
    cfg = {
        "model": {"provider": "hf"},
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

    engine = SegmentationEngine(_HFStyleSegModel(num_classes=3), cfg, torch.device("cpu"))
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16))),
        batch_size=2,
    )

    train_metrics = engine.train_one_round(loader)
    val_metrics = engine.evaluate(loader)

    assert train_metrics["num_steps_current_round"] == 1.0
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


def test_detection_engine_accepts_fcos_style_torchvision_loss_dict():
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

    engine = DetectionEngine(_DummyFCOSDetectionModel(), cfg, torch.device("cpu"))
    base_loader = DataLoader(
        TensorDataset(
            torch.randn(2, 3, 16, 16),
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]] * 2, dtype=torch.float32),
            torch.tensor([[1]] * 2, dtype=torch.int64),
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

    metrics = engine.train_one_round(WrappedLoader(base_loader))

    assert metrics["num_steps_current_round"] == 1.0
    assert metrics["train_loss"] >= 0.0


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


def test_detection_engine_hf_adapter_normalizes_training_boxes():
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

    model = _DummyHFDetrModel(
        logits=torch.zeros(1, 1, 2),
        pred_boxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
    )
    engine = DetectionEngine(model, cfg, torch.device("cpu"))
    image = torch.zeros(3, 32, 64)
    targets = [
        {
            "boxes": torch.tensor([[16.0, 8.0, 48.0, 24.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
            "orig_size": torch.tensor([40, 80], dtype=torch.int64),
        }
    ]

    labels = engine.io_adapter._to_hf_labels(targets, [image])

    assert torch.allclose(labels[0]["boxes"], torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32))
    assert torch.equal(labels[0]["class_labels"], torch.tensor([0], dtype=torch.int64))
    assert torch.equal(labels[0]["orig_size"], torch.tensor([40, 80], dtype=torch.int64))
    assert torch.equal(labels[0]["size"], torch.tensor([32, 64], dtype=torch.int64))


def test_detection_engine_auto_detects_hf_provider_when_provider_missing():
    cfg = {
        "model": {},
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

    model = _AutoDetectedHFDetrModel(
        logits=torch.zeros(1, 1, 2),
        pred_boxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
    )
    engine = DetectionEngine(model, cfg, torch.device("cpu"))

    assert engine.io_adapter.name == "hf"


def test_detection_engine_hf_eval_outputs_support_map50_for_class_zero():
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

    model = _DummyHFDetrModel(
        logits=torch.tensor([[[10.0, -10.0]]], dtype=torch.float32),
        pred_boxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
    )
    engine = DetectionEngine(model, cfg, torch.device("cpu"))

    class Loader:
        def __len__(self):
            return 1

        def __iter__(self):
            images = torch.zeros(1, 3, 32, 32)
            targets = {
                "boxes": torch.tensor([[[8.0, 8.0, 24.0, 24.0]]], dtype=torch.float32),
                "labels": torch.tensor([[0]], dtype=torch.int64),
            }
            yield images, targets

    metrics = engine.evaluate(Loader())

    assert metrics["avg_detections"] == pytest.approx(1.0)
    assert metrics["map_50"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_detection_engine_detr_no_object_logits_do_not_count_false_detections():
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

    model = _DummyHFDetrModel(
        logits=torch.tensor([[[0.1, 0.1, 10.0]]], dtype=torch.float32),
        pred_boxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        use_focal_loss=False,
    )
    engine = DetectionEngine(model, cfg, torch.device("cpu"))

    class Loader:
        def __len__(self):
            return 1

        def __iter__(self):
            images = torch.zeros(1, 3, 32, 32)
            targets = {
                "boxes": torch.tensor([[[8.0, 8.0, 24.0, 24.0]]], dtype=torch.float32),
                "labels": torch.tensor([[0]], dtype=torch.int64),
            }
            yield images, targets

    metrics = engine.evaluate(Loader())

    assert metrics["avg_detections"] == pytest.approx(0.0)


def test_detection_engine_hf_focal_loss_logits_use_sigmoid_scores():
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

    model = _DummyHFDetrModel(
        logits=torch.tensor([[[0.2, -2.0]]], dtype=torch.float32),
        pred_boxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
        use_focal_loss=True,
    )
    engine = DetectionEngine(model, cfg, torch.device("cpu"))

    class Loader:
        def __len__(self):
            return 1

        def __iter__(self):
            images = torch.zeros(1, 3, 32, 32)
            targets = {
                "boxes": torch.tensor([[[8.0, 8.0, 24.0, 24.0]]], dtype=torch.float32),
                "labels": torch.tensor([[0]], dtype=torch.int64),
            }
            yield images, targets

    metrics = engine.evaluate(Loader())

    assert metrics["avg_detections"] == pytest.approx(1.0)


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


def test_model_factory_torchvision_fcos_detection_path():
    model = create_model(
        model_cfg={
            "name": "fcos_resnet50_fpn",
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


def test_torchvision_adapter_resolves_weight_enum_strings():
    weights = torchvision_adapter._resolve_weight_enum(
        "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "detection",
    )

    assert str(weights) == "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"


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
