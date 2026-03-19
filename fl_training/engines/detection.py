from copy import deepcopy
import logging
from typing import Dict, Any, Optional, Callable, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from ._common import build_scheduler, cycle_batches, move_optimizer_state

log = logging.getLogger(__name__)

FORWARD_ATTEMPT_ERRORS = (
    TypeError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    NotImplementedError,
    AssertionError,
)


def _recursive_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _recursive_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        out = [_recursive_to_device(v, device) for v in value]
        return type(value)(out) if isinstance(value, tuple) else out
    return value


class _DetectionProviderAdapter:
    """Provider-specific detection I/O layer.

    Input contract is framework-agnostic:
    - images: list[Tensor(C,H,W)] with variable shapes allowed
    - targets: list[dict] canonical detection targets (boxes, labels, ...)
    """

    name = "generic"

    def forward_train(
        self,
        model: nn.Module,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
        targets: Any,
    ) -> Any:
        raise NotImplementedError

    def forward_eval(
        self,
        model: nn.Module,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
    ) -> Any:
        raise NotImplementedError

    @property
    def background_class_at_zero(self) -> bool:
        """Whether label 0 is reserved for background."""
        return True


class _TorchvisionDetectionAdapter(_DetectionProviderAdapter):
    name = "torchvision"

    def forward_train(
        self,
        model: nn.Module,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
        targets: Any,
    ) -> Any:
        if targets is None:
            raise ValueError("Torchvision detection training requires targets.")
        return model(images_list, targets)

    def forward_eval(
        self,
        model: nn.Module,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
    ) -> Any:
        return model(images_list)


class _HFDetectionAdapter(_DetectionProviderAdapter):
    name = "hf"

    @property
    def background_class_at_zero(self) -> bool:
        return False

    @staticmethod
    def _require_tensor_batch(images_tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if images_tensor is None:
            raise ValueError(
                "HF detection adapter requires batched tensor inputs with uniform spatial shape. "
                "Configure detection preprocessing/collate to produce fixed-size images."
            )
        return images_tensor

    @staticmethod
    def _xyxy_abs_to_cxcywh_norm(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        """Convert absolute xyxy boxes to normalised cxcywh."""
        if boxes.numel() == 0:
            return boxes

        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = ((x1 + x2) / 2.0) / float(img_w)
        cy = ((y1 + y2) / 2.0) / float(img_h)
        bw = (x2 - x1) / float(img_w)
        bh = (y2 - y1) / float(img_h)
        return torch.stack([cx, cy, bw, bh], dim=-1)

    @staticmethod
    def _to_hf_labels(targets: Any, images_list: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        if targets is None:
            raise ValueError("HF detection training requires targets.")
        if not isinstance(targets, list):
            raise ValueError("HF detection adapter expects list-of-dicts targets.")
        if len(targets) != len(images_list):
            raise ValueError("HF detection adapter requires one target dictionary per image.")

        labels: List[Dict[str, torch.Tensor]] = []
        for image_tensor, item in zip(images_list, targets):
            if not isinstance(item, dict):
                raise ValueError("HF detection target entries must be dictionaries.")
            if "boxes" not in item or "labels" not in item:
                raise KeyError("HF detection target requires 'boxes' and 'labels'.")

            boxes = torch.as_tensor(item["boxes"], dtype=torch.float32, device=image_tensor.device).reshape(-1, 4)
            class_labels = torch.as_tensor(item["labels"], dtype=torch.int64, device=image_tensor.device).reshape(-1)
            if boxes.shape[0] != class_labels.shape[0]:
                raise ValueError("HF detection target has mismatched boxes/labels lengths.")

            h = int(image_tensor.shape[-2])
            w = int(image_tensor.shape[-1])
            boxes = _HFDetectionAdapter._xyxy_abs_to_cxcywh_norm(boxes, img_w=w, img_h=h)

            if "orig_size" in item:
                orig_size = torch.as_tensor(item["orig_size"], dtype=torch.int64, device=image_tensor.device).reshape(-1)
                if orig_size.numel() != 2:
                    raise ValueError("HF detection target 'orig_size' must contain [height, width].")
            else:
                orig_size = torch.tensor([h, w], dtype=torch.int64, device=image_tensor.device)

            labels.append(
                {
                    "class_labels": class_labels,
                    "boxes": boxes,
                    "orig_size": orig_size,
                    "size": torch.tensor([h, w], dtype=torch.int64, device=image_tensor.device),
                }
            )
        return labels

    def forward_train(
        self,
        model: nn.Module,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
        targets: Any,
    ) -> Any:
        batched = self._require_tensor_batch(images_tensor)
        hf_labels = self._to_hf_labels(targets, images_list)
        return model(pixel_values=batched, labels=hf_labels)

    def forward_eval(
        self,
        model: nn.Module,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
    ) -> Any:
        batched = self._require_tensor_batch(images_tensor)
        return model(pixel_values=batched)


class _GenericDetectionAdapter(_DetectionProviderAdapter):
    name = "generic"

    def forward_train(
        self,
        model: nn.Module,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
        targets: Any,
    ) -> Any:
        if targets is None:
            raise ValueError("Detection training requires targets.")

        errors = []
        attempts = [lambda: model(images_list, targets)]  # torchvision-style
        if images_tensor is not None:
            attempts.append(lambda: model(images_tensor, targets))  # generic tensor+targets
            attempts.append(lambda: model(pixel_values=images_tensor, labels=targets))  # HF-style

        for attempt in attempts:
            try:
                return attempt()
            except FORWARD_ATTEMPT_ERRORS as exc:
                errors.append(exc)

        raise RuntimeError(f"Unable to run detection forward pass with targets. Last error: {errors[-1]}")

    def forward_eval(
        self,
        model: nn.Module,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
    ) -> Any:
        errors = []
        attempts = [lambda: model(images_list)]  # torchvision-style
        if images_tensor is not None:
            attempts.append(lambda: model(images_tensor))  # generic tensor
            attempts.append(lambda: model(pixel_values=images_tensor))  # HF-style

        for attempt in attempts:
            try:
                return attempt()
            except FORWARD_ATTEMPT_ERRORS as exc:
                errors.append(exc)

        raise RuntimeError(f"Unable to run detection inference. Last error: {errors[-1]}")


def _infer_provider(model: nn.Module) -> str:
    module_path = getattr(type(model), "__module__", "") or ""
    if "torchvision" in module_path:
        return "torchvision"
    if "transformers" in module_path:
        return "hf"
    return "generic"


def _resolve_detection_adapter(provider: str, model: Optional[nn.Module] = None) -> _DetectionProviderAdapter:
    normalized = str(provider).lower()
    if normalized in {"generic", "auto", ""} and model is not None:
        inferred = _infer_provider(model)
        if inferred != "generic":
            log.info("Auto-detected detection provider from model: %s", inferred)
            normalized = inferred
    if normalized == "torchvision":
        return _TorchvisionDetectionAdapter()
    if normalized in {"hf", "huggingface", "transformers"}:
        return _HFDetectionAdapter()
    return _GenericDetectionAdapter()


class DetectionEngine:
    """Stateful detection engine for federated local training."""

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model
        self.config = config
        self.compute_device = device
        self.cpu_device = torch.device("cpu")

        training_cfg = self.config["training"]
        self.steps_or_epochs = training_cfg["steps_or_epochs"]
        self.current_round = 0
        self.current_global_step = 0

        self.optim_resume = training_cfg["optim_resume"]
        self.optimizer = None
        self.optimizer_state = None

        self.scheduler = None
        self.scheduler_state = None
        self.max_steps = int(self.config["fed"]["rounds"]) * int(training_cfg["local_steps"])
        self.lr_min = float(training_cfg.get("lr_min", 0.0))

        self.amp_enabled = bool(training_cfg["amp_enabled"])
        self.scaler = GradScaler(enabled=self.amp_enabled)

        self.grad_clipping_type = str(training_cfg.get("grad_clipping_type", "none")).lower()
        self.grad_clipping_value = float(training_cfg.get("grad_clipping_value", 0.0))
        self.score_threshold = float(training_cfg.get("detection_score_threshold", 0.5))
        self.map_iou_threshold = float(training_cfg.get("detection_map_iou_threshold", 0.5))

        model_cfg = self.config.get("model")
        provider = "generic" if model_cfg is None else str(model_cfg.get("provider", "generic")).lower()
        self.io_adapter = _resolve_detection_adapter(provider, model=self.model)
        log.info("Detection I/O adapter: provider=%s adapter=%s", provider, self.io_adapter.name)

    def configure_optimizer(self):
        training_cfg = self.config["training"]
        optimizer_cfg = training_cfg["optimizer"]
        opt_class_name = optimizer_cfg["class_name"]
        opt_args = optimizer_cfg["args"]
        lr = float(training_cfg["lr"])

        OptimizerClass = getattr(optim, opt_class_name)
        self.optimizer = OptimizerClass(self.model.parameters(), lr=lr, **opt_args)

        if self.optim_resume == "resume" and self.optimizer_state:
            self.optimizer.load_state_dict(self.optimizer_state)

        self.scheduler = build_scheduler(self.optimizer, training_cfg, max_steps=self.max_steps)

        if self.scheduler_state:
            self.scheduler.load_state_dict(self.scheduler_state)

    def _get_batch_iterator(self, dataloader: DataLoader) -> Iterator:
        return cycle_batches(dataloader)

    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[Any, Any]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            images = batch.get("images")
            targets = batch.get("targets")
            if images is not None:
                return images, targets
        raise ValueError(
            "Detection batch format not supported. Expected (images, targets) or "
            "{'images': ..., 'targets': ...}."
        )

    def _prepare_images(self, images: Any) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        if torch.is_tensor(images):
            images_tensor = images.to(self.compute_device, non_blocking=True)
            if images_tensor.ndim != 4:
                raise ValueError(f"Expected image tensor with shape [B,C,H,W], got {tuple(images_tensor.shape)}")
            images_list = [img for img in images_tensor]
            return images_tensor, images_list

        if isinstance(images, (list, tuple)):
            images_list = [_recursive_to_device(img, self.compute_device) for img in images]
            if not images_list or not all(torch.is_tensor(img) for img in images_list):
                raise ValueError("Image list must contain tensors.")
            # Torchvision detection backends support list-of-tensors with variable
            # spatial shapes. Only stack when every image shares the same shape.
            if all(img.shape == images_list[0].shape for img in images_list):
                images_tensor = torch.stack(images_list, dim=0)
            else:
                images_tensor = None
            return images_tensor, images_list

        raise ValueError(f"Unsupported images type for detection: {type(images).__name__}")

    def _prepare_targets(self, targets: Any) -> Optional[Any]:
        if targets is None:
            return None

        if isinstance(targets, dict):
            # Convert collated dict-of-tensors into list-of-dicts when possible.
            tensor_keys = [k for k, v in targets.items() if torch.is_tensor(v)]
            if tensor_keys:
                batch_size = targets[tensor_keys[0]].shape[0]
                target_list = []
                for idx in range(batch_size):
                    item = {}
                    for key, value in targets.items():
                        if torch.is_tensor(value):
                            item[key] = value[idx]
                        else:
                            item[key] = value
                    target_list.append(item)
                return _recursive_to_device(target_list, self.compute_device)
            return _recursive_to_device(targets, self.compute_device)

        if isinstance(targets, (list, tuple)):
            return _recursive_to_device(list(targets), self.compute_device)

        raise ValueError(
            f"Unsupported target type for detection: {type(targets).__name__}. "
            "Provide list/dict targets compatible with model backend."
        )

    @staticmethod
    def _loss_from_outputs(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, dict):
            explicit_loss = outputs.get("loss")
            if torch.is_tensor(explicit_loss):
                return explicit_loss

            losses = [
                value for key, value in outputs.items()
                if torch.is_tensor(value) and "loss" in str(key).lower()
            ]
            if not losses:
                tensor_items = [(key, value) for key, value in outputs.items() if torch.is_tensor(value)]
                if len(tensor_items) == 1:
                    return tensor_items[0][1]
                if tensor_items:
                    prediction_keys = {"boxes", "scores", "labels", "logits", "pred_boxes", "masks"}
                    if not any(str(key) in prediction_keys for key, _ in tensor_items):
                        return sum(value for _, value in tensor_items)
                raise ValueError("Detection model returned a dict with no tensor losses.")
            return sum(losses)

        model_loss = getattr(outputs, "loss", None)
        if torch.is_tensor(model_loss):
            return model_loss

        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            return outputs[0]

        raise ValueError("Could not extract detection loss from model outputs.")

    def _predictions_from_outputs(self, outputs: Any) -> Any:
        if isinstance(outputs, list):
            return outputs
        if isinstance(outputs, dict):
            if any(key in outputs for key in ("pred_boxes", "scores", "boxes")):
                return outputs
            return None
        if hasattr(outputs, "pred_boxes") or hasattr(outputs, "logits"):
            return outputs
        return None

    @staticmethod
    def _extract_size_hw(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        size = torch.as_tensor(value, dtype=torch.int64).reshape(-1)
        if size.numel() != 2:
            return None
        return size

    @staticmethod
    def _cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes

        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - (bw / 2.0)) * float(img_w)
        y1 = (cy - (bh / 2.0)) * float(img_h)
        x2 = (cx + (bw / 2.0)) * float(img_w)
        y2 = (cy + (bh / 2.0)) * float(img_h)
        out = torch.stack([x1, y1, x2, y2], dim=-1)
        out[..., 0::2] = torch.clamp(out[..., 0::2], min=0.0, max=float(img_w))
        out[..., 1::2] = torch.clamp(out[..., 1::2], min=0.0, max=float(img_h))
        return out

    def _prediction_target_sizes(
        self,
        images_list: List[torch.Tensor],
        targets: Optional[Any],
    ) -> torch.Tensor:
        sizes: List[torch.Tensor] = []
        targets_list = targets if isinstance(targets, list) and len(targets) == len(images_list) else [None] * len(images_list)
        for image_tensor, target in zip(images_list, targets_list):
            size = None
            if isinstance(target, dict):
                size = self._extract_size_hw(target.get("size"))
                if size is None:
                    size = self._extract_size_hw(target.get("orig_size"))
            if size is None:
                size = torch.tensor(
                    [int(image_tensor.shape[-2]), int(image_tensor.shape[-1])],
                    dtype=torch.int64,
                )
            sizes.append(size)
        return torch.stack(sizes, dim=0).to(self.compute_device)

    def _normalize_predictions(
        self,
        predictions: Any,
        images_list: List[torch.Tensor],
        targets: Optional[Any] = None,
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        if isinstance(predictions, list):
            return predictions

        logits = None
        pred_boxes = None
        if isinstance(predictions, dict):
            logits = predictions.get("logits")
            pred_boxes = predictions.get("pred_boxes")
            if "boxes" in predictions and torch.is_tensor(predictions["boxes"]):
                return [predictions]
        else:
            logits = getattr(predictions, "logits", None)
            pred_boxes = getattr(predictions, "pred_boxes", None)

        if not torch.is_tensor(logits) or not torch.is_tensor(pred_boxes):
            return None
        if logits.ndim != 3 or pred_boxes.ndim != 3 or pred_boxes.shape[0] != len(images_list):
            return None

        target_sizes = self._prediction_target_sizes(images_list, targets)
        use_focal_loss = bool(getattr(getattr(self.model, "config", None), "use_focal_loss", False))
        if use_focal_loss:
            class_scores = torch.sigmoid(logits)
        else:
            class_scores = torch.softmax(logits, dim=-1)
            if class_scores.shape[-1] > 1:
                class_scores = class_scores[..., :-1]

        scores, labels = torch.max(class_scores, dim=-1)
        normalized: List[Dict[str, torch.Tensor]] = []
        for idx, image_tensor in enumerate(images_list):
            size_hw = target_sizes[idx]
            boxes_xyxy = self._cxcywh_norm_to_xyxy_abs(
                pred_boxes[idx],
                img_w=int(size_hw[1].item()),
                img_h=int(size_hw[0].item()),
            )
            normalized.append(
                {
                    "boxes": boxes_xyxy.to(device=image_tensor.device),
                    "scores": scores[idx].to(device=image_tensor.device),
                    "labels": labels[idx].to(device=image_tensor.device, dtype=torch.int64),
                }
            )
        return normalized

    def _forward_with_loss(
        self,
        images_tensor: Optional[torch.Tensor],
        images_list: List[torch.Tensor],
        targets: Any,
    ) -> Tuple[torch.Tensor, Any]:
        try:
            outputs = self.io_adapter.forward_train(
                self.model,
                images_tensor=images_tensor,
                images_list=images_list,
                targets=targets,
            )
            loss = self._loss_from_outputs(outputs)
            return loss, self._predictions_from_outputs(outputs)
        except FORWARD_ATTEMPT_ERRORS as exc:
            raise RuntimeError(
                f"Detection train forward failed for adapter '{self.io_adapter.name}': {exc}"
            ) from exc

    def _forward_inference(self, images_tensor: Optional[torch.Tensor], images_list: List[torch.Tensor]) -> Any:
        try:
            return self.io_adapter.forward_eval(
                self.model,
                images_tensor=images_tensor,
                images_list=images_list,
            )
        except FORWARD_ATTEMPT_ERRORS as exc:
            raise RuntimeError(
                f"Detection inference failed for adapter '{self.io_adapter.name}': {exc}"
            ) from exc

    def _count_detections(
        self,
        predictions: Any,
        images_list: List[torch.Tensor],
        targets: Optional[Any] = None,
    ) -> float:
        if predictions is None:
            return 0.0

        normalized = self._normalize_predictions(predictions, images_list, targets)
        if normalized is not None:
            counts = []
            for pred in normalized:
                if isinstance(pred, dict) and "scores" in pred and torch.is_tensor(pred["scores"]):
                    counts.append(float((pred["scores"] > self.score_threshold).sum().item()))
                elif isinstance(pred, dict) and "boxes" in pred and torch.is_tensor(pred["boxes"]):
                    counts.append(float(pred["boxes"].shape[0]))
                else:
                    counts.append(0.0)
            return float(sum(counts) / max(len(counts), 1))

        return 0.0

    @staticmethod
    def _box_iou_xyxy(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """IoU between one box [4] and many boxes [N,4] in xyxy format."""
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32, device=box.device)

        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])

        inter_w = torch.clamp(x2 - x1, min=0.0)
        inter_h = torch.clamp(y2 - y1, min=0.0)
        inter = inter_w * inter_h

        area_box = torch.clamp(box[2] - box[0], min=0.0) * torch.clamp(box[3] - box[1], min=0.0)
        area_boxes = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
        union = area_box + area_boxes - inter
        return inter / torch.clamp(union, min=1e-12)

    @staticmethod
    def _voc_ap(recalls: List[float], precisions: List[float]) -> float:
        """Continuous VOC AP from precision-recall curve."""
        if not recalls:
            return 0.0

        mrec = [0.0, *recalls, 1.0]
        mpre = [0.0, *precisions, 0.0]
        for idx in range(len(mpre) - 1, 0, -1):
            mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

        ap = 0.0
        for idx in range(len(mrec) - 1):
            if mrec[idx + 1] != mrec[idx]:
                ap += (mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]
        return float(ap)

    def _extract_eval_entries(
        self,
        predictions: Any,
        targets: Any,
        batch_size: int,
    ) -> Optional[Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]]:
        """Normalize predictions/targets to list-of-dicts for AP computation."""
        if not isinstance(predictions, list):
            return None
        if not isinstance(targets, list):
            return None
        if len(predictions) != len(targets):
            return None

        pred_entries: List[Dict[str, torch.Tensor]] = []
        gt_entries: List[Dict[str, torch.Tensor]] = []
        for idx in range(batch_size):
            pred = predictions[idx]
            tgt = targets[idx]
            if not isinstance(pred, dict) or not isinstance(tgt, dict):
                return None
            if "boxes" not in pred or "boxes" not in tgt or "labels" not in tgt:
                return None

            pred_boxes = torch.as_tensor(pred["boxes"], dtype=torch.float32, device=self.cpu_device).reshape(-1, 4)
            pred_labels = torch.as_tensor(pred.get("labels", torch.zeros((pred_boxes.shape[0],), dtype=torch.int64)), dtype=torch.int64, device=self.cpu_device).reshape(-1)
            pred_scores = torch.as_tensor(pred.get("scores", torch.ones((pred_boxes.shape[0],), dtype=torch.float32)), dtype=torch.float32, device=self.cpu_device).reshape(-1)
            if pred_labels.shape[0] != pred_boxes.shape[0] or pred_scores.shape[0] != pred_boxes.shape[0]:
                return None

            gt_boxes = torch.as_tensor(tgt["boxes"], dtype=torch.float32, device=self.cpu_device).reshape(-1, 4)
            gt_labels = torch.as_tensor(tgt["labels"], dtype=torch.int64, device=self.cpu_device).reshape(-1)
            if gt_labels.shape[0] != gt_boxes.shape[0]:
                return None

            pred_entries.append(
                {
                    "boxes": pred_boxes,
                    "labels": pred_labels,
                    "scores": pred_scores,
                }
            )
            gt_entries.append(
                {
                    "boxes": gt_boxes,
                    "labels": gt_labels,
                }
            )

        return pred_entries, gt_entries

    def _compute_map50(
        self,
        pred_entries: List[Dict[str, torch.Tensor]],
        gt_entries: List[Dict[str, torch.Tensor]],
    ) -> Optional[float]:
        """Compute mAP@0.5 over classes present in GT."""
        skip_zero = self.io_adapter.background_class_at_zero
        class_ids = set()
        for gt in gt_entries:
            labels = gt["labels"]
            for cls in labels.tolist():
                cls_int = int(cls)
                if skip_zero and cls_int == 0:
                    continue
                class_ids.add(cls_int)

        if not class_ids:
            return None

        aps: List[float] = []
        for cls in sorted(class_ids):
            gt_by_image: Dict[int, torch.Tensor] = {}
            total_gt = 0
            for img_idx, gt in enumerate(gt_entries):
                class_mask = gt["labels"] == cls
                boxes = gt["boxes"][class_mask]
                gt_by_image[img_idx] = boxes
                total_gt += int(boxes.shape[0])

            if total_gt == 0:
                continue

            predictions_for_cls: List[Tuple[float, int, torch.Tensor]] = []
            for img_idx, pred in enumerate(pred_entries):
                class_mask = pred["labels"] == cls
                cls_boxes = pred["boxes"][class_mask]
                cls_scores = pred["scores"][class_mask]
                for score, box in zip(cls_scores.tolist(), cls_boxes):
                    predictions_for_cls.append((float(score), img_idx, box))

            if not predictions_for_cls:
                aps.append(0.0)
                continue

            predictions_for_cls.sort(key=lambda x: x[0], reverse=True)
            matched = {
                img_idx: torch.zeros((boxes.shape[0],), dtype=torch.bool, device=self.cpu_device)
                for img_idx, boxes in gt_by_image.items()
            }

            tp: List[float] = []
            fp: List[float] = []
            for _, img_idx, pred_box in predictions_for_cls:
                gt_boxes = gt_by_image.get(img_idx)
                if gt_boxes is None or gt_boxes.numel() == 0:
                    tp.append(0.0)
                    fp.append(1.0)
                    continue

                ious = self._box_iou_xyxy(pred_box, gt_boxes)
                best_iou, best_idx = torch.max(ious, dim=0)
                if float(best_iou.item()) >= self.map_iou_threshold and not bool(matched[img_idx][best_idx]):
                    matched[img_idx][best_idx] = True
                    tp.append(1.0)
                    fp.append(0.0)
                else:
                    tp.append(0.0)
                    fp.append(1.0)

            cum_tp: List[float] = []
            cum_fp: List[float] = []
            tp_total = 0.0
            fp_total = 0.0
            for is_tp, is_fp in zip(tp, fp):
                tp_total += is_tp
                fp_total += is_fp
                cum_tp.append(tp_total)
                cum_fp.append(fp_total)

            recalls = [v / float(total_gt) for v in cum_tp]
            precisions = [
                cum_tp[idx] / max(cum_tp[idx] + cum_fp[idx], 1e-12)
                for idx in range(len(cum_tp))
            ]
            aps.append(self._voc_ap(recalls, precisions))

        if not aps:
            return None
        return float(sum(aps) / len(aps))

    def train_one_round(
        self,
        train_loader: DataLoader,
        metric_callback: Optional[Callable[[Dict[str, float]], None]] = None
    ) -> Dict[str, float]:
        self.model.to(self.compute_device)
        self.model.train()

        if self.steps_or_epochs == "epochs":
            num_steps = int(self.config["training"]["local_steps"]) * len(train_loader)
            self.max_steps = int(self.config["fed"]["rounds"]) * num_steps
        else:
            num_steps = int(self.config["training"]["local_steps"])

        if num_steps <= 0:
            metrics = {
                "round": float(self.current_round),
                "train_loss": 0.0,
                "avg_detections": 0.0,
                "num_steps_current_round": 0.0,
            }
            if metric_callback:
                metric_callback(metrics)
            return metrics

        self.configure_optimizer()
        batch_iterator = self._get_batch_iterator(train_loader)

        round_loss = 0.0
        round_detection_count = 0.0
        executed_steps = 0

        for _ in range(num_steps):
            batch = next(batch_iterator)
            images, raw_targets = self._unpack_batch(batch)
            images_tensor, images_list = self._prepare_images(images)
            targets = self._prepare_targets(raw_targets)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp_enabled, device_type=self.compute_device.type):
                loss, predictions = self._forward_with_loss(images_tensor, images_list, targets)

            self.scaler.scale(loss).backward()

            if self.grad_clipping_type == "norm":
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping_value)
            elif self.grad_clipping_type == "value":
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clipping_value)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            round_loss += float(loss.item())
            round_detection_count += self._count_detections(predictions, images_list, targets)

            self.current_global_step += 1
            executed_steps += 1

        metrics = {
            "round": float(self.current_round),
            "train_loss": float(round_loss / executed_steps if executed_steps else 0.0),
            "avg_detections": float(round_detection_count / executed_steps if executed_steps else 0.0),
            "num_steps_current_round": float(executed_steps),
        }
        if metric_callback:
            metric_callback(metrics)

        self.model.to(self.cpu_device)
        move_optimizer_state(self.optimizer, self.cpu_device)

        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optim_resume == "resume" else None
        self.scheduler_state = deepcopy(self.scheduler.state_dict())
        self.optimizer = None
        self.scheduler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.current_round += 1
        return metrics

    def evaluate(self, val_loader: Optional[DataLoader]) -> Dict[str, float]:
        if val_loader is None:
            return {}

        self.model.to(self.compute_device)
        self.model.eval()

        total_loss = 0.0
        n_loss_batches = 0
        total_detection_count = 0.0
        n_batches = 0
        all_pred_entries: List[Dict[str, torch.Tensor]] = []
        all_gt_entries: List[Dict[str, torch.Tensor]] = []
        map_supported = True
        can_compute_eval_loss = self.io_adapter.name != "torchvision"

        with torch.no_grad():
            for batch in val_loader:
                images, raw_targets = self._unpack_batch(batch)
                images_tensor, images_list = self._prepare_images(images)
                targets = self._prepare_targets(raw_targets)

                predictions = None
                if targets is not None and can_compute_eval_loss:
                    try:
                        loss, predictions = self._forward_with_loss(images_tensor, images_list, targets)
                        total_loss += float(loss.item())
                        n_loss_batches += 1
                    except RuntimeError:
                        predictions = None

                    if predictions is None:
                        predictions = self._forward_inference(images_tensor, images_list)
                else:
                    predictions = self._forward_inference(images_tensor, images_list)

                total_detection_count += self._count_detections(predictions, images_list, targets)
                n_batches += 1

                normalized_predictions = self._normalize_predictions(predictions, images_list, targets)
                normalized = self._extract_eval_entries(normalized_predictions, targets, len(images_list))
                if normalized is None:
                    map_supported = False
                else:
                    pred_entries, gt_entries = normalized
                    all_pred_entries.extend(pred_entries)
                    all_gt_entries.extend(gt_entries)

        self.model.to(self.cpu_device)
        metrics = {
            "avg_detections": float(total_detection_count / n_batches if n_batches else 0.0),
        }
        if n_loss_batches > 0:
            metrics["val_loss"] = float(total_loss / n_loss_batches)
        map_50 = self._compute_map50(all_pred_entries, all_gt_entries) if map_supported else None
        if map_50 is not None:
            metrics["map_50"] = float(map_50)
            metrics["accuracy"] = float(map_50)
        else:
            metrics["accuracy"] = float(metrics["avg_detections"])
        return metrics
