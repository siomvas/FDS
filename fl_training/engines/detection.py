from copy import deepcopy
import logging
from typing import Dict, Any, Optional, Callable, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

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

    def configure_optimizer(self):
        optimizer_cfg = self.config.get("optimizer") or self.config["training"].get("optimizer")
        if optimizer_cfg is None:
            raise ValueError("Missing optimizer configuration. Expected `optimizer` or `training.optimizer`.")

        opt_class_name = optimizer_cfg["class_name"]
        opt_args = optimizer_cfg["args"]
        lr = float(self.config["training"]["lr"])

        OptimizerClass = getattr(optim, opt_class_name)
        self.optimizer = OptimizerClass(self.model.parameters(), lr=lr, **opt_args)

        if self.optim_resume == "resume" and self.optimizer_state:
            self.optimizer.load_state_dict(self.optimizer_state)

        scheduler_name = str(self.config["training"]["scheduler"]).lower()
        if scheduler_name == "none":
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1, total_iters=self.max_steps)
        elif scheduler_name == "step":
            step_size = int(self.config.get("lr_step_size", 1000))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.1)
        elif scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_steps, eta_min=self.lr_min)
        elif scheduler_name == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=self.max_steps,
                pct_start=0.3,
                anneal_strategy="cos",
                final_div_factor=1e4,
            )
        else:
            log.warning(f"Unknown scheduler: {scheduler_name}. Falling back to constant LR.")
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1, total_iters=self.max_steps)

        if self.scheduler_state:
            self.scheduler.load_state_dict(self.scheduler_state)

    def _get_batch_iterator(self, dataloader: DataLoader) -> Iterator:
        it = iter(dataloader)
        while True:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)
            yield batch

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

    def _prepare_images(self, images: Any) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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
            images_tensor = torch.stack(images_list, dim=0)
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
            losses = [value for value in outputs.values() if torch.is_tensor(value)]
            if not losses:
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

    def _forward_with_loss(
        self,
        images_tensor: torch.Tensor,
        images_list: List[torch.Tensor],
        targets: Any,
    ) -> Tuple[torch.Tensor, Any]:
        if targets is None:
            raise ValueError("Detection training requires targets.")

        errors = []
        attempts = [
            lambda: self.model(images_list, targets),                     # torchvision detection
            lambda: self.model(images_tensor, targets),                   # generic tensor+targets
            lambda: self.model(pixel_values=images_tensor, labels=targets),  # transformers detection
        ]

        for attempt in attempts:
            try:
                outputs = attempt()
                loss = self._loss_from_outputs(outputs)
                return loss, self._predictions_from_outputs(outputs)
            except FORWARD_ATTEMPT_ERRORS as exc:
                errors.append(exc)

        raise RuntimeError(f"Unable to run detection forward pass with targets. Last error: {errors[-1]}")

    def _forward_inference(self, images_tensor: torch.Tensor, images_list: List[torch.Tensor]) -> Any:
        errors = []
        attempts = [
            lambda: self.model(images_list),             # torchvision detection
            lambda: self.model(images_tensor),           # generic tensor input
            lambda: self.model(pixel_values=images_tensor),  # transformers detection
        ]

        for attempt in attempts:
            try:
                return attempt()
            except FORWARD_ATTEMPT_ERRORS as exc:
                errors.append(exc)

        raise RuntimeError(f"Unable to run detection inference. Last error: {errors[-1]}")

    def _count_detections(self, predictions: Any, batch_size: int) -> float:
        if predictions is None:
            return 0.0

        if isinstance(predictions, list):
            counts = []
            for pred in predictions:
                if isinstance(pred, dict) and "scores" in pred and torch.is_tensor(pred["scores"]):
                    counts.append(float((pred["scores"] > self.score_threshold).sum().item()))
                elif isinstance(pred, dict) and "boxes" in pred and torch.is_tensor(pred["boxes"]):
                    counts.append(float(pred["boxes"].shape[0]))
                else:
                    counts.append(0.0)
            return float(sum(counts) / max(len(counts), 1))

        logits = getattr(predictions, "logits", None)
        if torch.is_tensor(logits):
            # Common for DETR-like models: [B, num_queries, num_classes(+1 no-object)].
            class_logits = logits
            if class_logits.shape[-1] > 1:
                class_logits = class_logits[..., :-1]
            probs = torch.softmax(class_logits, dim=-1)
            max_scores, _ = torch.max(probs, dim=-1)
            counts = (max_scores > self.score_threshold).sum(dim=-1).float()
            return float(counts.mean().item())

        return 0.0

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
            round_detection_count += self._count_detections(predictions, images_tensor.shape[0])

            if metric_callback:
                metric_callback({
                    "step": float(self.current_global_step),
                    "batch_loss": float(loss.item()),
                    "lr": float(self.scheduler.get_last_lr()[-1]),
                })

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
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.cpu_device)

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

        with torch.no_grad():
            for batch in val_loader:
                images, raw_targets = self._unpack_batch(batch)
                images_tensor, images_list = self._prepare_images(images)
                targets = self._prepare_targets(raw_targets)

                predictions = None
                if targets is not None:
                    try:
                        loss, predictions = self._forward_with_loss(images_tensor, images_list, targets)
                        total_loss += float(loss.item())
                        n_loss_batches += 1
                    except RuntimeError:
                        predictions = self._forward_inference(images_tensor, images_list)
                else:
                    predictions = self._forward_inference(images_tensor, images_list)

                total_detection_count += self._count_detections(predictions, images_tensor.shape[0])
                n_batches += 1

        self.model.to(self.cpu_device)
        metrics = {
            "avg_detections": float(total_detection_count / n_batches if n_batches else 0.0),
        }
        if n_loss_batches > 0:
            metrics["val_loss"] = float(total_loss / n_loss_batches)
        return metrics
