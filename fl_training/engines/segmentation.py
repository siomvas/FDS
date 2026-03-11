from copy import deepcopy
import logging
from typing import Dict, Any, Optional, Callable, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import autocast, GradScaler
from ._common import build_scheduler, cycle_batches, move_optimizer_state

log = logging.getLogger(__name__)


class SegmentationEngine:
    """Stateful segmentation engine for federated local training."""

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        self.model = model
        self.config = config
        self.compute_device = device
        self.cpu_device = torch.device("cpu")

        training_cfg = self.config["training"]
        self.steps_or_epochs = training_cfg["steps_or_epochs"]
        self.current_round = 0
        self.current_global_step = 0

        self.num_classes = int(training_cfg["num_classes"])
        self.ignore_index = int(training_cfg.get("ignore_index", 255))
        self.loss_name = str(training_cfg.get("loss", "cross_entropy")).lower()
        self.out_of_range_policy = str(training_cfg.get("segmentation_out_of_range_policy", "error")).lower()
        self._ce_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self._logged_out_of_range_warning = False

        if self.loss_name not in {"cross_entropy", "ce", "dice_loss", "dice", "ce_dice"}:
            raise ValueError(f"Unsupported segmentation loss: {self.loss_name}")

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

    def _prepare_masks(self, masks: torch.Tensor) -> torch.Tensor:
        masks = masks.to(self.compute_device, non_blocking=True)

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim == 4 and masks.shape[-1] == 1:
            masks = masks[..., 0]
        elif masks.ndim != 3:
            raise ValueError(f"Expected masks with shape [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")

        if masks.dtype.is_floating_point:
            max_value = float(masks.max().item()) if masks.numel() > 0 else 0.0
            if self.num_classes > 2 and max_value <= 1.0 + 1e-6:
                masks = torch.round(masks * 255.0)
            else:
                masks = torch.round(masks)

        masks = masks.long()

        # Validate label ranges early to avoid silently corrupting metrics.
        valid = (masks >= 0) & (masks < self.num_classes)
        ignore = masks == self.ignore_index
        out_of_range = ~(valid | ignore)
        if torch.any(out_of_range):
            count = int(out_of_range.sum().item())
            if self.out_of_range_policy in {"ignore", "map_to_ignore"}:
                if not self._logged_out_of_range_warning:
                    log.warning(
                        "Found %d out-of-range segmentation labels for num_classes=%d; "
                        "mapping them to ignore_index=%d.",
                        count,
                        self.num_classes,
                        self.ignore_index,
                    )
                    self._logged_out_of_range_warning = True
                masks = masks.clone()
                masks[out_of_range] = self.ignore_index
            else:
                raise ValueError(
                    "Found out-of-range segmentation labels for configured num_classes. "
                    f"count={count}, num_classes={self.num_classes}, ignore_index={self.ignore_index}. "
                    "Fix dataset labels or set training.segmentation_out_of_range_policy=ignore."
                )

        return masks

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index

        safe_targets = targets.clone()
        safe_targets[~valid] = 0
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denom = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + smooth) / (denom + smooth)

        present_classes = one_hot.sum(dims) > 0
        if present_classes.any():
            return 1.0 - dice[present_classes].mean()
        return 1.0 - dice.mean()

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_name in {"cross_entropy", "ce"}:
            return self._ce_criterion(logits, targets)
        if self.loss_name in {"dice_loss", "dice"}:
            return self._dice_loss(logits, targets)
        if self.loss_name == "ce_dice":
            ce = self._ce_criterion(logits, targets)
            dice = self._dice_loss(logits, targets)
            return ce + dice
        raise ValueError(f"Unsupported segmentation loss: {self.loss_name}")

    def _forward_with_loss(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            outputs = self.model(images, labels=targets)
        except TypeError:
            outputs = self.model(images)

        logits = outputs
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            # Torchvision semantic segmentation models usually return {"out": ..., "aux": ...}.
            if "logits" in outputs and torch.is_tensor(outputs["logits"]):
                logits = outputs["logits"]
            elif "out" in outputs and torch.is_tensor(outputs["out"]):
                logits = outputs["out"]

        if not torch.is_tensor(logits):
            raise ValueError("Segmentation model forward pass did not return tensor logits.")

        model_loss = getattr(outputs, "loss", None) if not torch.is_tensor(outputs) else None
        if model_loss is not None and self.loss_name in {"cross_entropy", "ce"}:
            return model_loss, logits

        return self._compute_loss(logits, targets), logits

    def _update_segmentation_stats(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        intersection: torch.Tensor,
        union: torch.Tensor,
    ) -> Tuple[int, int]:
        valid = targets != self.ignore_index
        correct = int((preds[valid] == targets[valid]).sum().item())
        total = int(valid.sum().item())

        for cls in range(self.num_classes):
            pred_cls = (preds == cls) & valid
            target_cls = (targets == cls) & valid
            intersection[cls] += int((pred_cls & target_cls).sum().item())
            union[cls] += int((pred_cls | target_cls).sum().item())

        return correct, total

    @staticmethod
    def _mean_iou(intersection: torch.Tensor, union: torch.Tensor) -> float:
        valid_classes = union > 0
        if not torch.any(valid_classes):
            return 0.0
        iou = intersection[valid_classes].float() / union[valid_classes].float()
        return float(iou.mean().item())

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
                "pixel_accuracy": 0.0,
                "mean_iou": 0.0,
                "num_steps_current_round": 0.0,
            }
            if metric_callback:
                metric_callback(metrics)
            return metrics

        self.configure_optimizer()
        batch_iterator = self._get_batch_iterator(train_loader)

        round_loss = 0.0
        executed_steps = 0
        total_correct = 0
        total_pixels = 0
        intersection = torch.zeros(self.num_classes, dtype=torch.long)
        union = torch.zeros(self.num_classes, dtype=torch.long)

        for _ in range(num_steps):
            images, masks = next(batch_iterator)
            images = images.to(self.compute_device, non_blocking=True)
            targets = self._prepare_masks(masks)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp_enabled, device_type=self.compute_device.type):
                loss, logits = self._forward_with_loss(images, targets)

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
            preds = torch.argmax(logits.detach(), dim=1)
            step_correct, step_total = self._update_segmentation_stats(preds, targets, intersection, union)
            total_correct += step_correct
            total_pixels += step_total

            if metric_callback:
                metric_callback({
                    "step": float(self.current_global_step),
                    "batch_loss": float(loss.item()),
                    "lr": float(self.scheduler.get_last_lr()[-1]),
                })

            self.current_global_step += 1
            executed_steps += 1

        avg_loss = round_loss / executed_steps if executed_steps > 0 else 0.0
        pixel_accuracy = 100.0 * total_correct / total_pixels if total_pixels > 0 else 0.0
        mean_iou = self._mean_iou(intersection, union)

        metrics = {
            "round": float(self.current_round),
            "train_loss": float(avg_loss),
            "pixel_accuracy": float(pixel_accuracy),
            "mean_iou": float(mean_iou),
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
        n_batches = 0
        total_correct = 0
        total_pixels = 0
        intersection = torch.zeros(self.num_classes, dtype=torch.long)
        union = torch.zeros(self.num_classes, dtype=torch.long)

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.compute_device, non_blocking=True)
                targets = self._prepare_masks(masks)
                loss, logits = self._forward_with_loss(images, targets)

                total_loss += float(loss.item())
                n_batches += 1

                preds = torch.argmax(logits, dim=1)
                step_correct, step_total = self._update_segmentation_stats(preds, targets, intersection, union)
                total_correct += step_correct
                total_pixels += step_total

        pixel_accuracy = 100.0 * total_correct / total_pixels if total_pixels > 0 else 0.0
        mean_iou = self._mean_iou(intersection, union)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        self.model.to(self.cpu_device)
        metrics = {
            "pixel_accuracy": float(pixel_accuracy),
            "mean_iou": float(mean_iou),
            "val_loss": float(avg_loss),
        }
        metrics["accuracy"] = float(pixel_accuracy)
        return metrics
