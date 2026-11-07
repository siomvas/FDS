from copy import deepcopy
import logging
from typing import Dict, Any, Optional, Callable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import autocast, GradScaler
from ._common import build_scheduler, cycle_batches, move_optimizer_state

log = logging.getLogger(__name__)


class ClassificationEngine:
    """
    Stateful Engine for Classification Tasks.

    Persists the Model, Optimizer, and Scheduler states across multiple Federated Learning rounds.
    This ensures that momentum buffers (SGD) and learning rate decays (Schedulers)
    are not reset effectively to zero/initial values at the start of every round.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        """
        Args:
            model: The PyTorch model (instantiated via Factory).
            config: Dictionary containing 'lr', 'optimizer', 'epochs', 'dataset', 'model' etc.
            device: torch.device('cuda' or 'cpu').
        """
        self.model = model  # CPU->GPU transfer deferred to training/evaluation time
        self.config = config
        self.compute_device = device
        self.cpu_device = torch.device("cpu")

        self.steps_or_epochs = self.config["training"]["steps_or_epochs"]
        self.current_round = 0
        self.current_global_step = 0

        if self.config["training"]["loss"] == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config['training']['loss']}")

        # Optimizer / Scheduler Related
        self.optim_resume = self.config["training"]["optim_resume"]
        self.optimizer = None
        self.optimizer_state = None

        self.scheduler = None
        self.scheduler_state = None
        self.max_steps = self.config["fed"]["rounds"] * self.config["training"]["local_steps"]
        self.lr_min = self.config["training"]["lr_min"]

        # AMP Related
        self.amp_enabled = self.config["training"]["amp_enabled"]
        self.scaler = GradScaler(enabled=self.amp_enabled)

        # Gradient clipping Related
        self.grad_clipping_type = self.config["training"]["grad_clipping_type"]
        self.grad_clipping_value = self.config["training"]["grad_clipping_value"]

    def configure_optimizer(self):
        training_cfg = self.config["training"]
        optimizer_cfg = training_cfg["optimizer"]
        opt_class_name = optimizer_cfg["class_name"]
        opt_args = optimizer_cfg["args"]
        lr = float(training_cfg["lr"])
        OptimizerClass = getattr(optim, opt_class_name)
        self.optimizer = OptimizerClass(self.model.parameters(), lr=lr, **opt_args)

        # Load optimizer state if resuming from previous round
        if self.optim_resume == "resume" and self.optimizer_state:
            self.optimizer.load_state_dict(self.optimizer_state)

        self.scheduler = build_scheduler(self.optimizer, training_cfg, max_steps=self.max_steps)
        
        if self.scheduler_state:
            self.scheduler.load_state_dict(self.scheduler_state)
    
    def _get_batch_iterator(self, dataloader: DataLoader) -> Iterator:
        return cycle_batches(dataloader)


    def train_one_round(
        self,
        train_loader: DataLoader,
        metric_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        method_hook: Any = None,
        method_ctx: Any = None,
    ) -> Dict[str, float]:
        self.model.to(self.compute_device)
        self.model.train()
        if self.steps_or_epochs == "epochs":
            num_steps = self.config["training"]["local_steps"] * len(train_loader)
            self.max_steps = self.config["fed"]["rounds"] * num_steps
        else:
            num_steps = self.config["training"]["local_steps"]

        if num_steps <= 0:
            metrics = {
                "round": float(self.current_round),
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "num_steps_current_round": 0.0,
            }
            if metric_callback:
                metric_callback(metrics)
            return metrics

        self.configure_optimizer()

        batch_iterator = self._get_batch_iterator(train_loader)


        round_loss = 0.0
        correct = 0
        total = 0
        executed_steps = 0
        for _ in range(num_steps):
            images, labels = next(batch_iterator)

            images, labels = images.to(self.compute_device), labels.to(self.compute_device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.amp_enabled, device_type=self.compute_device.type):
                outputs = self.model(images)
                base_loss = self.criterion(outputs, labels)
                loss = (
                    method_hook.augment_loss(
                        base_loss=base_loss,
                        model=self.model,
                        outputs=outputs,
                        batch=(images, labels),
                        ctx=method_ctx,
                    )
                    if method_hook is not None
                    else base_loss
                )
            self.scaler.scale(loss).backward()

            # Gradient Clipping
            if self.grad_clipping_type == "norm":
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping_value)
            elif self.grad_clipping_type == "value":
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clipping_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if method_hook is not None:
                method_hook.on_step_end(model=self.model, optimizer=self.optimizer, ctx=method_ctx)
            self.scheduler.step()

            round_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            self.current_global_step += 1
            executed_steps += 1


        
        # Logging metrics via callback
        avg_loss = round_loss / num_steps if num_steps > 0 else 0.0
        training_accuracy = correct / total if total > 0 else 0.0
        metrics = {
            "round": float(self.current_round),
            "train_loss": avg_loss,
            "train_accuracy": training_accuracy,
            "num_steps_current_round": float(executed_steps),
        }
        if metric_callback:
            metric_callback(metrics)

        # OLD WAY
        # self.logger.add_scalar(
        #     "Loss/Round Loss",
        #     avg_loss / num_steps,
        #     self.current_step / self.gradient_accumulation_steps,
        # )


        # Save optimizer and scheduler states for next round
        self.model.to(self.cpu_device)
        move_optimizer_state(self.optimizer, self.cpu_device)
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optim_resume == "resume" else None
        self.scheduler_state = deepcopy(self.scheduler.state_dict())
        self.optimizer = None
        self.scheduler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Increment round counter
        self.current_round += 1
        return metrics

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        if val_loader is None:
            return {}

        self.model.to(self.compute_device)
        self.model.eval()

        correct = 0
        total = 0
        loss_sum = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.compute_device), labels.to(self.compute_device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / len(val_loader) if len(val_loader) > 0 else 0.0

        self.model.to(self.cpu_device)

        return {"accuracy": accuracy, "val_loss": avg_loss}
