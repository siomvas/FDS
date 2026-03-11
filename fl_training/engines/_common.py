from collections.abc import Iterator
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def cycle_batches(dataloader: DataLoader) -> Iterator[Any]:
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            try:
                yield next(iterator)
            except StopIteration as exc:
                raise ValueError("Dataloader is empty.") from exc


def build_scheduler(
    optimizer: optim.Optimizer,
    training_cfg: dict[str, Any],
    *,
    max_steps: int,
) -> optim.lr_scheduler.LRScheduler:
    scheduler_name = str(training_cfg["scheduler"]).lower()
    lr = float(training_cfg["lr"])
    lr_min = float(training_cfg.get("lr_min", 0.0))

    if scheduler_name == "none":
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=max_steps)
    if scheduler_name == "step":
        if "lr_step_size" not in training_cfg:
            raise ValueError("training.lr_step_size is required when training.scheduler='step'.")
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training_cfg["lr_step_size"]),
            gamma=0.1,
        )
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr_min)
    if scheduler_name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=max_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )
    raise ValueError(
        f"Unknown scheduler '{scheduler_name}'. "
        "Supported values: ['none', 'step', 'cosine', 'onecycle']."
    )


def move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
