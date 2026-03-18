from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from data_loading.dataset import BaseDataset
from fl_training.engines.classification import ClassificationEngine
from fl_training.engines.detection import DetectionEngine
from fl_training.engines.segmentation import SegmentationEngine
from fl_training.models.factory import create_model

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SliceSpec:
    numerator: int
    denominator: int

    def label(self) -> str:
        if self.numerator == self.denominator:
            return "full"
        return f"{self.numerator}of{self.denominator}"

    def select_count(self, total: int) -> int:
        if total <= 0:
            return 0
        if self.numerator >= self.denominator:
            return total
        return max(1, (total * self.numerator) // self.denominator)


@dataclass(frozen=True)
class TuningPreset:
    key: str
    dataset_name: str
    task_type: str
    num_classes: int
    datamodule_class_path: str
    objective_metric: str
    direction: str
    slice_spec: SliceSpec
    default_max_epochs: int
    default_image_size: int | None
    model_cfg: dict[str, Any]
    training_defaults: dict[str, Any] = field(default_factory=dict)
    datamodule_defaults: dict[str, Any] = field(default_factory=dict)
    batch_size_choices: tuple[int, ...] = field(default_factory=tuple)
    optimizer_choices: tuple[str, ...] = field(default_factory=tuple)
    scheduler_choices: tuple[str, ...] = field(default_factory=tuple)
    tune_pretrained: bool = False

    def default_study_name(self) -> str:
        return f"{self.dataset_name}_{self.model_cfg['name']}_optuna"


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Optuna is required for tuning. Install the project with the FL extras or add 'optuna'."
        ) from exc
    return optuna


def _import_object(class_path: str):
    module_path, object_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, object_name)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    return torch.device(device_arg)


def _make_sample(task_type: str, image_path: str, label_value: str) -> dict[str, Any]:
    if task_type == "segmentation":
        return {"image": image_path, "mask_path": label_value, "label": label_value}
    return {"image": image_path, "label": label_value}


def _load_raw_samples(dataset_name: str, base_path: Path, split: str, task_type: str) -> list[dict[str, Any]]:
    dataset = BaseDataset(dataset_name, str(base_path), split)
    return [_make_sample(task_type, image_path, label_value) for image_path, label_value in dataset]


def build_slice_manifest(preset: TuningPreset, *, base_path: Path, output_path: Path, seed: int) -> Path:
    if output_path.exists():
        return output_path

    train_samples = _load_raw_samples(preset.dataset_name, base_path, "train", preset.task_type)
    val_samples = _load_raw_samples(preset.dataset_name, base_path, "val", preset.task_type)

    select_count = preset.slice_spec.select_count(len(train_samples))
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(train_samples)), select_count)) if select_count else []
    selected_train = [train_samples[idx] for idx in selected_indices]

    payload = {
        "metadata": {
            "dataset": preset.dataset_name,
            "task_type": preset.task_type,
            "slice": preset.slice_spec.label(),
            "slice_seed": int(seed),
            "train_selected": len(selected_train),
            "train_total": len(train_samples),
            "val_total": len(val_samples),
        },
        "clients": {
            "client_0": {
                "train": selected_train,
                "val": [],
            }
        },
        "global_val": val_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def _build_datamodule(
    preset: TuningPreset,
    *,
    base_path: Path,
    split_file: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    datamodule_cls = _import_object(preset.datamodule_class_path)
    kwargs = {
        "data_dir": str(base_path),
        "split_file": str(split_file),
        "client_idx": 0,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "seed": int(seed),
    }
    if preset.default_image_size is not None:
        kwargs["image_size"] = int(preset.default_image_size)
    kwargs.update(preset.datamodule_defaults)
    return datamodule_cls(**kwargs)


def _make_optimizer_config(trial, optimizer_name: str, weight_decay: float) -> dict[str, Any]:
    if optimizer_name == "SGD":
        return {
            "class_name": "SGD",
            "args": {
                "momentum": trial.suggest_float("momentum", 0.8, 0.99),
                "weight_decay": weight_decay,
            },
        }

    return {
        "class_name": optimizer_name,
        "args": {
            "weight_decay": weight_decay,
        },
    }


def _suggest_hparams(trial, preset: TuningPreset) -> tuple[dict[str, Any], dict[str, Any], int]:
    task_type = preset.task_type
    optimizer_name = trial.suggest_categorical("optimizer", list(preset.optimizer_choices))
    scheduler_name = trial.suggest_categorical("scheduler", list(preset.scheduler_choices))
    batch_size = int(trial.suggest_categorical("batch_size", list(preset.batch_size_choices)))

    if task_type == "classification":
        lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-2, log=True)
    elif task_type == "detection":
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)

    training_cfg = {
        "lr": float(lr),
        "batch_size": batch_size,
        "local_steps": 1,
        "steps_or_epochs": "epochs",
        "scheduler": scheduler_name,
        "optim_resume": "resume",
        "lr_min": float(max(lr * 1e-2, 1e-7)) if scheduler_name == "cosine" else 0.0,
        "grad_clipping_type": "none",
        "grad_clipping_value": 0.0,
        "amp_enabled": torch.cuda.is_available(),
        "grad_tracking": False,
        "optimizer": _make_optimizer_config(trial, optimizer_name, float(weight_decay)),
    }
    training_cfg.update(preset.training_defaults)

    model_cfg = {
        "name": str(preset.model_cfg["name"]),
        "provider": str(preset.model_cfg["provider"]),
        "args": dict(preset.model_cfg.get("args", {})),
    }
    if preset.tune_pretrained:
        model_cfg["args"]["pretrained"] = bool(trial.suggest_categorical("pretrained", [True, False]))

    return training_cfg, model_cfg, batch_size


def _build_runtime_config(
    preset: TuningPreset,
    *,
    max_epochs: int,
    training_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": 0,
        "dataset": {
            "name": preset.dataset_name,
            "class_path": preset.datamodule_class_path,
        },
        "fed": {
            "rounds": int(max_epochs),
        },
        "training": {
            **training_cfg,
            "task_type": preset.task_type,
            "num_classes": int(preset.num_classes),
        },
        "model": model_cfg,
    }


def _make_engine(task_type: str, model, config: dict[str, Any], device: torch.device):
    if task_type == "classification":
        return ClassificationEngine(model=model, config=config, device=device)
    if task_type == "segmentation":
        return SegmentationEngine(model=model, config=config, device=device)
    if task_type == "detection":
        return DetectionEngine(model=model, config=config, device=device)
    raise ValueError(f"Unsupported task type '{task_type}'.")


def _extract_objective(metric_name: str, metrics: dict[str, Any]) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Objective metric '{metric_name}' not present in validation metrics: {sorted(metrics)}")
    return float(metrics[metric_name])


def _cleanup_trial(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_trial(
    trial,
    *,
    preset: TuningPreset,
    base_path: Path,
    split_file: Path,
    max_epochs: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> float:
    trial_seed = int(seed + trial.number)
    _set_seed(trial_seed)

    training_cfg, model_cfg, batch_size = _suggest_hparams(trial, preset)
    config = _build_runtime_config(
        preset,
        max_epochs=max_epochs,
        training_cfg=training_cfg,
        model_cfg=model_cfg,
    )
    config["seed"] = trial_seed

    datamodule = _build_datamodule(
        preset,
        base_path=base_path,
        split_file=split_file,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=trial_seed,
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader
    val_loader = datamodule.val_dataloader
    if val_loader is None:
        raise RuntimeError("Validation dataloader is required for Optuna tuning.")

    model = create_model(model_cfg=model_cfg, task_type=preset.task_type, num_classes=preset.num_classes)
    engine = _make_engine(preset.task_type, model, config, device)

    best_value = None
    best_epoch = None
    last_metrics: dict[str, Any] = {}
    try:
        for epoch_idx in range(int(max_epochs)):
            engine.train_one_round(train_loader)
            val_metrics = engine.evaluate(val_loader)
            current_value = _extract_objective(preset.objective_metric, val_metrics)
            last_metrics = dict(val_metrics)
            if best_value is None or current_value > best_value:
                best_value = current_value
                best_epoch = epoch_idx + 1

            trial.report(current_value, step=epoch_idx + 1)
            if trial.should_prune():
                trial.set_user_attr("best_epoch", best_epoch)
                trial.set_user_attr("best_value", best_value)
                trial.set_user_attr("last_metrics", last_metrics)
                raise _require_optuna().TrialPruned()

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_value", best_value)
        trial.set_user_attr("last_metrics", last_metrics)
        return float(best_value if best_value is not None else float("-inf"))
    finally:
        datamodule.teardown()
        _cleanup_trial(engine, model, datamodule, train_loader, val_loader)


def _write_trials_csv(study, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "number",
                "state",
                "value",
                "duration_seconds",
                "params_json",
                "user_attrs_json",
            ],
        )
        writer.writeheader()
        for trial in study.trials:
            duration_seconds = None
            if trial.datetime_start and trial.datetime_complete:
                duration_seconds = (trial.datetime_complete - trial.datetime_start).total_seconds()
            writer.writerow(
                {
                    "number": trial.number,
                    "state": str(trial.state),
                    "value": trial.value,
                    "duration_seconds": duration_seconds,
                    "params_json": json.dumps(trial.params, sort_keys=True),
                    "user_attrs_json": json.dumps(trial.user_attrs, sort_keys=True, default=str),
                }
            )


def _write_best_params_json(study, preset: TuningPreset, output_path: Path) -> None:
    payload: dict[str, Any] = {
        "preset": preset.key,
        "dataset": preset.dataset_name,
        "model": preset.model_cfg,
        "objective_metric": preset.objective_metric,
        "direction": preset.direction,
    }
    try:
        best_trial = study.best_trial
    except ValueError:
        payload["best_trial"] = None
    else:
        payload["best_trial"] = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_summary_markdown(
    study,
    preset: TuningPreset,
    *,
    study_name: str,
    output_path: Path,
    manifest_path: Path,
    max_epochs: int,
    n_trials: int,
    timeout: int | None,
) -> None:
    lines = [
        f"# Optuna Summary: {study_name}",
        "",
        f"- Preset: `{preset.key}`",
        f"- Dataset: `{preset.dataset_name}`",
        f"- Task: `{preset.task_type}`",
        f"- Model: `{preset.model_cfg['provider']}::{preset.model_cfg['name']}`",
        f"- Objective: `{preset.objective_metric}` ({preset.direction})",
        f"- Slice Manifest: `{manifest_path}`",
        f"- Max Epochs: `{max_epochs}`",
        f"- Requested Trials: `{n_trials}`",
        f"- Timeout Seconds: `{timeout}`",
        f"- Completed Trials: `{len(study.trials)}`",
        "",
    ]

    try:
        best_trial = study.best_trial
    except ValueError:
        lines.extend([
            "## Best Trial",
            "",
            "No completed trials.",
        ])
    else:
        lines.extend([
            "## Best Trial",
            "",
            f"- Trial Number: `{best_trial.number}`",
            f"- Value: `{best_trial.value}`",
            f"- Best Epoch: `{best_trial.user_attrs.get('best_epoch')}`",
            f"- Params: `{json.dumps(best_trial.params, sort_keys=True)}`",
            f"- Last Metrics: `{json.dumps(best_trial.user_attrs.get('last_metrics', {}), sort_keys=True)}`",
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser(preset: TuningPreset) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Optuna local-proxy tuning for {preset.dataset_name} / {preset.model_cfg['name']}"
    )
    parser.add_argument("--base_path", required=True, type=str, help="Raw dataset base path.")
    parser.add_argument("--study_dir", required=True, type=str, help="Output directory for study DB and artifacts.")
    parser.add_argument("--study_name", default=None, type=str, help="Optuna study name override.")
    parser.add_argument("--n_trials", default=100, type=int, help="Number of Optuna trials to run.")
    parser.add_argument("--timeout", default=None, type=int, help="Optional walltime limit in seconds.")
    parser.add_argument("--max_epochs", default=preset.default_max_epochs, type=int, help="Epoch budget per trial.")
    parser.add_argument("--device", default="auto", type=str, help="torch device string or 'auto'.")
    parser.add_argument("--num_workers", default=4, type=int, help="Dataloader worker count.")
    parser.add_argument("--seed", default=42, type=int, help="Base seed for slice selection and trial seeding.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    return parser


def run_preset_cli(preset: TuningPreset) -> int:
    args = build_arg_parser(preset).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    base_path = Path(args.base_path).expanduser().resolve()
    study_dir = Path(args.study_dir).expanduser().resolve()
    study_name = args.study_name or preset.default_study_name()
    study_root = study_dir / study_name
    study_root.mkdir(parents=True, exist_ok=True)

    manifest_path = build_slice_manifest(
        preset,
        base_path=base_path,
        output_path=study_dir / "slices" / f"{preset.dataset_name}_{preset.slice_spec.label()}_seed{args.seed}.json",
        seed=int(args.seed),
    )

    resolved_device = _resolve_device(args.device)
    db_path = study_root / "study.sqlite3"
    trials_csv_path = study_root / "trials.csv"
    best_params_path = study_root / "best_params.json"
    summary_path = study_root / "summary.md"

    if args.dry_run:
        log.info("Preset: %s", preset.key)
        log.info("Dataset base path: %s", base_path)
        log.info("Study root: %s", study_root)
        log.info("Slice manifest: %s", manifest_path)
        log.info("Study DB: %s", db_path)
        log.info("Objective: %s (%s)", preset.objective_metric, preset.direction)
        log.info("Device: %s", resolved_device)
        return 0

    optuna = _require_optuna()
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(1, int(args.max_epochs) // 4))
    study = optuna.create_study(
        study_name=study_name,
        direction=preset.direction,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    log.info(
        "Starting Optuna study '%s' | dataset=%s model=%s trials=%s max_epochs=%s device=%s",
        study_name,
        preset.dataset_name,
        preset.model_cfg["name"],
        args.n_trials,
        args.max_epochs,
        resolved_device,
    )

    objective = lambda trial: _run_trial(
        trial,
        preset=preset,
        base_path=base_path,
        split_file=manifest_path,
        max_epochs=int(args.max_epochs),
        num_workers=int(args.num_workers),
        device=resolved_device,
        seed=int(args.seed),
    )
    study.optimize(objective, n_trials=int(args.n_trials), timeout=args.timeout, gc_after_trial=True)

    _write_trials_csv(study, trials_csv_path)
    _write_best_params_json(study, preset, best_params_path)
    _write_summary_markdown(
        study,
        preset,
        study_name=study_name,
        output_path=summary_path,
        manifest_path=manifest_path,
        max_epochs=int(args.max_epochs),
        n_trials=int(args.n_trials),
        timeout=args.timeout,
    )

    log.info("Study complete. Artifacts written to %s", study_root)
    return 0
