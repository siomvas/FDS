#!/usr/bin/env python3
"""Preflight validation for FL training configs and split files.

This script performs fail-fast checks before launching FL training:
- config structure (dataset/training/model/fed)
- model provider-task contract validation
- datamodule wiring (class import + constructor args)
- split file schema validation
- task-specific sample contract checks
- optional filesystem path existence checks for sample files
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fl_training.models.factory import validate_model_contract

log = logging.getLogger(__name__)

_DETECTION_ANNOTATION_KEYS = (
    "annotation",
    "annotation_path",
    "label_path",
    "bbox_path",
    "target_path",
    "label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FL config and split files before launch.")

    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to an explicit YAML/JSON config. If omitted, Hydra compose mode is used.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "fl_training" / "conf"),
        help="Hydra config directory (used when --config-file is omitted).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name (used when --config-file is omitted).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override. Repeatable, e.g. --override fed.rounds=1",
    )
    parser.add_argument(
        "--client-idx",
        type=int,
        default=0,
        help="Client index used to validate per-client data access when central=False.",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Check that referenced image/mask/annotation paths exist on disk.",
    )
    parser.add_argument(
        "--max-path-check",
        type=int,
        default=200,
        help="Max samples checked for path existence per split side; -1 means all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary only (default behavior is already non-mutating).",
    )

    return parser.parse_args()


def _load_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config_file:
        cfg: DictConfig = OmegaConf.load(args.config_file)
    else:
        config_dir = str(Path(args.config_dir).resolve())
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name=args.config_name, overrides=list(args.override or []))

    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise ValueError(f"Resolved config must be a mapping, got {type(container).__name__}.")
    return container


def _validate_cfg_structure(cfg: Dict[str, Any]) -> None:
    required_sections = ("dataset", "training", "model", "fed")
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"Config is missing required sections: {missing}")

    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    fed_cfg = cfg["fed"]

    if not isinstance(dataset_cfg, dict):
        raise ValueError("Config 'dataset' section must be a mapping.")
    if not isinstance(training_cfg, dict):
        raise ValueError("Config 'training' section must be a mapping.")
    if not isinstance(model_cfg, dict):
        raise ValueError("Config 'model' section must be a mapping.")
    if not isinstance(fed_cfg, dict):
        raise ValueError("Config 'fed' section must be a mapping.")

    if not isinstance(dataset_cfg.get("class_path"), str) or not dataset_cfg["class_path"].strip():
        raise ValueError("dataset.class_path must be a non-empty string.")

    if "task_type" not in training_cfg:
        raise ValueError("training.task_type is required.")
    if "batch_size" not in training_cfg:
        raise ValueError("training.batch_size is required.")
    if "num_classes" not in training_cfg:
        raise ValueError("training.num_classes is required.")
    optimizer_cfg = training_cfg.get("optimizer")
    if not isinstance(optimizer_cfg, dict):
        raise ValueError("training.optimizer must be a mapping.")
    if not isinstance(optimizer_cfg.get("class_name"), str) or not optimizer_cfg["class_name"].strip():
        raise ValueError("training.optimizer.class_name must be a non-empty string.")
    if not isinstance(optimizer_cfg.get("args"), dict):
        raise ValueError("training.optimizer.args must be a mapping.")

    if "rounds" not in fed_cfg:
        raise ValueError("fed.rounds is required.")
    if "seed" not in cfg:
        raise ValueError("seed is required.")


def _load_datamodule_class(class_path: str):
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid class path '{class_path}'. Expected module.ClassName") from exc

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def _build_datamodule_args(cfg: Dict[str, Any], client_idx: int) -> Dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    dataset_args = dataset_cfg.get("args", {})
    if not isinstance(dataset_args, dict):
        raise ValueError("dataset.args must be a mapping if provided.")
    dm_args = dict(dataset_args)

    if "root" in dm_args and "data_dir" not in dm_args:
        dm_args["data_dir"] = dm_args.pop("root")

    if "dataset_splits" in dataset_cfg and "split_file" not in dm_args:
        dm_args["split_file"] = dataset_cfg["dataset_splits"]

    if "label_map" in dataset_cfg and "label_map" not in dm_args:
        dm_args["label_map"] = dataset_cfg["label_map"]

    if "split_file" not in dm_args:
        raise ValueError(
            "Missing split file for datamodule. Provide dataset.dataset_splits or dataset.args.split_file."
        )
    if "data_dir" not in dm_args:
        raise ValueError("Missing datamodule data_dir. Provide dataset.args.data_dir (or legacy root).")

    dm_args.setdefault("client_idx", client_idx)
    dm_args.setdefault("seed", int(cfg["seed"]))
    dm_args.setdefault("batch_size", int(training_cfg["batch_size"]))
    dm_args.setdefault("num_workers", 0)

    return dm_args


def _coerce_client_idx_for_available_splits(dm: Any, splits: Dict[str, Any]) -> None:
    if bool(getattr(dm, "central", False)):
        return

    clients = splits.get("clients", {})
    if not clients:
        return

    requested = f"client_{getattr(dm, 'client_idx', 0)}"
    if requested in clients:
        return

    raise ValueError(
        f"Requested client '{requested}' is missing from split payload. "
        "Fix the client index or the split file instead of relying on fallback selection."
    )


def _iter_sample_groups(train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]]) -> Iterable[Tuple[str, List[Dict[str, Any]]]]:
    yield "train", train_data
    yield "val", val_data


def _validate_task_samples(task_type: str, train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]]) -> None:
    if not train_data:
        raise ValueError("No training samples found for preflight check.")

    for group_name, samples in _iter_sample_groups(train_data, val_data):
        for idx, sample in enumerate(samples):
            ref = f"{group_name}[{idx}]"
            if task_type == "classification":
                if "label" not in sample:
                    raise KeyError(f"Classification sample missing 'label' at {ref}.")
            elif task_type == "segmentation":
                if "mask_path" not in sample and "label" not in sample:
                    raise KeyError(
                        "Segmentation sample must include 'mask_path' or legacy 'label' "
                        f"at {ref}."
                    )
            elif task_type == "detection":
                has_inline = "boxes" in sample and "labels" in sample
                has_ann = any(key in sample for key in _DETECTION_ANNOTATION_KEYS)
                if not (has_inline or has_ann):
                    raise KeyError(
                        "Detection sample must include inline 'boxes'+'labels' or an annotation path "
                        f"key at {ref}."
                    )
            else:
                raise ValueError(f"Unsupported task_type '{task_type}'.")


def _resolve_image_path(image: str, data_dir: Path) -> Path:
    path = Path(image)
    if path.is_absolute():
        return path
    return data_dir / path


def _resolve_annotation_path(sample: Dict[str, Any], image_path: Path) -> Path | None:
    for key in _DETECTION_ANNOTATION_KEYS:
        raw = sample.get(key)
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw)
            if candidate.is_absolute():
                return candidate
            return (image_path.parent / candidate).resolve()
    return None


def _check_paths(
    task_type: str,
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    data_dir: Path,
    max_path_check: int,
) -> None:
    checked = 0

    for group_name, samples in _iter_sample_groups(train_data, val_data):
        limit = len(samples) if max_path_check < 0 else min(len(samples), max_path_check)
        for idx, sample in enumerate(samples[:limit]):
            ref = f"{group_name}[{idx}]"
            image_path = _resolve_image_path(str(sample["image"]), data_dir)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file does not exist at {ref}: {image_path}")
            checked += 1

            if task_type == "segmentation":
                mask_raw = sample.get("mask_path", sample.get("label"))
                if not isinstance(mask_raw, str):
                    raise ValueError(f"Segmentation mask path is invalid at {ref}: {mask_raw!r}")
                mask_path = _resolve_image_path(mask_raw, data_dir)
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask file does not exist at {ref}: {mask_path}")

            if task_type == "detection" and not ("boxes" in sample and "labels" in sample):
                ann_path = _resolve_annotation_path(sample, image_path=image_path)
                if ann_path is None:
                    raise FileNotFoundError(f"Detection annotation path missing at {ref}.")
                if not ann_path.exists():
                    raise FileNotFoundError(f"Detection annotation file does not exist at {ref}: {ann_path}")

    log.info("Path check completed for %d samples.", checked)


def run_preflight(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    _validate_cfg_structure(cfg)

    task_type = str(cfg["training"]["task_type"]).lower()
    num_classes = int(cfg["training"]["num_classes"])

    validate_model_contract(cfg["model"], task_type=task_type, num_classes=num_classes)

    dm_class_path = str(cfg["dataset"]["class_path"])
    dm_cls = _load_datamodule_class(dm_class_path)

    dm_args = _build_datamodule_args(cfg, client_idx=int(args.client_idx))
    dm = dm_cls(**dm_args)

    splits = dm.load_splits()
    _coerce_client_idx_for_available_splits(dm, splits)
    train_data, val_data = dm.get_client_data(splits)

    _validate_task_samples(task_type, train_data, val_data)

    if args.check_paths:
        _check_paths(
            task_type=task_type,
            train_data=train_data,
            val_data=val_data,
            data_dir=Path(str(dm_args["data_dir"])),
            max_path_check=int(args.max_path_check),
        )

    n_clients = len(splits["clients"])
    n_train = len(train_data)
    n_val = len(val_data)
    split_file = dm_args["split_file"]

    log.info("Preflight OK")
    log.info(
        "task=%s num_classes=%d provider=%s model=%s",
        task_type,
        num_classes,
        cfg["model"]["provider"],
        cfg["model"]["name"],
    )
    log.info("datamodule=%s split_file=%s n_clients=%d train=%d val=%d", dm_class_path, split_file, n_clients, n_train, n_val)

    if args.dry_run:
        log.info("Dry-run mode enabled (no mutations performed).")

    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    try:
        return run_preflight(args)
    except (ValueError, KeyError, TypeError, FileNotFoundError, ImportError, AttributeError) as exc:
        log.error("Preflight failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
