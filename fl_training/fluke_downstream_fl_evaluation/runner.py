"""Configuration, preflight, execution, and artifact writing for Fluke runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from types import MethodType

import yaml

from research.io import create_manifest, create_result, write_manifest, write_result

from .catalog import AlgorithmTemplate, resolve_algorithm_template
from .data import (
    PartitionDataSplitter,
    TinyImageNetPartition,
    build_tiny_imagenet_transforms,
    load_tiny_imagenet_partition,
)
from .fluke_compat import (
    compatibility_patches,
    configure_scaffold_local_steps,
    install_fluke_compatibility,
)
from .models import build_tiny_imagenet_model_bundle

LOG = logging.getLogger(__name__)


class FlukeEvaluationError(RuntimeError):
    """Configuration or runtime failure that should be persisted as a run artifact."""


def _require_fluke() -> None:
    try:
        import fluke  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency behavior
        raise FlukeEvaluationError(
            "Fluke downstream evaluation requires `pip install -e '.[fluke]'`."
        ) from exc


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if "$" in expanded:
            raise FlukeEvaluationError(f"Unresolved environment variable in configuration value: {value}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load one YAML config and expand environment variables recursively."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Fluke evaluation config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Fluke evaluation YAML root must be a mapping.")
    config = _expand_environment(payload)
    config["_config_path"] = str(config_path)
    return config


def _as_list(value: Any, *, field: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list.")
    return value


def resolve_matrix_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand partition, algorithm, and local-epoch choices into atomic run configs."""

    partitions = _as_list(config.get("partitions"), field="partitions")
    if not partitions:
        raise FlukeEvaluationError("Configuration requires at least one partition entry.")
    algorithms = _as_list(config.get("algorithms"), field="algorithms")
    if not algorithms:
        algorithm = config.get("algorithm")
        algorithms = [algorithm] if algorithm else []
    if not algorithms:
        raise FlukeEvaluationError("Configuration requires algorithms or algorithm.")

    protocol = config.get("protocol", {})
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping.")
    matrix = config.get("matrix", {})
    if matrix and not isinstance(matrix, Mapping):
        raise TypeError("matrix must be a mapping when provided.")
    local_epochs = list(matrix.get("local_epochs", [protocol.get("local_epochs", 1)]))
    if not local_epochs:
        raise FlukeEvaluationError("matrix.local_epochs must contain at least one value.")

    cells: list[dict[str, Any]] = []
    for partition in partitions:
        if not isinstance(partition, Mapping):
            raise TypeError("Each partition entry must be a mapping.")
        if not partition.get("path"):
            raise FlukeEvaluationError("Each partition entry requires path.")
        for algorithm in algorithms:
            algorithm_payload = (
                {"key": algorithm} if isinstance(algorithm, str) else dict(algorithm or {})
            )
            if not algorithm_payload:
                raise FlukeEvaluationError("Each algorithm entry must be a catalog key or mapping.")
            for epochs in local_epochs:
                if int(epochs) < 1:
                    raise FlukeEvaluationError("local_epochs must be >= 1.")
                cell = copy.deepcopy(dict(config))
                cell.pop("partitions", None)
                cell.pop("algorithms", None)
                cell.pop("matrix", None)
                cell["partition"] = copy.deepcopy(dict(partition))
                cell["algorithm"] = algorithm_payload
                cell.setdefault("protocol", {})["local_epochs"] = int(epochs)
                cells.append(cell)
    return cells


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _config_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _import_algorithm_class(class_path: str) -> type:
    module_name, _, name = class_path.rpartition(".")
    if not module_name or not name:
        raise FlukeEvaluationError(f"Invalid algorithm class path: {class_path!r}")
    try:
        target = getattr(import_module(module_name), name)
    except (ImportError, AttributeError) as exc:
        raise FlukeEvaluationError(f"Could not import Fluke algorithm {class_path}.") from exc
    if not isinstance(target, type):
        raise FlukeEvaluationError(f"Fluke algorithm target is not a class: {class_path}")
    return target


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish live run state without relying on buffered stdout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass
class RoundMetricsObserver:
    """Record Fluke events and publish durable progress for long-running jobs."""

    global_metrics: list[dict[str, Any]]
    local_metrics: list[dict[str, Any]]
    client_metrics: list[dict[str, Any]]
    method_diagnostics: list[dict[str, Any]]

    def __init__(
        self,
        *,
        progress_path: Path | None = None,
        total_rounds: int | None = None,
        checkpoint_dir: Path | None = None,
        global_model_supplier: Callable[[], Any] | None = None,
        round_diagnostics_supplier: Callable[[], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self.global_metrics = []
        self.local_metrics = []
        self.client_metrics = []
        self.method_diagnostics = []
        self._progress_path = progress_path
        self._total_rounds = total_rounds
        self._started_at = time.monotonic()
        self._active_round: int | None = None
        self._active_client: int | None = None
        self._round_started_at: float | None = None
        self._completed_rounds = 0
        self._completed_clients = 0
        self._round_durations: list[float] = []
        self._last_global_metrics: dict[str, Any] = {}
        self._checkpoint_dir = checkpoint_dir
        self._global_model_supplier = global_model_supplier
        self._round_diagnostics_supplier = round_diagnostics_supplier
        self._global_model: Any | None = None
        self._best_accuracy: float | None = None
        self._best_checkpoint: dict[str, Any] | None = None
        self._publish_progress("running")

    @staticmethod
    def _checkpoint_payload(model: Any, *, round: int, accuracy: float | None) -> dict[str, Any]:
        """Copy a model state to CPU so the best round survives later training."""

        import torch

        return {
            "schema_version": "1",
            "round": int(round),
            "accuracy": accuracy,
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
        }

    @staticmethod
    def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
        """Write a checkpoint atomically so an interrupted run leaves no partial model."""

        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)

    def finalize_checkpoints(self, *, round: int) -> dict[str, str]:
        """Persist the final and best global models after a successful run."""

        model = (
            self._global_model_supplier()
            if self._global_model_supplier is not None
            else self._global_model
        )
        if self._checkpoint_dir is None or model is None:
            return {}
        last_path = self._checkpoint_dir / "last.pt"
        best_path = self._checkpoint_dir / "best.pt"
        last_accuracy = _safe_float(self._last_global_metrics.get("accuracy"))
        self._write_checkpoint(
            last_path,
            self._checkpoint_payload(
                model,
                round=round,
                accuracy=last_accuracy,
            ),
        )
        if self._best_checkpoint is not None:
            self._write_checkpoint(best_path, self._best_checkpoint)
        return {
            "last_model_checkpoint": str(last_path),
            **(
                {"best_model_checkpoint": str(best_path)}
                if self._best_checkpoint is not None
                else {}
            ),
        }

    def _publish_progress(self, status: str) -> None:
        if self._progress_path is None:
            return
        elapsed = time.monotonic() - self._started_at
        mean_round_seconds = (
            sum(self._round_durations) / len(self._round_durations)
            if self._round_durations
            else None
        )
        estimated_remaining_seconds = (
            mean_round_seconds * max(self._total_rounds - self._completed_rounds, 0)
            if mean_round_seconds is not None and self._total_rounds is not None
            else None
        )
        _write_progress(
            self._progress_path,
            {
                "schema_version": "1",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "total_rounds": self._total_rounds,
                "rounds_completed": self._completed_rounds,
                "active_round": self._active_round,
                "active_client": self._active_client,
                "clients_completed_in_active_round": self._completed_clients,
                "elapsed_seconds": elapsed,
                "last_round_seconds": self._round_durations[-1] if self._round_durations else None,
                "mean_round_seconds": mean_round_seconds,
                "estimated_remaining_seconds": estimated_remaining_seconds,
                "last_global_metrics": self._last_global_metrics,
                "last_method_diagnostics": (
                    self.method_diagnostics[-1]["metrics"] if self.method_diagnostics else {}
                ),
            },
        )

    def server_evaluation(self, round: int, eval_type: str, evals: Any, **_: Any) -> None:
        entry = {"round": int(round), "metrics": dict(evals) if isinstance(evals, Mapping) else evals}
        if eval_type == "global":
            self.global_metrics.append(entry)
            self._last_global_metrics = dict(evals) if isinstance(evals, Mapping) else {}
            accuracy = _safe_float(self._last_global_metrics.get("accuracy"))
            model = (
                self._global_model_supplier()
                if self._global_model_supplier is not None
                else self._global_model
            )
            if (
                accuracy is not None
                and model is not None
                and (self._best_accuracy is None or accuracy > self._best_accuracy)
            ):
                self._best_accuracy = accuracy
                self._best_checkpoint = self._checkpoint_payload(
                    model,
                    round=int(round),
                    accuracy=accuracy,
                )
        else:
            self.local_metrics.append(entry)

    def client_evaluation(
        self,
        round: int,
        client_id: int,
        phase: str,
        evals: Mapping[str, Any] | None = None,
        results: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        # ParallelServer emits ``results`` while the centralized client observer
        # emits ``evals``. Retain one artifact schema across both Fluke protocols.
        metrics = evals if isinstance(evals, Mapping) else results
        self.client_metrics.append(
            {
                "round": int(round),
                "client_id": int(client_id),
                "phase": str(phase),
                "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
            }
        )

    # These methods also make the object compatible with Fluke's observer interfaces.
    def start_round(self, round: int, global_model: Any | None = None, **_: Any) -> None:
        self._active_round = int(round)
        if global_model is not None:
            self._global_model = global_model
        self._active_client = None
        self._completed_clients = 0
        self._round_started_at = time.monotonic()
        self._publish_progress("running")

    def end_round(self, round: int, **_: Any) -> None:
        if self._round_started_at is not None:
            self._round_durations.append(time.monotonic() - self._round_started_at)
        self._completed_rounds = max(self._completed_rounds, int(round))
        if self._round_diagnostics_supplier is not None:
            diagnostics = self._round_diagnostics_supplier()
            if diagnostics:
                self.method_diagnostics.append(
                    {"round": int(round), "metrics": dict(diagnostics)}
                )
        self._active_round = None
        self._active_client = None
        self._publish_progress("running")

    def finished(self, **_: Any) -> None:
        self._active_round = None
        self._active_client = None
        self._publish_progress("finished")

    def selected_clients(self, **_: Any) -> None:
        return None

    def start_fit(self, round: int, client_id: int, **_: Any) -> None:
        self._active_round = int(round)
        self._active_client = int(client_id)
        self._publish_progress("running")

    def end_fit(self, round: int, client_id: int, **_: Any) -> None:
        self._active_round = int(round)
        self._active_client = None
        self._completed_clients += 1
        self._publish_progress("running")


def _fluke_observer(
    *,
    progress_path: Path | None = None,
    total_rounds: int | None = None,
    checkpoint_dir: Path | None = None,
    global_model_supplier: Callable[[], Any] | None = None,
    round_diagnostics_supplier: Callable[[], Mapping[str, Any] | None] | None = None,
) -> Any:
    """Build an observer that is an instance of every relevant Fluke interface."""

    _require_fluke()
    from fluke.utils import ClientObserver, FederationObserver, ServerObserver

    class _Observer(RoundMetricsObserver, ClientObserver, ServerObserver, FederationObserver):
        pass

    return _Observer(
        progress_path=progress_path,
        total_rounds=total_rounds,
        checkpoint_dir=checkpoint_dir,
        global_model_supplier=global_model_supplier,
        round_diagnostics_supplier=round_diagnostics_supplier,
    )


def _template_from_cell(cell: Mapping[str, Any]) -> AlgorithmTemplate:
    algorithm = cell.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise FlukeEvaluationError("Resolved matrix cell requires an algorithm mapping.")
    return resolve_algorithm_template(algorithm)


def _algorithm_overrides(cell: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    algorithm = cell["algorithm"]
    if not isinstance(algorithm, Mapping):
        raise TypeError("algorithm must be a mapping.")
    client = algorithm.get("client", {})
    server = algorithm.get("server", {})
    model = algorithm.get("model", {})
    for name, value in (("algorithm.client", client), ("algorithm.server", server), ("algorithm.model", model)):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping.")
    return dict(client), dict(server), dict(model)


def _effective_model_settings(
    cell: Mapping[str, Any], template: AlgorithmTemplate
) -> tuple[int, str, str]:
    model_cfg = cell.get("model", {})
    if not isinstance(model_cfg, Mapping):
        raise TypeError("model must be a mapping.")
    _, _, model_override = _algorithm_overrides(cell)
    if "normalization" in model_override:
        normalization = str(model_override["normalization"])
    elif template.normalization != "group":
        # Some Fluke methods have a non-negotiable model contract: FedBN needs
        # batch statistics and SuPerFed cannot mix GroupNorm parameters.
        normalization = template.normalization
    else:
        normalization = str(model_cfg.get("normalization", template.normalization))
    if normalization not in {"group", "batch", "none"}:
        raise FlukeEvaluationError("model.normalization must be one of group, batch, none.")
    image_size = int(model_override.get("image_size", model_cfg.get("image_size", 64)))
    if image_size <= 0:
        raise FlukeEvaluationError("model.image_size must be > 0.")
    model_name = str(model_override.get("name", model_cfg.get("name", "tiny_encoder_head")))
    aliases = {
        "encoder_head": "tiny_encoder_head",
        "shallow_cnn": "tiny_encoder_head",
        "resnet18gn": "fluke_resnet18gn",
        "resnet18_pretrained_bn": "torchvision_resnet18_imagenet_bn",
        "resnet18_imagenet_bn": "torchvision_resnet18_imagenet_bn",
    }
    model_name = aliases.get(model_name.lower(), model_name.lower())
    if model_name not in {
        "tiny_encoder_head",
        "fluke_resnet18gn",
        "torchvision_resnet18_imagenet_bn",
    }:
        raise FlukeEvaluationError(
            "model.name must be tiny_encoder_head, fluke_resnet18gn, or "
            "torchvision_resnet18_imagenet_bn."
        )
    return image_size, normalization, model_name


def _validate_scaffold_option_ii_contract(
    template: AlgorithmTemplate, client: Mapping[str, Any], server: Mapping[str, Any]
) -> None:
    """Reject optimizer settings unsupported by the Option II compatibility patch."""

    if template.key != "scaffold":
        return
    optimizer = client.get("optimizer", {})
    scheduler = client.get("scheduler", {})
    if not isinstance(optimizer, Mapping) or not isinstance(scheduler, Mapping):
        raise FlukeEvaluationError("SCAFFOLD requires mapping-valued optimizer and scheduler settings.")
    optimizer_name = str(optimizer.get("name", "SGD")).lower()
    try:
        momentum = float(optimizer.get("momentum", 0.0))
        dampening = float(optimizer.get("dampening", 0.0))
    except (TypeError, ValueError) as exc:
        raise FlukeEvaluationError("SCAFFOLD SGD momentum and dampening must be numeric.") from exc
    if (
        optimizer_name != "sgd"
        or not 0.0 <= momentum < 1.0
        or dampening != 0.0
        or bool(optimizer.get("nesterov", False))
    ):
        raise FlukeEvaluationError(
            "Fluke SCAFFOLD Option II supports SGD with momentum in [0, 1), dampening=0, and "
            "nesterov=false. The adapter applies FedNova's scalar momentum normalization; Adam "
            "and AdamW remain unsupported because their adaptive per-parameter steps have no "
            "equivalent scalar."
        )
    scheduler_name = str(scheduler.get("name", "StepLR"))
    gamma = float(scheduler.get("gamma", 1.0))
    if scheduler_name != "StepLR" or gamma != 1.0:
        raise FlukeEvaluationError(
            "Fluke SCAFFOLD Option II requires a constant learning rate: use "
            "StepLR(step_size=1, gamma=1.0)."
        )
    if float(server.get("global_step", 1.0)) != 1.0:
        raise FlukeEvaluationError(
            "Fluke SCAFFOLD Option II requires server.global_step=1.0 because Fluke applies "
            "this value by scaling client aggregation weights rather than as an anchored server "
            "learning rate."
        )


def _build_hyperparameters(
    cell: Mapping[str, Any], template: AlgorithmTemplate, partition: TinyImageNetPartition
) -> tuple[Any, int, str, str]:
    """Build Fluke's DDict hyperparameters from generic and method-specific contracts."""

    _require_fluke()
    from fluke import DDict

    protocol = cell.get("protocol", {})
    training = cell.get("training", {})
    if not isinstance(protocol, Mapping) or not isinstance(training, Mapping):
        raise TypeError("protocol and training must be mappings.")

    client_override, server_override, _ = _algorithm_overrides(cell)
    image_size, normalization, model_name = _effective_model_settings(cell, template)

    configured_optimizer = training.get("optimizer", {})
    if not isinstance(configured_optimizer, Mapping):
        raise TypeError("training.optimizer must be a mapping.")
    optimizer_name = str(configured_optimizer.get("name", "SGD"))
    optimizer_defaults: dict[str, Any] = {
        "name": optimizer_name,
        "lr": 0.01,
        "weight_decay": 0.0,
    }
    if optimizer_name.lower() == "sgd":
        optimizer_defaults["momentum"] = 0.9
    optimizer = _deep_merge(
        optimizer_defaults,
        dict(configured_optimizer),
    )
    if optimizer_name.lower() in {"adam", "adamw"}:
        optimizer.pop("momentum", None)
    if "lr" in training:
        optimizer["lr"] = float(training["lr"])
    scheduler = _deep_merge(
        {"name": "StepLR", "step_size": 1, "gamma": 1.0},
        dict(training.get("scheduler", {})),
    )
    client = _deep_merge(
        {
            "batch_size": int(training.get("batch_size", 64)),
            "local_epochs": int(protocol.get("local_epochs", training.get("local_epochs", 1))),
            "loss": str(training.get("loss", "CrossEntropyLoss")),
            "optimizer": optimizer,
            "scheduler": scheduler,
            "fine_tuning_epochs": int(training.get("fine_tuning_epochs", 0)),
            "clipping": float(training.get("clipping", 0.0)),
        },
        template.client_defaults,
    )
    client = _deep_merge(client, client_override)
    if template.key == "per_fedavg":
        # Fluke replaces this with PerFedAVGOptimizer, whose constructor only
        # accepts ``params`` and ``lr`` rather than SGD's momentum/decay keys.
        client["optimizer"] = {
            "name": client["optimizer"].get("name", "SGD"),
            "lr": client["optimizer"].get("lr", 0.01),
        }
    elif template.key == "pfedme":
        # Fluke replaces this with PFedMeOptimizer. Keep its explicit native
        # controls while dropping unsupported SGD-only kwargs.
        client["optimizer"] = {
            "name": client["optimizer"].get("name", "SGD"),
            "lr": client["optimizer"].get("lr", 0.01),
            "lamda": client["optimizer"].get("lamda", 0.1),
            "mu": client["optimizer"].get("mu", 0.001),
        }
    server = _deep_merge({"weighted": True}, template.server_defaults)
    server = _deep_merge(server, server_override)
    _validate_scaffold_option_ii_contract(template, client, server)

    bundle = build_tiny_imagenet_model_bundle(
        num_classes=partition.num_classes,
        normalization=normalization,  # type: ignore[arg-type]
        name=model_name,  # type: ignore[arg-type]
    )
    if template.protocol == "parallel":
        # ParallelAlgorithm reconstructs its model in worker processes, so it needs an
        # importable constructor rather than an already-created module.
        if model_name == "fluke_resnet18gn":
            model: Any = "fluke.nets.ResNet18GN"
            model_args = {"output_size": partition.num_classes}
        elif model_name == "torchvision_resnet18_imagenet_bn":
            model = (
                "fl_training.fluke_downstream_fl_evaluation.models."
                "TorchvisionResNet18ImageNetBN"
            )
            model_args = {"output_size": partition.num_classes, "pretrained": True}
        else:
            model = "fl_training.fluke_downstream_fl_evaluation.models.TinyImageNetFlukeNet"
            model_args = {"num_classes": partition.num_classes, "normalization": normalization}
        hyperparameters: dict[str, Any] = {
            "model": model,
            "net_args": model_args,
            "model_args": model_args,
            "client": client,
            "server": server,
        }
    else:
        try:
            server_model = bundle.component(template.model_component)
            client_model = (
                bundle.component(template.client_model_component)
                if template.client_model_component is not None
                else None
            )
        except ValueError as exc:
            raise FlukeEvaluationError(
                f"Algorithm {template.key!r} requires an encoder/head-aware model, but "
                f"model.name={model_name!r} is full-model-only."
            ) from exc
        hyperparameters = {
            "model": server_model,
            "client": client,
            "server": server,
        }
        if client_model is not None:
            hyperparameters["client"]["model"] = client_model

    # Dataset-dependent prototype cardinality must never retain Fluke's MNIST defaults.
    for target in (hyperparameters["client"], hyperparameters["server"]):
        if "n_protos" in target:
            target["n_protos"] = partition.num_classes
    if "embedding_size" in hyperparameters["server"]:
        if bundle.embedding_dim is None:
            raise FlukeEvaluationError(
                f"Algorithm {template.key!r} requires an encoder embedding dimension, but "
                f"model.name={model_name!r} does not expose one."
            )
        hyperparameters["server"]["embedding_size"] = bundle.embedding_dim

    return DDict(hyperparameters), image_size, normalization, model_name


def _configure_fluke_environment(
    cell: Mapping[str, Any], *, num_classes: int, personalized: bool, parallel: bool
) -> str | list[str]:
    _require_fluke()
    from fluke import DDict, FlukeENV
    from fluke.evaluation import ClassificationEval

    run = cell.get("run", {})
    protocol = cell.get("protocol", {})
    if not isinstance(run, Mapping) or not isinstance(protocol, Mapping):
        raise TypeError("run and protocol must be mappings.")
    configured_device = run.get("device")
    if parallel:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise FlukeEvaluationError(
                "ParallelAlgorithm requires at least two CUDA devices; this matrix cell is not "
                "runnable on the current host."
            )
        if configured_device is None:
            configured_device = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if (
            not isinstance(configured_device, list)
            or len(configured_device) < 2
            or not all(isinstance(device, str) and device.startswith("cuda:") for device in configured_device)
        ):
            raise FlukeEvaluationError(
                "ParallelAlgorithm requires run.device to be a list of at least two CUDA devices, "
                "for example ['cuda:0', 'cuda:1']."
            )
    elif configured_device is None:
        configured_device = "cuda" if _cuda_available() else "cpu"
    if not isinstance(configured_device, (str, list)):
        raise FlukeEvaluationError("run.device must be a Fluke device string or a list of CUDA devices.")
    device = configured_device
    eval_every = int(run.get("eval_every", 1))
    if eval_every < 1:
        raise FlukeEvaluationError("run.eval_every must be >= 1.")
    every_round_personalized = bool(run.get("personalized_eval_every_round", False))
    environment = DDict(
        exp={"seed": int(run.get("seed", 42)), "device": device, "inmemory": True},
        save={"path": None, "save_every": 0, "global_only": True},
        eval={
            "eval_every": int(run.get("eval_every", 1)),
            "pre_fit": False,
            "post_fit": bool(personalized and every_round_personalized),
            "locals": False,
            "server": not personalized,
        },
    )
    FlukeENV().configure(environment)
    FlukeENV().set_evaluator(ClassificationEval(eval_every=eval_every, n_classes=num_classes))
    return device


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is an fds FL dependency
        return False


def _adapt_parallel_server_evaluate_signature(server: Any) -> None:
    """Bridge Fluke 0.8.1's incompatible ParallelServer evaluation signatures.

    ``ParallelServer.fit`` delegates to the inherited ``Server._compute_evaluation``,
    which calls ``evaluate(..., round)``. Fluke 0.8.1's ParallelServer override omits
    that argument. Keep the workaround instance-local so custom or future Fluke server
    implementations retain their own native behavior.
    """

    if "round" in inspect.signature(server.evaluate).parameters:
        return

    def evaluate(
        self: Any,
        evaluator: Any,
        test_set: Any,
        round: int | None = None,
    ) -> dict[str, float]:
        if test_set is None:
            return {}
        device = self.device[0] if isinstance(self.device, list) else self.device
        evaluation_round = int(round if round is not None else self._rounds + 1)
        return evaluator.evaluate(evaluation_round, self.model, test_set, device=device)

    server.evaluate = MethodType(evaluate, server)


def _build_fluke_algorithm(
    cell: Mapping[str, Any],
    partition: TinyImageNetPartition,
    template: AlgorithmTemplate,
) -> tuple[Any, str | list[str]]:
    """Construct, but do not run, a fully configured Fluke algorithm."""

    _require_fluke()
    from fluke import FlukeENV
    from fluke.algorithms.decentralized import Topology

    hyperparameters, image_size, _, _ = _build_hyperparameters(cell, template, partition)
    install_fluke_compatibility(template.key)
    device = _configure_fluke_environment(
        cell,
        num_classes=partition.num_classes,
        personalized=template.result_scope == "personalized_shared_global_val",
        parallel=template.protocol == "parallel",
    )
    if template.protocol == "parallel":
        if not isinstance(device, list):  # Defensive guard for third-party environment changes.
            raise FlukeEvaluationError("ParallelAlgorithm did not resolve a multi-CUDA-device list.")

    training = cell.get("training", {})
    if not isinstance(training, Mapping):
        raise TypeError("training must be a mapping.")
    data_loader_workers = int(training.get("num_workers", 0))
    if data_loader_workers < 0:
        raise FlukeEvaluationError("training.num_workers must be >= 0.")
    pin_memory = bool(training.get("pin_memory", _cuda_available()))

    train_transform = build_tiny_imagenet_transforms(
        image_size=image_size, profile=template.transform_profile, train=True
    )
    eval_transform = build_tiny_imagenet_transforms(
        image_size=image_size, profile=template.transform_profile, train=False
    )
    splitter = PartitionDataSplitter(
        partition,
        train_transform=train_transform,
        eval_transform=eval_transform,
        image_size=image_size,
        num_workers=data_loader_workers,
        pin_memory=pin_memory,
    )
    algorithm_class = _import_algorithm_class(template.class_path)
    n_clients = int(cell["protocol"]["n_clients"])
    try:
        if template.protocol == "decentralized":
            topology_name = str(cell.get("algorithm", {}).get("topology", "fully_connected"))
            if topology_name == "fully_connected":
                topology = Topology.fully_connected(n_clients)
            elif topology_name == "ring":
                topology = Topology.ring(n_clients)
            else:
                raise FlukeEvaluationError(
                    "algorithm.topology must be fully_connected or ring for GossipDFL."
                )
            algorithm = algorithm_class(n_clients, splitter, hyperparameters, topology=topology)
        else:
            algorithm = algorithm_class(n_clients, splitter, hyperparameters)
    except Exception as exc:  # noqa: BLE001 - the exact Fluke contract is valuable context
        try:
            FlukeENV().close_cache()
        except Exception:  # noqa: BLE001
            pass
        raise FlukeEvaluationError(
            f"Fluke preflight failed for {template.class_path}. Resolved client/server "
            f"hyperparameters are invalid for this method: {exc}"
        ) from exc
    if template.key == "scaffold":
        configure_scaffold_local_steps(algorithm.clients)
    if template.protocol == "parallel":
        _adapt_parallel_server_evaluate_signature(algorithm.server)
    return algorithm, device


def _close_fluke() -> None:
    try:
        from fluke import FlukeENV

        try:
            FlukeENV().close_cache()
        finally:
            FlukeENV.clear()
    except ImportError:  # pragma: no cover - optional dependency behavior
        return
    except Exception:  # noqa: BLE001 - cleanup must not hide primary failure
        return


def _close_parallel_runtime(algorithm: Any | None) -> None:
    """Release Fluke ParallelServer worker and manager processes deterministically."""

    if algorithm is None:
        return
    server = getattr(algorithm, "server", None)
    running_processes = list(getattr(server, "running_processes", []))
    for entry in running_processes:
        process = entry[0] if isinstance(entry, tuple) else entry
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        except Exception:  # noqa: BLE001 - cleanup must not hide primary failure
            continue
    manager = getattr(server, "manager", None)
    try:
        if manager is not None:
            manager.shutdown()
    except Exception:  # noqa: BLE001 - cleanup must not hide primary failure
        return


def _is_personalized(template: AlgorithmTemplate) -> bool:
    return template.result_scope == "personalized_shared_global_val"


def _completed_rounds(algorithm: Any) -> int:
    """Read the round counter from both centralized and parallel Fluke protocols."""

    candidates = (
        getattr(algorithm, "rounds", None),
        getattr(getattr(algorithm, "server", None), "_rounds", None),
    )
    return max(
        (int(value) for value in candidates if isinstance(value, int) and value >= 0),
        default=0,
    )


def _global_metric_summary(round_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_round: dict[int, dict[str, Any]] = {}
    for item in round_metrics:
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        accuracy = _safe_float(metrics.get("accuracy"))
        if accuracy is not None:
            by_round[int(item["round"])] = {"round": int(item["round"]), "accuracy": accuracy}
    accuracy = [by_round[round_number] for round_number in sorted(by_round)]
    if not accuracy:
        return {"global_accuracy_by_round": [], "final_accuracy": None, "best_accuracy": None}
    values = [item["accuracy"] for item in accuracy if item["accuracy"] is not None]
    return {
        "global_accuracy_by_round": accuracy,
        "final_accuracy": values[-1],
        "best_accuracy": max(values),
    }


def _deduplicate_round_metrics(round_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the final evaluation for each round after Fluke finalization re-evaluates it."""

    by_round: dict[int, dict[str, Any]] = {}
    for item in round_metrics:
        if "round" in item:
            by_round[int(item["round"])] = dict(item)
    return [by_round[round_number] for round_number in sorted(by_round)]


def _personalized_metric_summary(algorithm: Any) -> dict[str, Any]:
    """Evaluate each final client model on its shared canonical global validation loader."""

    from fluke import FlukeENV

    evaluator = FlukeENV().get_evaluator()
    original_eval_every = evaluator.eval_every
    evaluator.eval_every = 1
    values: list[dict[str, Any]] = []
    try:
        for client in algorithm.clients:
            metrics = client.evaluate(evaluator, client.test_set)
            accuracy = _safe_float(metrics.get("accuracy") if isinstance(metrics, Mapping) else None)
            values.append(
                {
                    "client_id": int(client.index),
                    "accuracy": accuracy,
                    "train_samples": int(client.n_examples),
                    "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
                }
            )
    finally:
        evaluator.eval_every = original_eval_every
    available = [item for item in values if item["accuracy"] is not None]
    if not available:
        return {
            "personalized_clients": values,
            "personalized_macro_accuracy": None,
            "personalized_train_weighted_accuracy": None,
        }
    macro = sum(float(item["accuracy"]) for item in available) / len(available)
    denominator = sum(int(item["train_samples"]) for item in available)
    weighted = (
        sum(float(item["accuracy"]) * int(item["train_samples"]) for item in available) / denominator
        if denominator
        else None
    )
    return {
        "personalized_clients": values,
        "personalized_macro_accuracy": macro,
        "personalized_train_weighted_accuracy": weighted,
    }


def _ensure_final_global_evaluation(algorithm: Any, observer: RoundMetricsObserver) -> None:
    """Record the final global metric even when the configured cadence skips it."""

    final_round = _completed_rounds(algorithm)
    if final_round < 1:
        return
    for entry in observer.global_metrics:
        if int(entry.get("round", -1)) != final_round:
            continue
        metrics = entry.get("metrics")
        if isinstance(metrics, Mapping) and _safe_float(metrics.get("accuracy")) is not None:
            return

    from fluke import FlukeENV

    evaluator = FlukeENV().get_evaluator()
    original_eval_every = evaluator.eval_every
    evaluator.eval_every = 1
    try:
        # ParallelServer.evaluate does not accept the centralized ``round`` keyword.
        if "round" in inspect.signature(algorithm.server.evaluate).parameters:
            metrics = algorithm.server.evaluate(
                evaluator,
                algorithm.server.test_set,
                round=final_round,
            )
        else:
            metrics = algorithm.server.evaluate(evaluator, algorithm.server.test_set)
    finally:
        evaluator.eval_every = original_eval_every
    observer.server_evaluation(round=final_round, eval_type="global", evals=metrics)


def _algorithm_hint(cell: Mapping[str, Any]) -> str:
    algorithm = cell.get("algorithm")
    if isinstance(algorithm, Mapping):
        for field in ("class_path", "key", "name"):
            if algorithm.get(field):
                return str(algorithm[field])
    elif algorithm:
        return str(algorithm)
    return "invalid-algorithm"


def _run_id(
    cell: Mapping[str, Any], split_hash: str, template: AlgorithmTemplate | None
) -> str:
    raw_partition = cell.get("partition")
    partition = raw_partition if isinstance(raw_partition, Mapping) else {}
    partition_id = str(partition.get("id") or Path(str(partition.get("path", "partition"))).stem)
    patches = _adapter_patches(template) if template is not None else []
    digest = _config_hash(
        {"cell": cell, "split_hash": split_hash, "adapter_patches": patches}
    )[:12]
    algorithm_key = template.key if template is not None else _algorithm_hint(cell)
    safe_algorithm = algorithm_key.replace("/", "-").replace(".", "-")
    safe_partition = partition_id.replace("/", "-").replace(" ", "-")
    return f"fluke-{safe_algorithm}-{safe_partition}-{digest}"


def _result_is_resumable(run_dir: Path, fingerprint: str) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    result_path = run_dir / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        result.get("status") == "success"
        and manifest.get("extras", {}).get("fluke_config_fingerprint") == fingerprint
    )


def _archive_previous_failure(run_dir: Path) -> None:
    """Keep a failed attempt intact before retrying its deterministic matrix cell."""

    result_path = run_dir / "result.json"
    manifest_path = run_dir / "run_manifest.json"
    if not result_path.is_file():
        return
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if result.get("status") != "failed":
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_dir = run_dir / "failed_attempts" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(result_path, archive_dir / "result.json")
    if manifest_path.is_file():
        shutil.copy2(manifest_path, archive_dir / "run_manifest.json")


def _prepare_partition(cell: Mapping[str, Any], verify_images: bool | None) -> TinyImageNetPartition:
    dataset = cell.get("dataset", {})
    partition = cell.get("partition", {})
    protocol = cell.get("protocol", {})
    if not isinstance(dataset, Mapping) or not isinstance(partition, Mapping) or not isinstance(protocol, Mapping):
        raise TypeError("dataset, partition, and protocol must be mappings.")
    if str(dataset.get("name", "tiny_imagenet")) != "tiny_imagenet":
        raise FlukeEvaluationError("The current Fluke downstream adapter supports only tiny_imagenet.")
    dataset_root = dataset.get("root")
    if not dataset_root:
        raise FlukeEvaluationError("dataset.root is required.")
    if not partition.get("path"):
        raise FlukeEvaluationError("partition.path is required.")
    if verify_images is None:
        verify_images = bool(dataset.get("verify_images", True))
    return load_tiny_imagenet_partition(
        partition["path"],
        dataset_root=dataset_root,
        expected_clients=int(protocol.get("n_clients", 10)),
        verify_images=verify_images,
    )


def run_cell(
    cell: Mapping[str, Any], *, resume: bool = True, verify_images: bool | None = None
) -> dict[str, Any]:
    """Preflight and execute one atomic Fluke matrix cell.

    Success and failure paths both emit canonical research artifacts.  The caller gets
    a compact outcome dictionary suitable for a Slurm-array worker or tests.
    """

    resolved = copy.deepcopy(dict(cell))
    run_cfg = resolved.get("run", {})
    if not isinstance(run_cfg, Mapping):
        raise TypeError("run must be a mapping.")
    output_root = run_cfg.get("output_root")
    if not output_root:
        raise FlukeEvaluationError("run.output_root is required.")

    raw_partition = resolved.get("partition")
    partition_cfg = raw_partition if isinstance(raw_partition, Mapping) else {}
    partition_path = (
        Path(str(partition_cfg["path"])).expanduser().resolve()
        if partition_cfg.get("path")
        else None
    )
    split_hash = (
        hashlib.sha256(partition_path.read_bytes()).hexdigest()
        if partition_path is not None and partition_path.is_file()
        else f"missing-{_config_hash({'partition_path': str(partition_path) if partition_path else None})}"
    )
    template: AlgorithmTemplate | None = None
    template_error: Exception | None = None
    try:
        template = _template_from_cell(resolved)
    except Exception as exc:  # noqa: BLE001 - persisted below as a configuration diagnostic
        template_error = exc
    run_id = _run_id(resolved, split_hash, template)
    run_dir = Path(str(output_root)).expanduser().resolve() / run_id
    patches = _adapter_patches(template) if template is not None else []
    fingerprint = _config_hash(
        {"cell": resolved, "split_hash": split_hash, "adapter_patches": patches}
    )
    if resume and _result_is_resumable(run_dir, fingerprint):
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "resumed"}

    run_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous_failure(run_dir)
    start = time.monotonic()
    training_started: float | None = None
    stage = "configuration"
    device: str | list[str] | None = None
    partition: TinyImageNetPartition | None = None
    algorithm: Any | None = None
    observer: RoundMetricsObserver | None = None
    algorithm_class_path = template.class_path if template is not None else _algorithm_hint(resolved)
    try:
        if template_error is not None:
            raise template_error
        assert template is not None
        _require_fluke()
        if partition_path is None:
            raise FlukeEvaluationError("partition.path is required.")
        if not partition_path.is_file():
            raise FileNotFoundError(f"Partition JSON does not exist: {partition_path}")
        stage = "partition_preflight"
        partition = _prepare_partition(resolved, verify_images)
        stage = "fluke_preflight"
        algorithm, device = _build_fluke_algorithm(resolved, partition, template)
        image_size, normalization, model_name = _effective_model_settings(resolved, template)
        checkpoint_dir = (
            run_dir / "checkpoints"
            if bool(run_cfg.get("save_checkpoints", True))
            and template.result_scope == "global_model"
            else None
        )
        checkpoint_artifacts = (
            {
                "best_model_checkpoint": str(checkpoint_dir / "best.pt"),
                "last_model_checkpoint": str(checkpoint_dir / "last.pt"),
            }
            if checkpoint_dir is not None
            else {}
        )
        manifest = create_manifest(
            run_id=run_id,
            script_name="fl_training.fluke_downstream_fl_evaluation",
            command=sys.argv,
            resolved_config=resolved,
            dataset={
                "name": "tiny_imagenet",
                "root": str(resolved["dataset"]["root"]),
                "canonical_global_validation": True,
                "num_classes": partition.num_classes,
            },
            partition={
                "path": str(partition.path),
                "sha256": partition.sha256,
                "n_clients": partition.n_clients,
                "train_samples": sum(len(samples) for samples in partition.client_samples),
                "global_validation_samples": len(partition.global_validation),
            },
            model={
                "name": model_name,
                "normalization": normalization,
                "image_size": image_size,
                "embedding_dim": 128 if model_name == "tiny_encoder_head" else None,
                "num_classes": partition.num_classes,
            },
            seed=int(resolved.get("run", {}).get("seed", 42)),
            artifacts=checkpoint_artifacts,
            extras={
                "backend": "fluke",
                "fluke_version": _fluke_version(),
                "fluke_source_revision": _fluke_source_revision(),
                "adapter_patches": _adapter_patches(template),
                "fluke_algorithm": algorithm_class_path,
                "result_scope": template.result_scope,
                "fluke_config_fingerprint": fingerprint,
            },
            cwd=Path.cwd(),
            device=device,
        )
        write_manifest(run_dir, manifest)

        rounds = int(resolved["protocol"].get("rounds", 100))
        if rounds < 1:
            raise FlukeEvaluationError("protocol.rounds must be >= 1.")
        stage = "training"
        observer = _fluke_observer(
            progress_path=run_dir / "progress.json",
            total_rounds=rounds,
            checkpoint_dir=checkpoint_dir,
            global_model_supplier=lambda: algorithm.server.model,
            round_diagnostics_supplier=lambda: _method_round_diagnostics(
                algorithm.server, template
            ),
        )
        algorithm.set_callbacks([observer])
        training_started = time.monotonic()
        if template.protocol == "decentralized":
            algorithm.run(rounds)
        else:
            eligible = float(resolved["protocol"].get("eligible_perc", 1.0))
            if not 0 < eligible <= 1:
                raise FlukeEvaluationError("protocol.eligible_perc must be in (0, 1].")
            if template.protocol == "parallel":
                # Fluke 0.8.1 ParallelAlgorithm inherits CentralizedFL.run(), whose
                # client.local_update call is incompatible with ParallelClient. Its
                # public ParallelServer.fit() is the native process/GPU executor.
                # Its inherited finalization broadcasts to ParallelClient, whose
                # receive_model() is intentionally unsupported. This adapter reports
                # only the parallel global model, so per-round evaluation plus the
                # explicit final-global check below is the correct lifecycle.
                algorithm.server.fit(rounds, eligible, finalize=False)
            else:
                algorithm.run(rounds, eligible)

        stage = "metrics"
        if _is_personalized(template) or template.protocol == "decentralized":
            metrics = _personalized_metric_summary(algorithm)
            metrics["round_metrics"] = observer.client_metrics
        else:
            _ensure_final_global_evaluation(algorithm, observer)
            metrics = _global_metric_summary(observer.global_metrics)
            metrics["round_metrics"] = _deduplicate_round_metrics(observer.global_metrics)
            if observer.method_diagnostics:
                metrics["method_diagnostics_by_round"] = observer.method_diagnostics
        if checkpoint_dir is not None:
            checkpoint_artifacts = observer.finalize_checkpoints(
                round=_completed_rounds(algorithm)
            )
        elapsed = time.monotonic() - start
        training_elapsed = time.monotonic() - training_started
        result = create_result(
            run_id=run_id,
            result_type="fl_run",
            status="success",
            summary={
                "backend": "fluke",
                "fluke_version": _fluke_version(),
                "fluke_source_revision": _fluke_source_revision(),
                "adapter_patches": _adapter_patches(template),
                "algorithm": algorithm_class_path,
                "algorithm_key": template.key,
                "result_scope": template.result_scope,
                "rounds_completed": _completed_rounds(algorithm),
                "runtime_seconds": elapsed,
                "training_runtime_seconds": training_elapsed,
                "preflight_runtime_seconds": training_started - start,
                "device": device,
            },
            metrics=metrics,
            artifacts={
                "manifest_path": str(run_dir / "run_manifest.json"),
                **checkpoint_artifacts,
            },
        )
        write_result(run_dir, result)
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "success", "metrics": metrics}
    except Exception as exc:  # noqa: BLE001 - persisted diagnostics are part of the contract
        if observer is not None:
            observer._publish_progress("failed")
        elapsed = time.monotonic() - start
        if not (run_dir / "run_manifest.json").exists():
            minimal_manifest = create_manifest(
                run_id=run_id,
                script_name="fl_training.fluke_downstream_fl_evaluation",
                command=sys.argv,
                resolved_config=resolved,
                dataset={"name": "tiny_imagenet"},
                partition={
                    "path": str(partition_path) if partition_path is not None else None,
                    "sha256": split_hash,
                    **({"n_clients": partition.n_clients} if partition else {}),
                },
                model={},
                seed=_safe_int(resolved.get("run", {}).get("seed", 42)),
                extras={
                    "backend": "fluke",
                    "fluke_algorithm": algorithm_class_path,
                    "adapter_patches": patches,
                    "fluke_config_fingerprint": fingerprint,
                },
                cwd=Path.cwd(),
                device=device,
            )
            write_manifest(run_dir, minimal_manifest)
        result = create_result(
            run_id=run_id,
            result_type="fl_run",
            status="failed",
            summary={
                "backend": "fluke",
                "algorithm": algorithm_class_path,
                "adapter_patches": patches,
                "runtime_seconds": elapsed,
            },
            failure={
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        write_result(run_dir, result)
        raise
    finally:
        _close_parallel_runtime(algorithm)
        _close_fluke()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fluke_version() -> str | None:
    try:
        import fluke

        return getattr(fluke, "__version__", None)
    except ImportError:
        return None


def _fluke_source_revision() -> str | None:
    """Return the installed Fluke checkout revision when it is source-backed."""

    try:
        import fluke

        checkout = Path(fluke.__file__).resolve().parent.parent
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (ImportError, OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def _adapter_patches(template: AlgorithmTemplate) -> list[str]:
    """Expose behavior-changing adapter patches in preflight and run artifacts."""

    return compatibility_patches(template.key)


def _method_round_diagnostics(server: Any, template: AlgorithmTemplate) -> Mapping[str, Any] | None:
    """Extract method state published by a compatibility patch after server aggregation."""

    if template.key != "feddyn":
        return None
    diagnostics = getattr(server, "_fds_last_feddyn_dual_norms", None)
    return dict(diagnostics) if isinstance(diagnostics, Mapping) else None


def preflight_cell(cell: Mapping[str, Any], *, verify_images: bool | None = None) -> dict[str, Any]:
    """Construct one method without training it and return the resolved contract."""

    _require_fluke()
    resolved = copy.deepcopy(dict(cell))
    template = _template_from_cell(resolved)
    partition = _prepare_partition(resolved, verify_images)
    algorithm: Any | None = None
    try:
        algorithm, device = _build_fluke_algorithm(resolved, partition, template)
        return {
            "algorithm": template.class_path,
            "algorithm_key": template.key,
            "fluke_version": _fluke_version(),
            "fluke_source_revision": _fluke_source_revision(),
            "adapter_patches": _adapter_patches(template),
            "protocol": template.protocol,
            "result_scope": template.result_scope,
            "device": device,
            "n_clients": len(algorithm.clients),
            "num_classes": partition.num_classes,
            "server_lr": float(algorithm.server.hyper_params.lr),
            "server_global_step": (
                float(algorithm.server.hyper_params.global_step)
                if "global_step" in algorithm.server.hyper_params
                else None
            ),
        }
    finally:
        _close_parallel_runtime(algorithm)
        _close_fluke()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a Fluke-backed Tiny-ImageNet downstream FL evaluation matrix."
    )
    parser.add_argument("--config", required=True, help="YAML matrix configuration.")
    parser.add_argument("--task-index", type=int, default=None, help="Zero-based matrix cell index.")
    parser.add_argument("--list-tasks", action="store_true", help="Print resolved matrix cells and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight the selected cell without training.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip a matching successful cell.")
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Verify every referenced image header during partition preflight.",
    )
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Check path and label integrity without decoding every image header.",
    )
    args = parser.parse_args(argv)
    if args.verify_images and args.skip_image_verification:
        parser.error("--verify-images and --skip-image-verification cannot be combined.")

    try:
        config = load_config(args.config)
        cells = resolve_matrix_cells(config)
        if args.list_tasks:
            for index, cell in enumerate(cells):
                template = _template_from_cell(cell)
                print(
                    json.dumps(
                        {
                            "task_index": index,
                            "algorithm": template.class_path,
                            "partition": cell["partition"].get("id", cell["partition"]["path"]),
                            "local_epochs": cell["protocol"]["local_epochs"],
                        },
                        sort_keys=True,
                    )
                )
            return 0
        if args.task_index is None:
            raise FlukeEvaluationError("--task-index is required when executing a matrix cell.")
        if not 0 <= args.task_index < len(cells):
            raise FlukeEvaluationError(
                f"task-index {args.task_index} is outside the matrix range [0, {len(cells) - 1}]."
            )
        verification = True if args.verify_images else False if args.skip_image_verification else None
        selected = cells[args.task_index]
        if args.dry_run:
            print(json.dumps(preflight_cell(selected, verify_images=verification), indent=2, sort_keys=True))
        else:
            outcome = run_cell(
                selected,
                resume=not args.no_resume,
                verify_images=verification,
            )
            print(json.dumps(outcome, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should retain artifact then return non-zero
        LOG.error("Fluke downstream evaluation failed: %s", exc)
        return 1
