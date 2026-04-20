import json
import base64
import inspect
import logging
from pathlib import Path
import sys
from typing import Any

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# NVFlare Imports
from nvflare.recipe import SimEnv, add_experiment_tracking

from fl_training.fed_methods import build_recipe, validate_method_task_compatibility
from fl_training.models.factory import create_model, validate_model_contract
from research import create_manifest, create_result, file_sha256, write_manifest, write_result

log = logging.getLogger(__name__)

# Optional compatibility hook for tests that monkeypatch a recipe class directly.
FedAvgRecipe = None


def _is_success_status(raw_status: object) -> bool:
    normalized = str(raw_status).strip().lower()
    if not normalized:
        return False
    success_tokens = ("success", "succeeded", "complete", "completed", "ok", "done", "finished")
    return any(token in normalized for token in success_tokens)


def _normalize_gpu_config(raw_gpu_ids) -> str:
    """Normalize Hydra/OmegaConf GPU config to NVFlare SimEnv string format."""
    if OmegaConf.is_config(raw_gpu_ids):
        raw_gpu_ids = OmegaConf.to_container(raw_gpu_ids, resolve=True)

    if raw_gpu_ids is None:
        return ""
    if isinstance(raw_gpu_ids, str):
        return raw_gpu_ids
    if isinstance(raw_gpu_ids, (list, tuple)):
        return ",".join(str(gpu) for gpu in raw_gpu_ids)
    return str(raw_gpu_ids)


class PersistableFactoryModel(nn.Module):
    """Wrap factory-created models with a constructor that NVFlare can rehydrate."""

    def __init__(
        self,
        provider: str = "timm",
        name: str = "resnet18",
        task_type: str = "classification",
        num_classes: int = 1000,
        model_args_b64: str = "e30=",  # base64url-encoded "{}"
    ):
        super().__init__()
        self.provider = str(provider)
        self.name = str(name)
        self.task_type = str(task_type)
        self.num_classes = int(num_classes)
        self.model_args_b64 = str(model_args_b64)

        try:
            json_bytes = base64.urlsafe_b64decode(self.model_args_b64.encode("utf-8"))
            model_args = json.loads(json_bytes.decode("utf-8"))
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            raise ValueError("model_args_b64 must be a base64-encoded JSON object.") from e
        if not isinstance(model_args, dict):
            raise ValueError("Decoded model_args_b64 must be a mapping.")

        self._inner_model = create_model(
            model_cfg={"provider": self.provider, "name": self.name, "args": model_args},
            task_type=self.task_type,
            num_classes=self.num_classes,
        )

    def forward(self, *args, **kwargs):
        return self._inner_model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._inner_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self._inner_model.load_state_dict(state_dict, strict=strict)


def encode_config(cfg: DictConfig) -> str:

    client_config = {
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "fed": OmegaConf.to_container(cfg.fed, resolve=True),
        "training": OmegaConf.to_container(cfg.training, resolve=True),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "seed": cfg.seed
    }
    
    # 2. Dump to JSON
    json_str = json.dumps(client_config)
    
    # 3. Encode to Base64
    b64_str = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    
    return b64_str

def get_initial_model(cfg: DictConfig) -> nn.Module:
    """
    Instantiates the initial global model on the Server using the factory.
    """

    task_type = cfg.training.task_type
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    
    # Retrieve num_classes (injected by the dataset config into training)
    if "num_classes" not in cfg.training:
         raise ValueError("cfg.training.num_classes is missing. Ensure your dataset config sets this.")
    
    num_classes = cfg.training.num_classes

    validated = validate_model_contract(
        model_cfg,
        task_type=str(task_type),
        num_classes=int(num_classes),
    )
    log.info(
        "Initializing Server Model via Factory | Task: %s | Classes: %s | Model Config: %s",
        task_type,
        num_classes,
        model_cfg,
    )

    provider = validated["provider"]
    name = validated["name"]
    model_args = validated["args"]

    # Use a persistable wrapper so NVFlare can safely reconstruct the model.
    model_args_b64 = base64.urlsafe_b64encode(
        json.dumps(model_args, sort_keys=True).encode("utf-8")
    ).decode("utf-8")

    return PersistableFactoryModel(
        provider=provider,
        name=name,
        task_type=validated["task_type"],
        num_classes=validated["num_classes"],
        model_args_b64=model_args_b64,
    )


def _dataset_metadata(cfg: DictConfig) -> dict:
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_splits = dataset_cfg.get("dataset_splits")
    partition = {
        "path": dataset_splits,
        "filename": Path(dataset_splits).name if dataset_splits else None,
        "sha256": file_sha256(Path(dataset_splits)) if dataset_splits and Path(dataset_splits).exists() else None,
    }
    dataset = {
        "name": dataset_cfg.get("name") or dataset_cfg.get("class_path") or "unknown",
        "base_path": (dataset_cfg.get("args") or {}).get("root"),
        "class_path": dataset_cfg.get("class_path"),
        "split": "federated",
    }
    return {"dataset": dataset, "partition": partition}


def _nvflare_runtime_overrides(cfg: DictConfig) -> tuple[dict, dict]:
    """Return client/server runtime overrides for generated NVFlare app configs."""
    fed_cfg = cfg.get("fed")
    if fed_cfg is None:
        return {}, {}

    nvflare_cfg = fed_cfg.get("nvflare")
    if nvflare_cfg is None:
        return {}, {}

    if OmegaConf.is_config(nvflare_cfg):
        nvflare_cfg = OmegaConf.to_container(nvflare_cfg, resolve=True)
    if not isinstance(nvflare_cfg, dict):
        raise TypeError(f"fed.nvflare must be a mapping, got {type(nvflare_cfg).__name__}")

    client_cfg = nvflare_cfg.get("client") or {}
    server_cfg = nvflare_cfg.get("server") or {}
    if not isinstance(client_cfg, dict):
        raise TypeError(f"fed.nvflare.client must be a mapping, got {type(client_cfg).__name__}")
    if not isinstance(server_cfg, dict):
        raise TypeError(f"fed.nvflare.server must be a mapping, got {type(server_cfg).__name__}")
    return dict(client_cfg), dict(server_cfg)


def _fed_settings(cfg: DictConfig) -> tuple[str, dict[str, Any]]:
    fed_cfg = cfg.get("fed")
    if fed_cfg is None:
        raise KeyError("Missing required 'fed' config section.")

    method = str(fed_cfg.get("method", "fedavg"))
    fed_args_cfg = fed_cfg.get("args", {})
    if OmegaConf.is_config(fed_args_cfg):
        fed_args = OmegaConf.to_container(fed_args_cfg, resolve=True)
    else:
        fed_args = fed_args_cfg
    if not isinstance(fed_args, dict):
        raise TypeError(f"fed.args must be a mapping, got {type(fed_args).__name__}")
    return method, dict(fed_args)


def _build_fedavg_recipe_override(
    *,
    name: str,
    initial_model: nn.Module,
    min_clients: int,
    num_rounds: int,
    train_script: str,
    train_args: str,
):
    if FedAvgRecipe is None:
        return None

    recipe_kwargs = {
        "name": name,
        "min_clients": min_clients,
        "num_rounds": num_rounds,
        "train_script": train_script,
        "train_args": train_args,
    }
    signature = inspect.signature(FedAvgRecipe.__init__)
    has_model_kw = "model" in signature.parameters
    has_initial_model_kw = "initial_model" in signature.parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if has_model_kw:
        recipe_kwargs["model"] = initial_model
    elif has_initial_model_kw or accepts_var_kwargs:
        recipe_kwargs["initial_model"] = initial_model
    else:
        recipe_kwargs["model"] = initial_model
    return FedAvgRecipe(**recipe_kwargs)


def run_job(cfg: DictConfig):
    """
    Builds and executes the configured FL recipe using the Config Payload pattern.
    """
    workspace_path = Path(cfg.paths.output_dir) / cfg.name
    workspace_path.mkdir(parents=True, exist_ok=True)
    gpu_config = _normalize_gpu_config(cfg.system.gpu_ids)
    metadata = _dataset_metadata(cfg)
    validated_model = validate_model_contract(
        OmegaConf.to_container(cfg.model, resolve=True),
        task_type=str(cfg.training.task_type),
        num_classes=int(cfg.training.num_classes),
    )
    fed_method, fed_args = _fed_settings(cfg)
    validate_method_task_compatibility(
        {
            "fed": {"method": fed_method, "args": fed_args},
            "training": {"task_type": str(cfg.training.task_type)},
        },
        str(cfg.training.task_type),
    )

    manifest = create_manifest(
        run_id=str(cfg.name),
        script_name="fl_training/fl_job.py",
        command=list(sys.argv),
        resolved_config=OmegaConf.to_container(cfg, resolve=True),
        dataset=metadata["dataset"],
        partition=metadata["partition"],
        model={
            "provider": validated_model["provider"],
            "name": validated_model["name"],
            "task_type": validated_model["task_type"],
            "num_classes": validated_model["num_classes"],
            "args": validated_model["args"],
        },
        seed=int(cfg.seed),
        artifacts={
            "run_dir": str(workspace_path),
            "manifest_path": str(workspace_path / "run_manifest.json"),
            "result_path": str(workspace_path / "result.json"),
        },
        device=None,
        gpu_config=gpu_config,
    )
    write_manifest(workspace_path, manifest)

    stage = "payload_generation"
    try:
        # ---------------------------------------------------------------------
        # 1. Payload Generation
        # ---------------------------------------------------------------------
        config_payload = encode_config(cfg)
        train_args = (
            f"--task_type {cfg.training.task_type} "
            f"--config_payload {config_payload}"
        )
        log.info("Client Launch Command Args: %s", train_args)

        # ---------------------------------------------------------------------
        # 2. Server Setup
        # ---------------------------------------------------------------------
        stage = "initial_model"
        initial_model = get_initial_model(cfg)
        train_script = str(Path(__file__).parent / "client_entry.py")

        recipe = None
        if fed_method.lower() == "fedavg":
            recipe = _build_fedavg_recipe_override(
                name=str(cfg.name),
                initial_model=initial_model,
                min_clients=int(cfg.fed.n_clients),
                num_rounds=int(cfg.fed.rounds),
                train_script=train_script,
                train_args=train_args,
            )
        if recipe is None:
            recipe = build_recipe(
                fed_method,
                name=str(cfg.name),
                initial_model=initial_model,
                min_clients=int(cfg.fed.n_clients),
                num_rounds=int(cfg.fed.rounds),
                train_script=train_script,
                train_args=train_args,
                method_args=fed_args,
            )
        client_runtime_cfg, server_runtime_cfg = _nvflare_runtime_overrides(cfg)
        if client_runtime_cfg and hasattr(recipe, "add_client_config"):
            recipe.add_client_config(client_runtime_cfg)
            log.info("Applied NVFlare client runtime config overrides: %s", client_runtime_cfg)
        if server_runtime_cfg and hasattr(recipe, "add_server_config"):
            recipe.add_server_config(server_runtime_cfg)
            log.info("Applied NVFlare server runtime config overrides: %s", server_runtime_cfg)

        add_experiment_tracking(recipe, tracking_type="tensorboard")

        # ---------------------------------------------------------------------
        # 3. Execution
        # ---------------------------------------------------------------------
        log.info("Launching NVFlare Simulation: %s", cfg.name)

        num_threads = cfg.fed.nvflare.get("num_threads", None)
        env = SimEnv(
            num_clients=cfg.fed.n_clients,
            num_threads=int(num_threads) if num_threads is not None else None,
            gpu_config=gpu_config,
            workspace_root=str(workspace_path)
        )
        stage = "execute"
        run_status = recipe.execute(env=env)
        status_text = str(run_status.get_status())
        result_status = "success" if _is_success_status(status_text) else "failed"

        log.info("Job Status: %s", status_text)
        log.info("Result Path: %s", run_status.get_result())
        result = create_result(
            run_id=str(cfg.name),
            result_type="fl_run",
            status=result_status,
            summary={
                "name": str(cfg.name),
                "task_type": str(cfg.training.task_type),
                "n_clients": int(cfg.fed.n_clients),
                "rounds": int(cfg.fed.rounds),
                "method": fed_method,
                "workspace_root": str(workspace_path),
            },
            artifacts={
                "workspace_root": str(workspace_path),
                "train_script": train_script,
                "nvflare_result": str(run_status.get_result()),
            },
            metrics={
                "status_text": status_text,
            },
            failure=None
            if result_status == "success"
            else {
                "error_type": "NVFlareRunFailed",
                "error_message": f"NVFlare run ended with status: {status_text}",
                "stage": stage,
            },
        )
        write_result(workspace_path, result)
        return run_status
    except Exception as exc:  # noqa: BLE001 - persist canonical failure artifact
        write_result(
            workspace_path,
            create_result(
                run_id=str(cfg.name),
                result_type="fl_run",
                status="failed",
                summary={
                    "name": str(cfg.name),
                    "task_type": str(cfg.training.task_type),
                    "n_clients": int(cfg.fed.n_clients),
                    "rounds": int(cfg.fed.rounds),
                    "workspace_root": str(workspace_path),
                },
                artifacts={
                    "workspace_root": str(workspace_path),
                    "train_script": str(Path(__file__).parent / "client_entry.py"),
                },
                failure={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "stage": stage,
                },
            ),
        )
        raise
