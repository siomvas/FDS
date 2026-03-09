import json
import base64
import logging
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# NVFlare Imports
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

from fl_training.models.factory import create_model

log = logging.getLogger(__name__)


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

    log.info(f"Initializing Server Model via Factory | Task: {task_type} | Classes: {num_classes} | Model Config: {model_cfg}")

    provider = str(model_cfg.get("provider", "timm"))
    name = str(model_cfg.get("name"))
    model_args = model_cfg.get("args", {})
    if not isinstance(model_args, dict):
        raise TypeError(f"model.args must be a mapping, got {type(model_args).__name__}.")

    # Use a persistable wrapper so NVFlare can safely reconstruct the model.
    model_args_b64 = base64.urlsafe_b64encode(
        json.dumps(model_args, sort_keys=True).encode("utf-8")
    ).decode("utf-8")

    return PersistableFactoryModel(
        provider=provider,
        name=name,
        task_type=str(task_type),
        num_classes=int(num_classes),
        model_args_b64=model_args_b64,
    )

def run_job(cfg: DictConfig):
    """
    Builds and executes the FedAvgRecipe using the Config Payload pattern.
    """
    
    # -------------------------------------------------------------------------
    # 1. Payload Generation
    # -------------------------------------------------------------------------
    # Instead of manual args, we create one robust payload.
    config_payload = encode_config(cfg)

    # We pass the Task Type explicitly for client_entry.py
    # to route quickly, and the Payload for everything else.
    train_args = (
        f"--task_type {cfg.training.task_type} "
        f"--config_payload {config_payload}"
    )
    
    log.info(f"Client Launch Command Args: {train_args}")

    # -------------------------------------------------------------------------
    # 2. Server Setup
    # -------------------------------------------------------------------------
    initial_model = get_initial_model(cfg)

    recipe = FedAvgRecipe(
        name=cfg.name,
        initial_model=initial_model,
        min_clients=cfg.fed.n_clients,
        num_rounds=cfg.fed.rounds,
        train_script=str(Path(__file__).parent / "client_entry.py"), 
        train_args=train_args,
    )

    add_experiment_tracking(recipe, tracking_type="tensorboard")

    # -------------------------------------------------------------------------
    # 3. Execution
    # -------------------------------------------------------------------------
    log.info(f"Launching NVFlare Simulation: {cfg.name}")
    
    workspace_path = Path(cfg.paths.output_dir) / cfg.name

    env = SimEnv(
        num_clients=cfg.fed.n_clients,
        gpu_config=_normalize_gpu_config(cfg.system.gpu_ids),
        workspace_root=str(workspace_path)
    )
    run_status = recipe.execute(env=env)
    
    log.info(f"Job Status: {run_status.get_status()}")
    log.info(f"Result Path: {run_status.get_result()}")
