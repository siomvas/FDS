import json
import base64
import logging
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# NVFlare Imports
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

from models.factory import create_model

log = logging.getLogger(__name__)

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

    # The factory handles the provider dispatch (timm, hf, smp)
    return create_model(model_cfg, task_type, num_classes)

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
        gpu_config=cfg.system.gpu_ids,
        workspace_root=str(workspace_path)
    )
    run_status = recipe.execute(env=env)
    
    log.info(f"Job Status: {run_status.get_status()}")
    log.info(f"Result Path: {run_status.get_result()}")
