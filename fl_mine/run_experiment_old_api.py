import os
import json
import shutil
import subprocess
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# Configure logging
log = logging.getLogger(__name__)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the experiment.
    1. Prepares the experiment directory.
    2. Reads base NVFlare JSON configs.
    3. Injects hyperparameters from Hydra config (replacing setup.py).
    4. Runs NVFlare simulator.
    """
    
    # -------------------------------------------------------------------------
    # 1. Setup Paths & Directories
    # -------------------------------------------------------------------------
    # Hydra changes the working directory to a unique run dir automatically.
    # We will write our modified configs here.
    current_work_dir = Path.cwd()
    log.info(f"Working directory: {current_work_dir}")

    # Path to the original 'template' JSONs (source of truth)
    template_root = Path(cfg.paths.template_root)
    if not template_root.exists():
        raise FileNotFoundError(f"Template root not found: {template_root}")
    
    # Define filenames
    client_conf_name = "config_fed_client.json"
    server_conf_name = "config_fed_server.json"
    train_conf_name = "config_train.json" # Inside the 'config' subfolder in the job structure
    meta_conf_name = "meta.json"

    # Create the standard NVFlare job structure in the current run dir
    # JobName/
    #   config/
    #     config_fed_client.json
    #     config_fed_server.json
    #     config_train.json
    #   meta.json
    
    job_name = cfg.name
    job_dir = current_work_dir / job_name
    config_dir = job_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Load & Modify Configurations (Logic ported from setup.py)
    # -------------------------------------------------------------------------
    
    # Load base templates
    # Note: setup.py assumed structure job/config/*.json. We adapt to find them in template_root.
    # We look for them in template_root/config/ or template_root/ directly depending on your structure.
    # Based on your setup.py, they seem to be in a 'config' subdir.
    try:
        server_config = load_json(template_root / "config" / server_conf_name)
        client_config = load_json(template_root / "config" / client_conf_name)
        train_config = load_json(template_root / "config" / train_conf_name)
        meta_config = load_json(template_root / meta_conf_name)
    except FileNotFoundError:
        # Fallback if templates are flat
        server_config = load_json(template_root / server_conf_name)
        client_config = load_json(template_root / client_conf_name)
        train_config = load_json(template_root / train_conf_name)
        meta_config = load_json(template_root / meta_conf_name)

    # --- Apply Overrides from Hydra Config ---

    # Method Logic
    method = cfg.fed.method
    if method == "central":
        method = "fedavg" # NVFlare treats central as fedavg with 1 client usually, or specific logic
    
    log.info(f"Configuring experiment for method: {method}")

    # Update Client Config
    # Assuming the first component is the learner, or finding it by ID
    # In your setup.py: client_config["components"][0]["args"]["method"] = args.method
    # A safer way is to find the component by name if possible, but we stick to index 0 for compatibility
    client_config["components"][0]["args"]["method"] = method
    client_config["components"][0]["args"]["seed"] = cfg.seed
    client_config["components"][0]["args"]["hyperion"] = cfg.system.use_hyperion

    # Update Server Config
    server_config["min_clients"] = cfg.fed.min_clients
    server_config["num_rounds"] = cfg.fed.rounds
    server_config["seed"] = cfg.seed
    server_config["hyperion"] = cfg.system.use_hyperion
    
    # Model Path injection
    model_path = f"fl_bench.networks.fed_isic_nets.{cfg.training.model_name}"
    server_config["model_path"] = model_path # Used in server config substitution

    # FedOpt Logic
    if method == "fedopt":
        # workflow[0] is scatter_and_gather
        server_config["workflows"][0]["args"]["shareable_generator_id"] = "shareable_generator_fedopt"
    else:
        server_config["workflows"][0]["args"]["shareable_generator_id"] = "shareable_generator_base"

    # SCAFFOLD Logic
    # Replicating the specific dictionary replacements from setup.py
    if method == "scaffold":
        server_config["task_result_filters"] = [
            {
                "tasks": ["train", "submit_model", "validate"],
                "filters": [
                    {
                        "name": "ExcludeVars",
                        "args": {"exclude_vars": "(?:num_batches_tracked)|(?:attn.relative_position_index)"},
                    }
                ],
            }
        ]
        # Update aggregator expected data kind
        # Finding aggregator component (usually index 4 in your file, but let's be safe)
        for comp in server_config["components"]:
            if comp["id"] == "aggregator":
                comp["args"]["expected_data_kind"] = {
                    "_model_weights_": "WEIGHT_DIFF",
                    "scaffold_c_diff": "WEIGHT_DIFF",
                }
        
        server_config["workflows"][0]["name"] = "ScatterAndGatherScaffold"
    else:
        # Reset to defaults for non-scaffold
        server_config["task_result_filters"] = []
        for comp in server_config["components"]:
            if comp["id"] == "aggregator":
                comp["args"]["expected_data_kind"] = "WEIGHT_DIFF"
        server_config["workflows"][0]["name"] = "ScatterAndGather"

    # Exclude Vars Logic (FedBN, FedPer, etc)
    exclude_vars = ""
    if method in ["fedattn", "fedper", "fedbn"]:
        exclude_vars = "" # Your setup.py actually sets this to empty string for these methods, intentionally?
        # If you meant to set it, change here. adhering to setup.py logic:
        # setup.py line 93-97: sets to "" for fedattn, fedper, fedbn
    
    for comp in server_config["components"]:
        if comp["id"] == "aggregator":
            if "exclude_vars" in comp["args"]:
                comp["args"]["exclude_vars"] = exclude_vars

    # Update Train Config
    train_config["learning_rate"] = cfg.training.lr
    train_config["fedprox_mu"] = cfg.fed.prox_mu if method == "fedprox" else 0.0
    train_config["batch_size"] = cfg.training.batch_size
    train_config["local_steps"] = cfg.training.local_steps
    train_config["max_steps"] = cfg.fed.rounds * cfg.training.local_steps
    train_config["scheduler"] = cfg.training.scheduler
    train_config["model_name"] = cfg.training.model_name
    train_config["optim_resume"] = cfg.training.optim_resume
    train_config["use_half_precision"] = cfg.training.amp
    train_config["grad_clipping_type"] = cfg.training.grad_clipping_type
    train_config["grad_clipping_value"] = cfg.training.grad_clipping_value
    train_config["steps_or_epochs"] = cfg.training.steps_or_epochs
    
    # Optimizer args injection
    if "optimizer" in cfg.training:
         train_config["optimizer_args"] = OmegaConf.to_container(cfg.training.optimizer.args, resolve=True)
         train_config["optimizer_args"]["class"] = cfg.training.optimizer.class_name

    # Update Meta Config
    meta_config["name"] = job_name
    meta_config["min_clients"] = cfg.fed.min_clients

    # -------------------------------------------------------------------------
    # 3. Write Configs to Run Directory
    # -------------------------------------------------------------------------
    save_json(server_config, config_dir / server_conf_name)
    save_json(client_config, config_dir / client_conf_name)
    save_json(train_config, config_dir / train_conf_name)
    save_json(meta_config, job_dir / meta_conf_name)

    log.info(f"Generated configuration files in {job_dir}")

    # -------------------------------------------------------------------------
    # 4. Run NVFlare Simulator
    # -------------------------------------------------------------------------
    # Construct command
    # nvflare simulator ${job_folder} -w ${workspace} -n ${clients} -gpu ${gpus}
    
    workspace = Path(cfg.paths.output_dir) / job_name
    gpu_ids = ",".join(map(str, cfg.system.gpu_ids))
    
    cmd = [
        "nvflare", "simulator",
        str(job_dir),
        "-w", str(workspace),
        "-n", str(cfg.fed.min_clients),
        "-gpu", gpu_ids
    ]

    log.info(f"Running command: {' '.join(cmd)}")
    
    # Execute
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log.error("NVFlare Simulator Failed.")
        raise e

if __name__ == "__main__":
    main()