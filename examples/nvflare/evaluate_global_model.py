"""Evaluate a global model saved by an NVFlare run of fds on the split's ``global_val`` set.

Usage:
    DATASETS_ROOT=/path/to/datasets \\
    tiny_imagenet_chosen_split_path=/path/to/split.json \\
    python examples/nvflare/evaluate_global_model.py /path/to/FL_global_model.pt \\
        dataset=tiny_imagenet model=resnet18 [more Hydra overrides]

The Hydra overrides should be those of the training run, so that the model and the
validation transforms match. The script builds the data module, the model and the
classification engine exactly as a client does, loads the global weights and prints the
engine's validation metrics.
"""

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fl_training.client_entry import build_datamodule_args, get_data_module  # noqa: E402
from fl_training.engines.classification import ClassificationEngine  # noqa: E402
from fl_training.models.factory import create_model  # noqa: E402


def load_global_weights(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    weights = {key: torch.as_tensor(value) for key, value in payload.items()}
    if weights and all(key.startswith("model.") for key in weights):
        weights = {key[len("model."):]: value for key, value in weights.items()}
    return weights


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    checkpoint = Path(sys.argv[1])
    overrides = sys.argv[2:] + ["paths.output_dir=/tmp/fds-evaluate-global-model"]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "fl_training" / "conf"), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name="config", overrides=overrides), resolve=True)

    data_module = get_data_module(cfg["dataset"]["class_path"], build_datamodule_args(cfg, "site-1"))
    data_module.setup()
    model = create_model(
        model_cfg=cfg["model"], task_type="classification", num_classes=cfg["training"]["num_classes"]
    )
    missing, unexpected = model.load_state_dict(load_global_weights(checkpoint), strict=False)
    if missing or unexpected:
        sys.exit(f"Checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    engine = ClassificationEngine(model=model, config=cfg, device=torch.device("cpu"))
    print(engine.evaluate(data_module.val_dataloader))


if __name__ == "__main__":
    main()
