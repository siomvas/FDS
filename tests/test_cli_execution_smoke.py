"""CLI execution smoke tests for partitioning and training entrypoints."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rgb(path: Path, color=(128, 128, 128), size=(64, 64)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return str(path)


def _build_tiny_imagenet_minimal(base_path: Path) -> None:
    root = base_path / "tiny-imagenet-200"
    train_dir = root / "train"
    val_img_dir = root / "val" / "images"

    # train split: 2 classes x 10 images each (enough for Dirichlet min-per-client defaults)
    classes = ["n000", "n001"]
    for class_idx, class_name in enumerate(classes):
        class_img_dir = train_dir / class_name / "images"
        for img_idx in range(10):
            _write_rgb(
                class_img_dir / f"{class_name}_{img_idx}.JPEG",
                color=(40 * class_idx, 30 * img_idx, 100),
            )

    # val split with annotation mapping
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_rows = []
    for idx in range(2):
        img_name = f"val_{idx}.JPEG"
        _write_rgb(val_img_dir / img_name, color=(20 * idx, 10, 200))
        label = classes[idx % len(classes)]
        val_rows.append(f"{img_name}\t{label}\t0\t0\t64\t64")

    with open(root / "val" / "val_annotations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_rows) + "\n")


def test_generate_baselines_cli_smoke(tmp_path, monkeypatch):
    _build_tiny_imagenet_minimal(tmp_path)
    module = _load_module("generate_baselines_smoke", "scripts/generate_baselines.py")

    out_dir = tmp_path / "baseline_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_baselines.py",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            str(tmp_path),
            "--output",
            str(out_dir),
            "--n_clients",
            "2",
            "--alphas",
            "0.5",
            "--seed",
            "7",
        ],
    )

    rc = module.main()
    assert rc == 0

    random_split = out_dir / "random_n2_seed7.json"
    dirichlet_split = out_dir / "dirichlet_n2_alpha0.5_seed7.json"

    assert random_split.exists()
    assert dirichlet_split.exists()

    with open(random_split, "r", encoding="utf-8") as f:
        random_payload = json.load(f)
    with open(dirichlet_split, "r", encoding="utf-8") as f:
        dirichlet_payload = json.load(f)

    assert sorted(random_payload["clients"].keys()) == ["client_0", "client_1"]
    assert sorted(dirichlet_payload["clients"].keys()) == ["client_0", "client_1"]
    assert isinstance(random_payload.get("global_val"), list)
    assert isinstance(random_payload["clients"]["client_0"]["train"][0], dict)
    assert "image" in random_payload["clients"]["client_0"]["train"][0]
    assert "label" in random_payload["clients"]["client_0"]["train"][0]


def test_generate_embedding_partitions_cli_smoke(tmp_path, monkeypatch):
    _build_tiny_imagenet_minimal(tmp_path)
    module = _load_module("generate_embedding_partitions_smoke", "scripts/generate_embedding_partitions.py")

    def fake_get_embeddings(self, dataset, dataset_type="train"):
        rng = np.random.default_rng(42 if dataset_type == "train" else 99)
        return rng.normal(size=(len(dataset), 8)).astype(np.float32)

    def fake_cluster(self, embeddings):
        n_clusters = int(self.clustering_params.get("n_clusters", 2))
        labels = np.array([i % n_clusters for i in range(len(embeddings))], dtype=int)
        self.cluster_info_ = {
            "method": "KMeans",
            "num_clusters": n_clusters,
            "cluster_sizes": {
                cluster_id: int((labels == cluster_id).sum()) for cluster_id in range(n_clusters)
            },
        }
        return labels

    monkeypatch.setattr(module.EmbeddingSplitter, "_get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(module.EmbeddingSplitter, "_cluster", fake_cluster)

    out_file = tmp_path / "embedding_splits.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_embedding_partitions.py",
            "--dataset",
            "tiny_imagenet",
            "--base_path",
            str(tmp_path),
            "--extractor",
            "dinov2-base",
            "--clustering",
            "kmeans",
            "--n_clusters",
            "2",
            "--tokens",
            "cls",
            "--valset_type",
            "global",
            "--output",
            str(out_file),
            "--cache_dir",
            str(tmp_path / "cache"),
        ],
    )

    rc = module.main()
    assert rc == 0
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert sorted(payload["clients"].keys()) == ["client_0", "client_1"]
    assert isinstance(payload.get("global_val"), list)


def test_run_experiment_cli_smoke_invokes_job_runner(tmp_path, monkeypatch, hydra_restore_singletons):
    # Ensure env-backed interpolations in dataset config are resolvable if touched.
    monkeypatch.setenv("DATASETS_ROOT", str(tmp_path))
    monkeypatch.setenv("tiny_imagenet_chosen_split_path", str(tmp_path / "dummy_splits.json"))

    from hydra import compose, initialize_config_dir
    import fl_training.fl_job as fl_job
    import fl_training.run_experiment as run_experiment

    captured: dict[str, DictConfig] = {}

    def fake_run_job(cfg: DictConfig):
        captured["cfg"] = cfg

    monkeypatch.setattr(fl_job, "run_job", fake_run_job)

    hydra_run_dir = tmp_path / "hydra_run"
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fl_training" / "conf")):
        cfg = compose(
            config_name="config",
            overrides=[
                "fed.rounds=1",
                "fed.n_clients=2",
                "training.local_steps=1",
                "training.batch_size=2",
                "training.amp_enabled=false",
                f"paths.output_dir={tmp_path / 'outputs'}",
                "hydra.job.chdir=false",
                f"hydra.run.dir={hydra_run_dir}",
            ],
        )

    run_experiment.main.__wrapped__(cfg)

    assert "cfg" in captured
    cfg = captured["cfg"]
    assert int(cfg.fed.rounds) == 1
    assert int(cfg.fed.n_clients) == 2
    assert int(cfg.training.local_steps) == 1
    assert int(cfg.training.batch_size) == 2


def test_fl_job_run_smoke_with_stubbed_nvflare(tmp_path, monkeypatch):
    import fl_training.fl_job as fl_job
    from omegaconf import OmegaConf

    captured = {}

    class DummyStatus:
        def get_status(self):
            return "SUCCESS"

        def get_result(self):
            return "ok"

    class DummyRecipe:
        def __init__(self, **kwargs):
            captured["recipe_kwargs"] = kwargs

        def execute(self, env):
            captured["env_instance"] = env
            return DummyStatus()

    class DummyEnv:
        def __init__(self, num_clients, gpu_config, workspace_root):
            captured["env_kwargs"] = {
                "num_clients": num_clients,
                "gpu_config": gpu_config,
                "workspace_root": workspace_root,
            }

    def fake_add_tracking(recipe, tracking_type="tensorboard"):
        captured["tracking_type"] = tracking_type

    monkeypatch.setattr(fl_job, "FedAvgRecipe", DummyRecipe)
    monkeypatch.setattr(fl_job, "SimEnv", DummyEnv)
    monkeypatch.setattr(fl_job, "add_experiment_tracking", fake_add_tracking)
    monkeypatch.setattr(fl_job, "create_model", lambda *args, **kwargs: nn.Linear(4, 2))

    cfg = OmegaConf.create(
        {
            "name": "smoke_job",
            "seed": 123,
            "dataset": {
                "class_path": "data_loading.data_modules.TinyImageNetDataModule",
                "dataset_splits": str(tmp_path / "dummy_splits.json"),
                "args": {"root": str(tmp_path / "tiny_data")},
            },
            "fed": {"n_clients": 2, "rounds": 1},
            "training": {
                "task_type": "classification",
                "num_classes": 3,
                "batch_size": 2,
                "local_steps": 1,
                "steps_or_epochs": "steps",
                "lr": 1e-3,
                "scheduler": "none",
                "optim_resume": "resume",
                "amp_enabled": False,
                "grad_clipping_type": "none",
                "grad_clipping_value": 0.0,
                "loss": "cross_entropy",
            },
            "model": {
                "name": "resnet18",
                "provider": "torchvision",
                "args": {"weights": None},
            },
            "optimizer": {"class_name": "Adam", "args": {}},
            "paths": {"output_dir": str(tmp_path / "out")},
            "system": {"gpu_ids": [0, 1]},
        }
    )

    fl_job.run_job(cfg)

    recipe_kwargs = captured["recipe_kwargs"]
    assert recipe_kwargs["name"] == "smoke_job"
    assert recipe_kwargs["min_clients"] == 2
    assert recipe_kwargs["num_rounds"] == 1
    assert "--task_type classification" in recipe_kwargs["train_args"]
    assert "--config_payload" in recipe_kwargs["train_args"]
    assert str(REPO_ROOT / "fl_training" / "client_entry.py") == recipe_kwargs["train_script"]
    initial_model = recipe_kwargs["initial_model"]
    assert hasattr(initial_model, "model_args_b64")
    assert "{" not in initial_model.model_args_b64

    env_kwargs = captured["env_kwargs"]
    assert env_kwargs["num_clients"] == 2
    assert env_kwargs["gpu_config"] == "0,1"
    assert env_kwargs["workspace_root"].endswith("/out/smoke_job")
    assert captured["tracking_type"] == "tensorboard"
