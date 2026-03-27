"""CLI execution smoke tests for partitioning and training entrypoints."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
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


def _build_pascal_voc_minimal(base_path: Path, n_train: int = 20, n_val: int = 4) -> list[str]:
    root = base_path / "VOC2012"
    train_img_dir = root / "images" / "train"
    train_lbl_dir = root / "labels" / "train"
    val_img_dir = root / "images" / "val"
    val_lbl_dir = root / "labels" / "val"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_names: list[str] = []
    for idx in range(n_train):
        name = f"2008_{idx:06d}"
        train_names.append(name)
        _write_rgb(train_img_dir / f"{name}.jpg", color=((idx * 7) % 255, 100, 120))
        with open(train_lbl_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("person 0.5 0.5 0.4 0.4\n")

    for idx in range(n_val):
        name = f"2011_{idx:06d}"
        _write_rgb(val_img_dir / f"{name}.jpg", color=(80, (idx * 11) % 255, 150))
        with open(val_lbl_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("dog 0.5 0.5 0.3 0.3\n")

    return train_names


def _build_pascal_vlm_annotations(vlm_root: Path, train_names: list[str]) -> None:
    criterion_to_dir = {
        "color": "output_color_train",
        "light": "output_light_train",
        "size": "output_size_train",
        "location": "output_location_train",
        "main_object": "output_main_object_train",
    }
    criterion_labels = {
        "color": ("warm", "cool"),
        "light": ("bright", "dim"),
        "size": ("close", "wide"),
        "location": ("indoor", "outdoor"),
        "main_object": ("person", "animal"),
    }

    for criterion, dirname in criterion_to_dir.items():
        out_dir = vlm_root / dirname
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / "step3_result.jsonl"
        labels = criterion_labels[criterion]
        with open(file_path, "w", encoding="utf-8") as f:
            for idx, name in enumerate(train_names):
                # Different user-root on purpose to verify suffix-based normalization.
                entry = {
                    "image": f"/users/other/FDS/VOC2012/images/train/{name}.jpg",
                    "class": labels[idx % len(labels)],
                }
                f.write(json.dumps(entry) + "\n")


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


def test_generate_baselines_pascal_voc_vlm_cli_smoke(tmp_path, monkeypatch):
    train_names = _build_pascal_voc_minimal(tmp_path, n_train=40, n_val=4)
    vlm_root = tmp_path / "pascal_vlm"
    _build_pascal_vlm_annotations(vlm_root, train_names)
    module = _load_module("generate_baselines_pascal_smoke", "scripts/generate_baselines.py")

    out_dir = tmp_path / "baseline_pascal_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_baselines.py",
            "--dataset",
            "pascal_voc",
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
            "--cluster_labels_dir",
            str(vlm_root),
        ],
    )

    rc = module.main()
    assert rc == 0

    random_split = out_dir / "random_n2_seed7.json"
    dirichlet_split = out_dir / "dirichlet_n2_alpha0.5_seed7.json"
    vlm_split = out_dir / "vlm_n2_alpha0.5_seed7.json"

    assert random_split.exists()
    assert dirichlet_split.exists()
    assert vlm_split.exists()

    with open(vlm_split, "r", encoding="utf-8") as f:
        vlm_payload = json.load(f)

    assert sorted(vlm_payload["clients"].keys()) == ["client_0", "client_1"]
    assert isinstance(vlm_payload.get("global_val"), list)
    sample = vlm_payload["clients"]["client_0"]["train"][0]
    assert "image" in sample
    assert "label" in sample


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


def test_hydra_compose_ade20k_deeplab_contract(tmp_path, monkeypatch, hydra_restore_singletons):
    monkeypatch.setenv("DATASETS_ROOT", str(tmp_path))
    monkeypatch.setenv("ade20k_chosen_split_path", str(tmp_path / "ade_splits.json"))

    from hydra import compose, initialize_config_dir
    from fl_training.models.factory import validate_model_contract
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "fl_training" / "conf")):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=ade20k",
                "model=deeplabv3_resnet50",
                "hydra.job.chdir=false",
                f"paths.output_dir={tmp_path / 'outputs'}",
            ],
        )

    assert str(cfg.dataset.args.data_dir) == str(tmp_path)
    assert str(cfg.dataset.name) == "ade20k"
    assert str(cfg.model.provider) == "torchvision"
    assert str(cfg.model.name) == "deeplabv3_resnet50"
    assert str(cfg.training.task_type) == "segmentation"
    assert int(cfg.training.num_classes) == 150

    validated = validate_model_contract(
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        task_type=str(cfg.training.task_type),
        num_classes=int(cfg.training.num_classes),
    )
    assert validated["provider"] == "torchvision"
    assert validated["task_type"] == "segmentation"


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

    monkeypatch.setattr(fl_job, "build_recipe", lambda *args, **kwargs: DummyRecipe(**kwargs))
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
            "fed": {"method": "fedavg", "args": {}, "n_clients": 2, "rounds": 1},
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
                "optimizer": {"class_name": "Adam", "args": {}},
            },
            "model": {
                "name": "resnet18",
                "provider": "torchvision",
                "args": {"weights": None},
            },
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
    payload = recipe_kwargs["train_args"].split("--config_payload ", 1)[1]
    decoded = json.loads(base64.b64decode(payload).decode("utf-8"))
    assert decoded["training"]["optimizer"]["class_name"] == "Adam"
    assert "optimizer" not in decoded
    assert str(REPO_ROOT / "fl_training" / "client_entry.py") == recipe_kwargs["train_script"]
    initial_model = recipe_kwargs["initial_model"]
    assert hasattr(initial_model, "model_args_b64")
    assert "{" not in initial_model.model_args_b64

    env_kwargs = captured["env_kwargs"]
    assert env_kwargs["num_clients"] == 2
    assert env_kwargs["gpu_config"] == "0,1"
    assert env_kwargs["workspace_root"].endswith("/out/smoke_job")
    assert captured["tracking_type"] == "tensorboard"
    assert (tmp_path / "out" / "smoke_job" / "run_manifest.json").exists()
    assert (tmp_path / "out" / "smoke_job" / "result.json").exists()

    result_payload = json.loads((tmp_path / "out" / "smoke_job" / "result.json").read_text(encoding="utf-8"))
    assert result_payload["result_type"] == "fl_run"
    assert result_payload["status"] == "success"


def test_fl_job_marks_failed_when_nvflare_status_not_success(tmp_path, monkeypatch):
    import fl_training.fl_job as fl_job
    from omegaconf import OmegaConf

    class DummyStatus:
        def get_status(self):
            return "ABORTED"

        def get_result(self):
            return "aborted_result"

    class DummyRecipe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute(self, env):
            return DummyStatus()

    class DummyEnv:
        def __init__(self, num_clients, gpu_config, workspace_root):
            self.num_clients = num_clients
            self.gpu_config = gpu_config
            self.workspace_root = workspace_root

    monkeypatch.setattr(fl_job, "FedAvgRecipe", DummyRecipe)
    monkeypatch.setattr(fl_job, "SimEnv", DummyEnv)
    monkeypatch.setattr(fl_job, "add_experiment_tracking", lambda recipe, tracking_type="tensorboard": None)
    monkeypatch.setattr(fl_job, "create_model", lambda *args, **kwargs: nn.Linear(4, 2))

    cfg = OmegaConf.create(
        {
            "name": "aborted_job",
            "seed": 123,
            "dataset": {
                "name": "tiny_imagenet",
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
                "optimizer": {"class_name": "Adam", "args": {}},
            },
            "model": {
                "name": "resnet18",
                "provider": "torchvision",
                "args": {"weights": None},
            },
            "paths": {"output_dir": str(tmp_path / "out")},
            "system": {"gpu_ids": [0]},
        }
    )

    fl_job.run_job(cfg)

    result_payload = json.loads((tmp_path / "out" / "aborted_job" / "result.json").read_text(encoding="utf-8"))
    assert result_payload["status"] == "failed"
    assert result_payload["metrics"]["status_text"] == "ABORTED"
    assert result_payload["failure"]["error_type"] == "NVFlareRunFailed"


def test_fl_job_writes_failure_artifact_when_execute_raises(tmp_path, monkeypatch):
    import fl_training.fl_job as fl_job
    from omegaconf import OmegaConf

    class DummyRecipe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute(self, env):
            raise RuntimeError("boom")

    class DummyEnv:
        def __init__(self, num_clients, gpu_config, workspace_root):
            self.num_clients = num_clients
            self.gpu_config = gpu_config
            self.workspace_root = workspace_root

    monkeypatch.setattr(fl_job, "build_recipe", lambda *args, **kwargs: DummyRecipe(**kwargs))
    monkeypatch.setattr(fl_job, "SimEnv", DummyEnv)
    monkeypatch.setattr(fl_job, "add_experiment_tracking", lambda recipe, tracking_type="tensorboard": None)
    monkeypatch.setattr(fl_job, "create_model", lambda *args, **kwargs: nn.Linear(4, 2))

    cfg = OmegaConf.create(
        {
            "name": "failed_job",
            "seed": 123,
            "dataset": {
                "name": "tiny_imagenet",
                "class_path": "data_loading.data_modules.TinyImageNetDataModule",
                "dataset_splits": str(tmp_path / "dummy_splits.json"),
                "args": {"root": str(tmp_path / "tiny_data")},
            },
            "fed": {"method": "fedavg", "args": {}, "n_clients": 2, "rounds": 1},
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
                "optimizer": {"class_name": "Adam", "args": {}},
            },
            "model": {
                "name": "resnet18",
                "provider": "torchvision",
                "args": {"weights": None},
            },
            "paths": {"output_dir": str(tmp_path / "out")},
            "system": {"gpu_ids": [0]},
        }
    )

    with pytest.raises(RuntimeError, match="boom"):
        fl_job.run_job(cfg)

    result_payload = json.loads((tmp_path / "out" / "failed_job" / "result.json").read_text(encoding="utf-8"))
    assert result_payload["status"] == "failed"
    assert result_payload["failure"]["error_message"] == "boom"
