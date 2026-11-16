"""Tests for the isolated Fluke downstream evaluation adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
from PIL import Image
from torchvision import transforms

pytest.importorskip("fluke")

from fluke.data import FastDataLoader
from fl_training.fluke_downstream_fl_evaluation.catalog import (
    FLUKE_ALGORITHM_CATALOG,
    catalog_coverage_gaps,
    resolve_algorithm_template,
)
from fl_training.fluke_downstream_fl_evaluation.data import (
    LazyFastDataLoader,
    LazyImageTensor,
    PartitionDataSplitter,
    build_tiny_imagenet_transforms,
    load_tiny_imagenet_partition,
)
from fl_training.fluke_downstream_fl_evaluation.fluke_compat import (
    apply_fednova_delta_update,
    compatibility_patches,
    corrected_feddyn_dual,
    executed_local_steps,
    fednova_delta_weights,
    scaffold_momentum_normalization,
    scaffold_control_weights,
    scaffold_steps_per_epoch,
    zero_sgd_momentum_buffers,
)
from fl_training.fluke_downstream_fl_evaluation.models import (
    TinyImageNetFlukeNet,
    TorchvisionResNet18ImageNetBN,
    build_tiny_imagenet_model_bundle,
)
from fl_training.fluke_downstream_fl_evaluation.runner import (
    FlukeEvaluationError,
    RoundMetricsObserver,
    _adapt_parallel_server_evaluate_signature,
    _close_parallel_runtime,
    _completed_rounds,
    _config_hash,
    _fluke_source_revision,
    _run_id,
    load_config,
    preflight_cell,
    resolve_matrix_cells,
    run_cell,
)


@pytest.mark.unit
def test_adapter_modules_remain_importable_without_the_optional_fluke_extra():
    script = """
import importlib.abc
import sys

class BlockFluke(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fluke' or fullname.startswith('fluke.'):
            raise ModuleNotFoundError('blocked optional dependency')
        return None

sys.meta_path.insert(0, BlockFluke())
import fl_training.fluke_downstream_fl_evaluation.data
import fl_training.fluke_downstream_fl_evaluation.runner
print('imported')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "imported"


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


@pytest.fixture
def tiny_partition_fixture(tmp_path: Path) -> dict[str, Path]:
    dataset_root = tmp_path / "datasets"
    tiny_root = dataset_root / "tiny-imagenet-200"
    classes = ("n000", "n001")
    clients = {"client_0": {"train": []}, "client_1": {"train": []}}

    for class_index, label in enumerate(classes):
        image_dir = tiny_root / "train" / label / "images"
        for sample_index in range(4):
            image_path = image_dir / f"{label}_{sample_index}.JPEG"
            _write_image(image_path, (50 * class_index, 30 * sample_index, 90))
            # Deliberately use a foreign root to cover suffix-based path rebasing.
            clients[f"client_{sample_index % 2}"]["train"].append(
                {
                    "image": (
                        f"/mnt/elsewhere/datasets/tiny-imagenet-200/train/{label}/images/"
                        f"{image_path.name}"
                    ),
                    "label": label,
                }
            )

    val_dir = tiny_root / "val" / "images"
    annotations: list[str] = []
    for sample_index, label in enumerate(classes):
        image_path = val_dir / f"val_{sample_index}.JPEG"
        _write_image(image_path, (20 * sample_index, 100, 120))
        annotations.append(f"{image_path.name}\t{label}\t0\t0\t64\t64")
    (tiny_root / "val" / "val_annotations.txt").parent.mkdir(parents=True, exist_ok=True)
    (tiny_root / "val" / "val_annotations.txt").write_text(
        "\n".join(annotations) + "\n", encoding="utf-8"
    )

    partition_path = tmp_path / "split.json"
    partition_path.write_text(
        json.dumps(
            {
                "clients": clients,
                # The adapter intentionally ignores this field in favour of canonical val.
                "global_val": [{"image": "/invalid/path.jpg", "label": "n000"}],
            }
        ),
        encoding="utf-8",
    )
    return {"dataset_root": dataset_root, "partition_path": partition_path}


def _cell(tiny_partition_fixture: dict[str, Path], *, algorithm: str = "fedavg") -> dict:
    return {
        "run": {
            "output_root": str(tiny_partition_fixture["dataset_root"].parent / "outputs"),
            "device": "cpu",
            "seed": 7,
        },
        "dataset": {
            "name": "tiny_imagenet",
            "root": str(tiny_partition_fixture["dataset_root"]),
            "verify_images": True,
        },
        "model": {"image_size": 32, "normalization": "group"},
        "protocol": {"n_clients": 2, "rounds": 1, "eligible_perc": 1.0, "local_epochs": 1},
        "training": {
            "batch_size": 2,
            "optimizer": {"name": "SGD", "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0},
            "scheduler": {"name": "StepLR", "step_size": 1, "gamma": 1.0},
        },
        "partition": {"id": "fixture", "path": str(tiny_partition_fixture["partition_path"])},
        "algorithm": {"key": algorithm},
    }


@pytest.mark.unit
def test_partition_rebases_paths_and_uses_canonical_global_validation(tiny_partition_fixture):
    partition = load_tiny_imagenet_partition(
        tiny_partition_fixture["partition_path"],
        dataset_root=tiny_partition_fixture["dataset_root"],
        expected_clients=2,
        verify_images=True,
    )

    assert partition.n_clients == 2
    assert sum(len(samples) for samples in partition.client_samples) == 8
    assert len(partition.global_validation) == 2
    assert all("tiny-imagenet-200" in str(sample.image) for sample in partition.client_samples[0])
    assert all(sample.image.is_file() for sample in partition.global_validation)


@pytest.mark.unit
def test_partition_rebases_even_when_the_foreign_source_path_exists(tiny_partition_fixture):
    dataset_root = tiny_partition_fixture["dataset_root"]
    target = dataset_root / "tiny-imagenet-200" / "train" / "n000" / "images" / "n000_0.JPEG"
    foreign_root = dataset_root.parent / "foreign" / "tiny-imagenet-200"
    foreign = foreign_root / "train" / "n000" / "images" / "n000_0.JPEG"
    _write_image(foreign, (255, 0, 0))

    from fl_training.fluke_downstream_fl_evaluation.data import rebase_tiny_imagenet_path

    assert rebase_tiny_imagenet_path(foreign, dataset_root) == target.resolve()


@pytest.mark.unit
def test_partition_path_rebasing_rejects_parent_directory_escape(tiny_partition_fixture):
    from fl_training.fluke_downstream_fl_evaluation.data import rebase_tiny_imagenet_path

    with pytest.raises(ValueError, match="escapes the Tiny-ImageNet root"):
        rebase_tiny_imagenet_path(
            "/foreign/tiny-imagenet-200/train/n000/images/../../../../outside.JPEG",
            tiny_partition_fixture["dataset_root"],
        )


@pytest.mark.unit
def test_partition_rejects_mismatched_labels_and_cross_client_duplicates(tiny_partition_fixture):
    payload = json.loads(tiny_partition_fixture["partition_path"].read_text(encoding="utf-8"))
    duplicate = payload["clients"]["client_0"]["train"][0]
    payload["clients"]["client_1"]["train"].append(duplicate)
    tiny_partition_fixture["partition_path"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="assigned to both"):
        load_tiny_imagenet_partition(
            tiny_partition_fixture["partition_path"],
            dataset_root=tiny_partition_fixture["dataset_root"],
            expected_clients=2,
            verify_images=False,
        )

    payload["clients"]["client_1"]["train"].pop()
    payload["clients"]["client_0"]["train"][0]["label"] = "n001"
    tiny_partition_fixture["partition_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with class"):
        load_tiny_imagenet_partition(
            tiny_partition_fixture["partition_path"],
            dataset_root=tiny_partition_fixture["dataset_root"],
            expected_clients=2,
            verify_images=False,
        )


@pytest.mark.unit
def test_lazy_fast_data_loader_preserves_required_fluke_surface(tiny_partition_fixture):
    partition = load_tiny_imagenet_partition(
        tiny_partition_fixture["partition_path"],
        dataset_root=tiny_partition_fixture["dataset_root"],
        expected_clients=2,
        verify_images=False,
    )
    loader = LazyFastDataLoader(
        partition.client_samples[0],
        num_labels=partition.num_classes,
        batch_size=2,
        shuffle=True,
        transform=build_tiny_imagenet_transforms(image_size=32, profile="imagenet", train=True),
        image_size=32,
    )

    assert isinstance(loader, FastDataLoader)
    assert isinstance(loader.tensors[0], LazyImageTensor)
    assert loader.tensors[0].shape == torch.Size((4, 3, 32, 32))
    images, labels = next(iter(loader))
    assert images.shape == (2, 3, 32, 32)
    assert labels.dtype == torch.long

    shuffled = loader.tensors[0][torch.randperm(loader.size)]
    assert isinstance(shuffled, LazyImageTensor)
    class_subset = loader.tensors[0][loader.tensors[1] == 0]
    assert isinstance(class_subset, torch.Tensor)
    assert class_subset.shape[1:] == (3, 32, 32)

    torch_loader = loader.as_dataloader(num_workers=0)
    torch_images, torch_labels = next(iter(torch_loader))
    assert torch_images.shape == (2, 3, 32, 32)
    assert torch_labels.shape == (2,)

    sampled = LazyFastDataLoader(
        partition.client_samples[0],
        num_labels=partition.num_classes,
        batch_size=2,
        percentage=0.5,
        transform=build_tiny_imagenet_transforms(image_size=32, profile="imagenet", train=True),
        image_size=32,
    )
    assert sampled.clone().percentage == pytest.approx(0.5)
    assert sampled.clone().size == sampled.size
    sampled_batches = list(sampled.as_dataloader(num_workers=0))
    assert sum(batch_labels.numel() for _, batch_labels in sampled_batches) == sampled.size
    with pytest.raises(IndexError):
        _ = sampled[sampled.size]

    worker_loader = LazyFastDataLoader(
        partition.client_samples[0],
        num_labels=partition.num_classes,
        batch_size=2,
        shuffle=True,
        transform=build_tiny_imagenet_transforms(image_size=64, profile="imagenet", train=True),
        image_size=64,
        num_workers=1,
        pin_memory=False,
    )
    worker_images, worker_labels = next(iter(worker_loader))
    assert worker_images.shape == (2, 3, 64, 64)
    assert worker_labels.shape == (2,)


@pytest.mark.unit
def test_native_tiny_imagenet_transform_keeps_64px_inputs_and_augmentation():
    train_transform = build_tiny_imagenet_transforms(
        image_size=64,
        profile="imagenet",
        train=True,
    )
    validation_transform = build_tiny_imagenet_transforms(
        image_size=64,
        profile="imagenet",
        train=False,
    )

    assert not any(isinstance(item, transforms.Resize) for item in train_transform.transforms)
    assert not any(isinstance(item, transforms.Resize) for item in validation_transform.transforms)
    crop = next(item for item in train_transform.transforms if isinstance(item, transforms.RandomCrop))
    assert crop.padding == 4
    assert any(isinstance(item, transforms.ColorJitter) for item in train_transform.transforms)
    assert train_transform(Image.new("RGB", (64, 64))).shape == (3, 64, 64)
    assert validation_transform(Image.new("RGB", (64, 64))).shape == (3, 64, 64)


@pytest.mark.unit
def test_round_progress_is_written_after_client_and_round_events(tmp_path: Path):
    progress_path = tmp_path / "progress.json"
    observer = RoundMetricsObserver(progress_path=progress_path, total_rounds=3)
    observer.start_round(round=1, global_model=None)
    observer.start_fit(round=1, client_id=4, model=None)
    observer.end_fit(round=1, client_id=4, model=None, loss=1.0)
    observer.server_evaluation(round=1, eval_type="global", evals={"accuracy": 0.5})
    observer.end_round(round=1)

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "running"
    assert progress["rounds_completed"] == 1
    assert progress["total_rounds"] == 3
    assert progress["clients_completed_in_active_round"] == 1
    assert progress["last_global_metrics"] == {"accuracy": 0.5}
    assert progress["last_round_seconds"] is not None

    observer.finished(round=2)
    assert json.loads(progress_path.read_text(encoding="utf-8"))["status"] == "finished"


@pytest.mark.unit
def test_round_progress_persists_method_diagnostics(tmp_path: Path):
    progress_path = tmp_path / "progress.json"
    observer = RoundMetricsObserver(
        progress_path=progress_path,
        total_rounds=1,
        round_diagnostics_supplier=lambda: {
            "feddyn_dual_norm_before_mean": 1.0,
            "feddyn_dual_norm_after_mean": 2.0,
        },
    )
    observer.start_round(round=1)
    observer.end_round(round=1)

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["last_method_diagnostics"] == {
        "feddyn_dual_norm_before_mean": 1.0,
        "feddyn_dual_norm_after_mean": 2.0,
    }
    assert observer.method_diagnostics == [
        {"round": 1, "metrics": progress["last_method_diagnostics"]}
    ]


@pytest.mark.unit
def test_splitter_gives_every_client_the_same_canonical_validation_loader(tiny_partition_fixture):
    partition = load_tiny_imagenet_partition(
        tiny_partition_fixture["partition_path"],
        dataset_root=tiny_partition_fixture["dataset_root"],
        expected_clients=2,
        verify_images=False,
    )
    splitter = PartitionDataSplitter(
        partition,
        train_transform=build_tiny_imagenet_transforms(image_size=32, profile="imagenet", train=True),
        eval_transform=build_tiny_imagenet_transforms(image_size=32, profile="imagenet", train=False),
        image_size=32,
    )
    (train_loaders, validation_loaders), server_loader = splitter.assign(2, batch_size=2)

    assert [loader.size for loader in train_loaders] == [4, 4]
    assert [loader.size for loader in validation_loaders] == [2, 2]
    assert server_loader.size == 2
    assert validation_loaders[0] is not validation_loaders[1]


@pytest.mark.unit
def test_model_bundle_exposes_encoder_head_and_fedaws_path():
    bundle = build_tiny_imagenet_model_bundle(num_classes=200, normalization="group")
    outputs = bundle.full(torch.randn(2, 3, 32, 32))

    assert outputs.shape == (2, 200)
    assert bundle.embedding_dim == 128
    assert "_head.classifier.weight" in bundle.full.state_dict()
    assert isinstance(TinyImageNetFlukeNet(num_classes=200), type(bundle.full))


@pytest.mark.unit
def test_fluke_resnet18gn_is_a_full_model_only_bundle():
    bundle = build_tiny_imagenet_model_bundle(
        num_classes=200,
        name="fluke_resnet18gn",
    )

    assert bundle.name == "fluke_resnet18gn"
    assert bundle.full(torch.randn(2, 3, 64, 64)).shape == (2, 200)
    assert bundle.embedding_dim is None
    with pytest.raises(ValueError, match="does not expose"):
        bundle.component("encoder")


@pytest.mark.unit
def test_torchvision_resnet18_imagenet_bn_replaces_only_the_classifier_head():
    bundle = build_tiny_imagenet_model_bundle(
        num_classes=200,
        name="torchvision_resnet18_imagenet_bn",
        pretrained=False,
    )

    assert bundle.name == "torchvision_resnet18_imagenet_bn"
    assert isinstance(bundle.full, TorchvisionResNet18ImageNetBN)
    assert bundle.full.resnet.fc.out_features == 200
    assert any(isinstance(module, torch.nn.BatchNorm2d) for module in bundle.full.modules())
    assert bundle.full(torch.randn(2, 3, 64, 64)).shape == (2, 200)
    with pytest.raises(ValueError, match="does not expose"):
        bundle.component("encoder")


@pytest.mark.unit
def test_resnet18gn_preflight_accepts_adam_and_rejects_component_methods(tiny_partition_fixture):
    cell = _cell(tiny_partition_fixture)
    cell["model"]["name"] = "fluke_resnet18gn"
    cell["training"]["optimizer"] = {
        "name": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
    }

    summary = preflight_cell(cell, verify_images=False)
    assert summary["algorithm_key"] == "fedavg"

    cell["algorithm"] = {"key": "fedbabu"}
    with pytest.raises(FlukeEvaluationError, match="full-model-only"):
        preflight_cell(cell, verify_images=False)


@pytest.mark.unit
def test_catalog_covers_the_installed_public_fluke_methods():
    assert not catalog_coverage_gaps()
    assert resolve_algorithm_template("fedavg").class_path == "fluke.algorithms.fedavg.FedAVG"
    assert resolve_algorithm_template(
        "fluke.algorithms.fedavg.FedAVG"
    ).key == "fedavg"
    # The SCAFFOLD server receives normalized client weights; its separate
    # aggregation learning rate handles any intended server-side update scaling.
    assert resolve_algorithm_template("scaffold").server_defaults == {"global_step": 1.0}
    assert resolve_algorithm_template("fedopt_adam").server_defaults["bias_correction"] is True


@pytest.mark.unit
def test_fluke_source_revision_is_optional_but_well_formed():
    revision = _fluke_source_revision()
    assert revision is None or (len(revision) == 40 and all(char in "0123456789abcdef" for char in revision))


@pytest.mark.unit
def test_scaffold_accepts_sgd_momentum_and_rejects_unsupported_optimizer_contracts(
    tiny_partition_fixture,
):
    cell = _cell(tiny_partition_fixture, algorithm="scaffold")
    cell["training"]["optimizer"]["momentum"] = 0.9
    summary = preflight_cell(cell, verify_images=False)
    assert summary["adapter_patches"] == ["scaffold_option_ii_median_steps_weighted_control_v3"]
    assert summary["server_lr"] == 1.0
    assert summary["server_global_step"] == 1.0

    cell = _cell(tiny_partition_fixture, algorithm="scaffold")
    cell["training"]["optimizer"] = {"name": "Adam", "lr": 1e-3}
    with pytest.raises(FlukeEvaluationError, match="Adam and AdamW remain unsupported"):
        preflight_cell(cell, verify_images=False)

    cell = _cell(tiny_partition_fixture, algorithm="scaffold")
    cell["training"]["optimizer"].update({"momentum": 1.0, "dampening": 0.1})
    with pytest.raises(FlukeEvaluationError, match="momentum in \\[0, 1\\)"):
        preflight_cell(cell, verify_images=False)

    cell = _cell(tiny_partition_fixture, algorithm="scaffold")
    cell["training"]["scheduler"]["gamma"] = 0.5
    with pytest.raises(FlukeEvaluationError, match="constant learning rate"):
        preflight_cell(cell, verify_images=False)

    cell = _cell(tiny_partition_fixture, algorithm="scaffold")
    cell["algorithm"]["server"] = {"global_step": 0.5}
    with pytest.raises(FlukeEvaluationError, match="server.global_step=1.0"):
        preflight_cell(cell, verify_images=False)


@pytest.mark.unit
def test_scaffold_momentum_normalization_matches_fednova_effective_steps():
    assert scaffold_momentum_normalization(1, 0.9) == pytest.approx(1.0)
    assert scaffold_momentum_normalization(2, 0.9) == pytest.approx(2.9)
    assert scaffold_momentum_normalization(7, 0.0) == pytest.approx(7.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        scaffold_momentum_normalization(1, 1.0)


@pytest.mark.unit
def test_feddyn_dual_correction_has_the_paper_sign_and_scale():
    previous = torch.tensor([2.0, -3.0])
    global_params = torch.tensor([4.0, 1.0])
    local_params = torch.tensor([5.5, -2.0])

    corrected = corrected_feddyn_dual(previous, global_params, local_params)

    assert torch.equal(corrected, torch.tensor([3.5, -6.0]))


@pytest.mark.unit
def test_fednova_uses_executed_steps_and_an_anchored_delta_update():
    class SingletonSkippingLoader:
        size = 65
        batch_size = 64
        skip_singleton = True

        def __len__(self) -> int:
            return 2

    assert executed_local_steps(SingletonSkippingLoader(), epochs=2) == 2
    coefficient, delta_weights = fednova_delta_weights([0.5, 0.5], [1.0, 3.0])
    assert coefficient == pytest.approx(2.0)
    assert delta_weights == pytest.approx([1.0, 1.0 / 3.0])

    server = torch.nn.Linear(1, 1, bias=False)
    client_a = torch.nn.Linear(1, 1, bias=False)
    client_b = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        server.weight.fill_(10.0)
        client_a.weight.fill_(9.0)
        client_b.weight.fill_(8.0)
    diagnostics = apply_fednova_delta_update(
        server,
        [client_a, client_b],
        [0.5, 0.5],
        [1.0, 3.0],
        eta=1.0,
    )
    # The correction updates 10 + 1 * ((9 - 10) + (8 - 10) / 3), rather than
    # averaging client models with weights that do not sum to one.
    assert server.weight.item() == pytest.approx(10.0 - 1.0 - 2.0 / 3.0)
    assert diagnostics["fednova_delta_weight_sum"] == pytest.approx(4.0 / 3.0)
    with pytest.raises(ValueError, match="non-positive"):
        fednova_delta_weights([1.0], [0.0])
    with pytest.raises(ValueError, match="invalid client"):
        fednova_delta_weights([-1.0], [1.0])


@pytest.mark.unit
def test_scaffold_reset_clears_persistent_sgd_momentum():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    assert zero_sgd_momentum_buffers(optimizer) == 2
    assert all(
        torch.count_nonzero(state["momentum_buffer"]).item() == 0
        for state in optimizer.state.values()
    )


@pytest.mark.unit
def test_compatibility_patches_change_feddyn_run_identity(tiny_partition_fixture):
    cell = _cell(tiny_partition_fixture, algorithm="feddyn")
    template = resolve_algorithm_template("feddyn")
    split_hash = "split-hash"

    patched = _run_id(cell, split_hash, template)
    legacy = _config_hash({"cell": cell, "split_hash": split_hash})[:12]

    assert compatibility_patches("feddyn") == ["feddyn_dual_state_sign_and_scale_v1"]
    assert patched.endswith(_config_hash({
        "cell": cell,
        "split_hash": split_hash,
        "adapter_patches": compatibility_patches("feddyn"),
    })[:12])
    assert not patched.endswith(legacy)


@pytest.mark.unit
def test_parallel_callback_payload_and_round_counter_are_recorded_correctly():
    observer = RoundMetricsObserver()
    observer.client_evaluation(
        round=3,
        client_id=1,
        phase="post_fit",
        results={"accuracy": 0.75},
    )
    parallel_like = types.SimpleNamespace(
        rounds=0,
        server=types.SimpleNamespace(_rounds=3),
    )

    assert observer.client_metrics == [
        {
            "round": 3,
            "client_id": 1,
            "phase": "post_fit",
            "metrics": {"accuracy": 0.75},
        }
    ]
    assert _completed_rounds(parallel_like) == 3


@pytest.mark.unit
def test_parallel_server_evaluation_adapter_accepts_the_inherited_round_argument():
    class ParallelServerLike:
        device = ["cuda:0", "cuda:1"]
        model = object()
        _rounds = 1

        def evaluate(self, evaluator, test_set):
            raise AssertionError("The Fluke 0.8.1 signature should be replaced for this instance.")

    class Evaluator:
        def __init__(self):
            self.calls: list[tuple[int, object, object, str]] = []

        def evaluate(self, round, model, test_set, *, device):
            self.calls.append((round, model, test_set, device))
            return {"accuracy": 0.5}

    server = ParallelServerLike()
    evaluator = Evaluator()
    _adapt_parallel_server_evaluate_signature(server)

    assert server.evaluate(evaluator, "validation", 2) == {"accuracy": 0.5}
    assert evaluator.calls == [(2, server.model, "validation", "cuda:0")]


@pytest.mark.unit
def test_parallel_runtime_cleanup_releases_workers_and_manager():
    class Process:
        def __init__(self, alive: bool):
            self.alive = alive
            self.terminated = False
            self.join_timeout: int | None = None

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True

        def join(self, timeout):
            self.join_timeout = timeout

    class Manager:
        def __init__(self):
            self.stopped = False

        def shutdown(self):
            self.stopped = True

    first, second = Process(alive=True), Process(alive=False)
    manager = Manager()
    algorithm = types.SimpleNamespace(
        server=types.SimpleNamespace(
            running_processes=[(first, "cuda:0"), (second, "cuda:1")],
            manager=manager,
        )
    )

    _close_parallel_runtime(algorithm)

    assert first.terminated is True
    assert second.terminated is False
    assert first.join_timeout == 5
    assert second.join_timeout == 5
    assert manager.stopped is True


@pytest.mark.unit
def test_algorithm_class_path_is_a_first_class_selector(tiny_partition_fixture):
    cell = _cell(tiny_partition_fixture)
    cell["algorithm"] = {"class_path": "fluke.algorithms.fedavg.FedAVG"}

    summary = preflight_cell(cell, verify_images=False)

    assert summary["algorithm"] == "fluke.algorithms.fedavg.FedAVG"
    assert summary["protocol"] == "centralized"


@pytest.mark.unit
def test_supported_custom_fluke_fqcn_uses_the_generic_preflight_contract(tiny_partition_fixture):
    from fluke.algorithms.fedavg import FedAVG

    module_name = "_fluke_adapter_test_custom"
    module = types.ModuleType(module_name)

    class CustomFedAvg(FedAVG):
        pass

    CustomFedAvg.__module__ = module_name
    module.CustomFedAvg = CustomFedAvg
    sys.modules[module_name] = module
    cell = _cell(tiny_partition_fixture)
    cell["algorithm"] = {"class_path": f"{module_name}.CustomFedAvg"}
    try:
        summary = preflight_cell(cell, verify_images=False)
    finally:
        sys.modules.pop(module_name, None)

    assert summary["algorithm"] == f"{module_name}.CustomFedAvg"
    assert summary["algorithm_key"] == "customfedavg"


@pytest.mark.slow
def test_catalog_preflight_instantiates_every_cpu_capable_method(tiny_partition_fixture):
    base = _cell(tiny_partition_fixture)
    failures: list[str] = []
    for key, template in FLUKE_ALGORITHM_CATALOG.items():
        if template.protocol == "parallel":
            continue
        cell = {**base, "algorithm": {"key": key}}
        try:
            summary = preflight_cell(cell, verify_images=False)
            assert summary["n_clients"] == 2
        except Exception as exc:  # noqa: BLE001 - collect every contract gap in one assertion
            failures.append(f"{key}: {type(exc).__name__}: {exc}")
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.parametrize(
    "algorithm",
    ["fedavg", "fedprox", "fednova", "scaffold", "fedlc", "fedld"],
)
def test_lightweight_centralized_methods_complete_one_round(tiny_partition_fixture, algorithm):
    outcome = run_cell(_cell(tiny_partition_fixture, algorithm=algorithm), verify_images=False)

    assert outcome["status"] == "success"
    assert outcome["metrics"]["global_accuracy_by_round"]
    assert outcome["metrics"]["final_accuracy"] is not None


@pytest.mark.slow
def test_run_writes_artifacts_and_matching_success_resumes(tiny_partition_fixture):
    cell = _cell(tiny_partition_fixture, algorithm="fedavg")
    first = run_cell(cell, verify_images=False)
    run_dir = Path(first["run_dir"])
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    assert manifest["extras"]["backend"] == "fluke"
    assert result["result_type"] == "fl_run"
    assert result["status"] == "success"
    assert result["summary"]["result_scope"] == "global_model"
    assert result["summary"]["adapter_patches"] == []
    assert result["summary"]["fluke_version"]
    assert result["summary"]["training_runtime_seconds"] >= 0
    assert result["summary"]["preflight_runtime_seconds"] >= 0
    for name in ("best_model_checkpoint", "last_model_checkpoint"):
        checkpoint_path = Path(result["artifacts"][name])
        assert checkpoint_path.is_file()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["round"] == 1
        assert checkpoint["model_state_dict"]
        assert manifest["artifacts"][name] == str(checkpoint_path)
    assert run_cell(cell, verify_images=False)["status"] == "resumed"


@pytest.mark.slow
def test_sparse_evaluation_still_records_final_global_accuracy(tiny_partition_fixture):
    cell = _cell(tiny_partition_fixture, algorithm="fedavg")
    cell["protocol"]["rounds"] = 2
    cell["run"]["eval_every"] = 3

    outcome = run_cell(cell, verify_images=False)

    assert outcome["metrics"]["final_accuracy"] is not None
    assert outcome["metrics"]["global_accuracy_by_round"][-1]["round"] == 2


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", sorted(key for key in FLUKE_ALGORITHM_CATALOG if key != "parallel"))
def test_all_fluke_methods_complete_one_round_when_explicitly_enabled(
    tiny_partition_fixture, algorithm
):
    if os.environ.get("FDS_RUN_FLUKE_FULL_SMOKE") != "1":
        pytest.skip("Set FDS_RUN_FLUKE_FULL_SMOKE=1 to run the complete Fluke method smoke suite.")
    outcome = run_cell(_cell(tiny_partition_fixture, algorithm=algorithm), verify_images=False)
    assert outcome["status"] == "success"
    template = FLUKE_ALGORITHM_CATALOG[algorithm]
    if template.result_scope == "global_model":
        assert outcome["metrics"]["final_accuracy"] is not None
    else:
        assert outcome["metrics"]["personalized_macro_accuracy"] is not None


@pytest.mark.gpu
@pytest.mark.slow
def test_parallel_fluke_algorithm_requires_two_cuda_devices(tiny_partition_fixture):
    if os.environ.get("FDS_RUN_FLUKE_FULL_SMOKE") != "1":
        pytest.skip("Set FDS_RUN_FLUKE_FULL_SMOKE=1 on a multi-GPU host.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("ParallelAlgorithm requires at least two CUDA devices.")
    cell = _cell(tiny_partition_fixture, algorithm="parallel")
    cell["run"].pop("device")
    outcome = run_cell(cell, verify_images=False)
    assert outcome["status"] == "success"


@pytest.mark.unit
def test_parallel_method_fails_early_on_a_non_multi_gpu_host(tiny_partition_fixture):
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        pytest.skip("This guard only applies to hosts without multiple CUDA devices.")
    with pytest.raises(FlukeEvaluationError, match="at least two CUDA devices"):
        preflight_cell(_cell(tiny_partition_fixture, algorithm="parallel"), verify_images=False)


@pytest.mark.gpu
@pytest.mark.unit
def test_parallel_method_rejects_a_non_parallel_device_configuration(tiny_partition_fixture):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("This contract requires a multi-GPU host.")
    cell = _cell(tiny_partition_fixture, algorithm="parallel")
    with pytest.raises(FlukeEvaluationError, match="run.device to be a list"):
        preflight_cell(cell, verify_images=False)


@pytest.mark.unit
def test_retry_archives_a_previous_failure_artifact(tiny_partition_fixture):
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        pytest.skip("This failure-path test needs a host without two CUDA devices.")
    cell = _cell(tiny_partition_fixture, algorithm="parallel")
    with pytest.raises(FlukeEvaluationError):
        run_cell(cell, verify_images=False)
    with pytest.raises(FlukeEvaluationError):
        run_cell(cell, verify_images=False)

    result_paths = list(
        Path(cell["run"]["output_root"]).rglob("failed_attempts/*/result.json")
    )
    assert len(result_paths) == 1
    archived = json.loads(result_paths[0].read_text(encoding="utf-8"))
    assert archived["status"] == "failed"


@pytest.mark.unit
def test_invalid_algorithm_configuration_writes_failure_artifacts(tiny_partition_fixture):
    cell = _cell(tiny_partition_fixture)
    cell["algorithm"] = {"class_path": "fluke.algorithms.not_a_real_method.Missing"}

    with pytest.raises(Exception):
        run_cell(cell, verify_images=False)

    result_paths = list(Path(cell["run"]["output_root"]).rglob("result.json"))
    assert len(result_paths) == 1
    result = json.loads(result_paths[0].read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "configuration"
    assert "not_a_real_method" in result["summary"]["algorithm"]


@pytest.mark.unit
def test_matrix_expansion_keeps_e_as_an_explicit_cell_axis(tiny_partition_fixture):
    config = {
        "partitions": [{"id": "one", "path": str(tiny_partition_fixture["partition_path"])}],
        "algorithms": ["fedavg", "fedprox"],
        "protocol": {"n_clients": 2, "local_epochs": 1},
        "matrix": {"local_epochs": [1, 5]},
    }
    cells = resolve_matrix_cells(config)
    assert len(cells) == 4
    assert {cell["protocol"]["local_epochs"] for cell in cells} == {1, 5}


SHIPPED_CONFIG_MODELS = {
    "tiny_imagenet_resnet18gn.yaml": {
        "name": "fluke_resnet18gn",
        "image_size": 64,
        "normalization": "group",
    },
    "tiny_imagenet_resnet18_imagenet_bn.yaml": {
        "name": "torchvision_resnet18_imagenet_bn",
        "image_size": 64,
        "normalization": "batch",
    },
}


@pytest.mark.unit
@pytest.mark.parametrize("config_name", sorted(SHIPPED_CONFIG_MODELS))
def test_shipped_configs_expand_to_seven_methods_on_three_partitions(monkeypatch, config_name):
    monkeypatch.setenv("FDS_RUNS_ROOT", "/tmp/fds-runs")
    monkeypatch.setenv("FDS_DATASETS_ROOT", "/tmp/fds-datasets")
    monkeypatch.setenv("FDS_SPLITS_ROOT", "/tmp/fds-splits")
    monkeypatch.setenv("FDS_FLUKE_SEED", "7")
    config_path = (
        Path(__file__).resolve().parents[1]
        / "fl_training"
        / "fluke_downstream_fl_evaluation"
        / "configs"
        / config_name
    )

    config = load_config(config_path)
    cells = resolve_matrix_cells(config)

    assert config["model"] == SHIPPED_CONFIG_MODELS[config_name]
    assert [entry["key"] for entry in config["algorithms"]] == [
        "fedavg",
        "feddyn",
        "fednova",
        "fedprox",
        "fedsam",
        "scaffold",
        "fedlc",
    ]
    assert [partition["id"] for partition in config["partitions"]] == [
        "dirichlet_alpha_0.1",
        "geometric",
        "semantic_alpha_0.1",
    ]
    assert all(
        partition["path"].startswith("/tmp/fds-splits/tiny_imagenet/")
        for partition in config["partitions"]
    )
    assert int(config["run"]["seed"]) == 7
    assert len(cells) == 21
    assert {cell["protocol"]["local_epochs"] for cell in cells} == {1}


def test_scaffold_steps_per_epoch_is_the_rounded_median_epoch_length():
    # Batches per epoch of the ten clients of the Tiny-ImageNet geometric partition at batch 64.
    assert scaffold_steps_per_epoch([280, 213, 355, 112, 110, 259, 128, 23, 8, 80]) == 120
    assert scaffold_steps_per_epoch([3, 1, 2]) == 2
    assert scaffold_steps_per_epoch([1, 2]) == 2
    with pytest.raises(ValueError):
        scaffold_steps_per_epoch([])


def test_scaffold_control_weights_follow_samples_times_effective_steps():
    assert scaffold_control_weights({0: 100, 1: 300}, {0: 5.0, 1: 5.0}) == pytest.approx(
        {0: 0.25, 1: 0.75}
    )
    assert scaffold_control_weights({0: 100, 1: 100}, {0: 1.0, 1: 3.0}) == pytest.approx(
        {0: 0.25, 1: 0.75}
    )
