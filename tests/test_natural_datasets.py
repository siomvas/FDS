from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_loading.dataset import BaseDataset, ImageDataset


def test_fed_isic2019_loader_parses_manifest_and_client_ids(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for idx in range(3):
        Image.new("RGB", (8, 8), color=(idx * 30, 20, 10)).save(images_dir / f"img_{idx}.jpg")

    manifest = tmp_path / "metadata.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_path,diagnosis,center,split",
                "images/img_0.jpg,melanoma,site_a,train",
                "images/img_1.jpg,nevus,site_b,train",
                "images/img_2.jpg,melanoma,site_a,test",
            ]
        ),
        encoding="utf-8",
    )

    dataset = BaseDataset("fed_isic2019", str(tmp_path), "train")
    assert len(dataset) == 2
    assert dataset[0][1] == "melanoma"
    assert dataset.get_metadata(0)["natural_client_id"] == "site_a"
    assert dataset.get_metadata(1)["natural_client_id"] == "site_b"


def _write_femnist_processed_dataset(root: Path):
    processed_dir = root / "FEMNIST" / "processed"
    processed_dir.mkdir(parents=True)
    train_data = np.stack(
        [
            np.zeros((28, 28), dtype=np.float32),
            np.ones((28, 28), dtype=np.float32),
        ]
    )
    train_targets = np.array([1, 2], dtype=np.int64)
    train_users = ["user_0", "user_1"]
    test_data = np.full((1, 28, 28), 127.0, dtype=np.float32)
    test_targets = np.array([3], dtype=np.int64)
    test_users = ["user_test"]

    torch.save((train_data, train_targets, train_users), processed_dir / "femnist_train.pt")
    torch.save((test_data, test_targets, test_users), processed_dir / "femnist_test.pt")
    torch.save(["user_0", "user_1", "user_test"], processed_dir / "femnist_user_keys.pt")
    return processed_dir


def test_femnist_loader_materializes_deterministic_png_cache(tmp_path):
    _write_femnist_processed_dataset(tmp_path)

    dataset = BaseDataset("femnist", str(tmp_path), "train")
    assert len(dataset) == 2
    first_path, first_label = dataset[0]
    assert first_label == "1"
    assert first_path.startswith("femnist://")
    assert dataset.get_metadata(0)["natural_client_id"] == "user_0"

    dataset_reloaded = BaseDataset("femnist", str(tmp_path), "train")
    assert dataset_reloaded[0][0] == first_path


def test_femnist_loader_uses_test_split_for_val_without_train_leakage(tmp_path):
    _write_femnist_processed_dataset(tmp_path)

    dataset = BaseDataset("femnist", str(tmp_path), "val")

    assert len(dataset) == 1
    assert dataset[0][1] == "3"
    assert dataset.get_metadata(0)["natural_client_id"] == "user_test"
    assert dataset.get_metadata(0)["split"] == "test"


def test_femnist_loader_scales_unit_interval_float_images(tmp_path):
    _write_femnist_processed_dataset(tmp_path)

    dataset = BaseDataset("femnist", str(tmp_path), "train")
    second_path, _ = dataset[1]
    image = ImageDataset([{"image": second_path}])[0]
    array = np.asarray(image.convert("L"))

    assert array.shape == (28, 28)
    assert int(array.max()) == 255


def test_femnist_virtual_image_refs_resolve_lazily(tmp_path):
    _write_femnist_processed_dataset(tmp_path)

    dataset = BaseDataset("femnist", str(tmp_path), "test")
    image_ref, label = dataset[0]
    image = ImageDataset([{"image": image_ref}])[0]

    assert label == "3"
    assert image.size == (28, 28)
    assert image.mode == "RGB"
