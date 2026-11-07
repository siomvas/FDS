import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from data_loading.dataset import DetectionDataset
from data_loading.data_modules import PascalVOCDataModule, get_datamodule


def _write_image(path: Path, size=(100, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def _write_annotation(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")


def test_detection_dataset_parses_yolo_txt(tmp_path):
    img_path = tmp_path / "sample.jpg"
    ann_path = tmp_path / "sample.txt"
    _write_image(img_path, size=(100, 80))
    _write_annotation(ann_path, rows=["0 0.5 0.5 0.4 0.5"])

    dataset = DetectionDataset(
        split_data=[{"image": str(img_path), "label": str(ann_path)}],
        transform=T.ToTensor(),
        label_offset=1,
        box_format="auto",
    )

    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [1]

    expected = torch.tensor([30.0, 20.0, 70.0, 60.0], dtype=torch.float32)
    assert torch.allclose(target["boxes"][0], expected)
    assert torch.equal(target["orig_size"], torch.tensor([80, 100], dtype=torch.int64))
    assert torch.equal(target["size"], torch.tensor([80, 100], dtype=torch.int64))


def test_detection_dataset_applies_synchronized_horizontal_flip_to_boxes(tmp_path):
    img_path = tmp_path / "sample.jpg"
    _write_image(img_path, size=(100, 80))

    dataset = DetectionDataset(
        split_data=[
            {
                "image": str(img_path),
                "boxes": [[10.0, 20.0, 40.0, 60.0]],
                "labels": [1],
            }
        ],
        transform=T.ToTensor(),
        label_offset=0,
        sync_hflip_prob=1.0,
    )

    _, target = dataset[0]

    expected = torch.tensor([[60.0, 20.0, 90.0, 60.0]], dtype=torch.float32)
    assert torch.allclose(target["boxes"], expected)
    assert torch.equal(target["size"], torch.tensor([80, 100], dtype=torch.int64))


def test_pascal_voc_datamodule_wires_detection_collate(tmp_path):
    # Build tiny split file with two train samples and one global validation sample.
    train_img_0 = tmp_path / "train_0.jpg"
    train_img_1 = tmp_path / "train_1.jpg"
    val_img = tmp_path / "val_0.jpg"
    train_ann_0 = tmp_path / "train_0.txt"
    train_ann_1 = tmp_path / "train_1.txt"
    val_ann = tmp_path / "val_0.txt"

    for img in (train_img_0, train_img_1, val_img):
        _write_image(img, size=(64, 64))
    _write_annotation(train_ann_0, rows=["1 10 10 30 30"])
    _write_annotation(train_ann_1, rows=["2 15 20 40 50"])
    _write_annotation(val_ann, rows=["3 0.5 0.5 0.2 0.2"])

    split_file = tmp_path / "splits.json"
    splits = {
        "clients": {
            "client_0": {
                "train": [
                    {"image": str(train_img_0), "label": str(train_ann_0)},
                    {"image": str(train_img_1), "label": str(train_ann_1)},
                ],
                "val": [],
            }
        },
        "global_val": [{"image": str(val_img), "label": str(val_ann)}],
    }
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    dm = PascalVOCDataModule(
        data_dir=str(tmp_path),
        split_file=str(split_file),
        client_idx=0,
        batch_size=2,
        num_workers=0,
        detection_label_offset=0,
    )
    dm.setup()

    train_images, train_targets = next(iter(dm.train_dataloader))
    assert isinstance(train_images, list)
    assert isinstance(train_targets, list)
    assert len(train_images) == 2
    assert all(isinstance(img, torch.Tensor) for img in train_images)
    assert all("boxes" in target and "labels" in target for target in train_targets)

    val_images, val_targets = next(iter(dm.val_dataloader))
    assert isinstance(val_images, list)
    assert isinstance(val_targets, list)
    assert len(val_images) == 1


def test_pascal_voc_datamodule_hf_provider_resizes_and_normalizes(tmp_path):
    train_img = tmp_path / "train_0.jpg"
    val_img = tmp_path / "val_0.jpg"
    train_ann = tmp_path / "train_0.txt"
    val_ann = tmp_path / "val_0.txt"

    _write_image(train_img, size=(64, 48))
    _write_image(val_img, size=(80, 40))
    _write_annotation(train_ann, rows=["1 8 12 40 36"])
    _write_annotation(val_ann, rows=["1 10 5 30 20"])

    split_file = tmp_path / "splits.json"
    splits = {
        "clients": {
            "client_0": {
                "train": [{"image": str(train_img), "label": str(train_ann)}],
                "val": [],
            }
        },
        "global_val": [{"image": str(val_img), "label": str(val_ann)}],
    }
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    dm = PascalVOCDataModule(
        data_dir=str(tmp_path),
        split_file=str(split_file),
        client_idx=0,
        batch_size=1,
        num_workers=0,
        image_size=32,
        detection_label_offset=0,
        detection_provider="hf",
        detection_train_hflip_prob=0.0,
    )
    dm.setup()

    train_images, train_targets = next(iter(dm.train_dataloader))
    val_images, val_targets = next(iter(dm.val_dataloader))

    assert train_images[0].shape == (3, 32, 32)
    assert val_images[0].shape == (3, 32, 32)
    assert train_images[0].dtype == torch.float32
    assert val_images[0].dtype == torch.float32
    assert torch.equal(train_targets[0]["orig_size"], torch.tensor([48, 64], dtype=torch.int64))
    assert torch.equal(train_targets[0]["size"], torch.tensor([32, 32], dtype=torch.int64))
    assert torch.equal(val_targets[0]["orig_size"], torch.tensor([40, 80], dtype=torch.int64))
    assert torch.equal(val_targets[0]["size"], torch.tensor([32, 32], dtype=torch.int64))
    assert torch.allclose(train_targets[0]["boxes"], torch.tensor([[4.0, 8.0, 20.0, 24.0]], dtype=torch.float32))
    assert torch.allclose(val_targets[0]["boxes"], torch.tensor([[4.0, 4.0, 12.0, 16.0]], dtype=torch.float32))


def test_get_datamodule_accepts_pascal_voc_alias():
    dm = get_datamodule(
        dataset_name="pascal_voc",
        data_dir="./data",
        split_file="./dummy.json",
        client_idx=0,
    )
    assert isinstance(dm, PascalVOCDataModule)
