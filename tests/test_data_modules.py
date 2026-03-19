"""
Unit tests for data loading modules and datasets.

Tests verify correct behavior of dataset loaders, data modules, registry pattern,
transforms, and error handling.
"""

import pytest
import json
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch
import torch
from torchvision import transforms as T

from data_loading.dataset import (
    ImageDataset,
    ClassificationDataset,
    SegmentationDataset,
    BaseDataset,
)
from data_loading.data_modules import (
    BaseDataModule,
    TinyImageNetDataModule,
    Food101DataModule,
    ADE20KDataModule,
    get_datamodule,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary test image."""
    img = Image.new('RGB', (64, 64), color=(255, 0, 0))
    img_path = tmp_path / "test_img.jpg"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def temp_images(tmp_path):
    """Create multiple temporary test images."""
    image_paths = []
    for i in range(5):
        img = Image.new('RGB', (64, 64), color=(i * 50, i * 50, i * 50))
        img_path = tmp_path / f"img_{i}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))
    return image_paths


@pytest.fixture
def corrupt_image_file(tmp_path):
    """Create a corrupt image file."""
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_text("This is not a valid image")
    return str(corrupt_path)


@pytest.fixture
def temp_mask_files(tmp_path):
    """Create temporary mask files for segmentation."""
    mask_paths = []
    for i in range(3):
        mask = Image.new('L', (64, 64), color=i * 50)
        mask_path = tmp_path / f"mask_{i}.png"
        mask.save(mask_path)
        mask_paths.append(str(mask_path))
    return mask_paths


@pytest.fixture
def mock_split_file(tmp_path):
    """Create a mock split file for testing."""
    mock_splits = {
        "clients": {
            "client_0": {
                "train": [
                    {"image": "/fake/path/img1.jpg", "label": "class_a"},
                    {"image": "/fake/path/img2.jpg", "label": "class_b"},
                ],
                "val": [
                    {"image": "/fake/path/img3.jpg", "label": "class_a"},
                ],
            },
            "client_1": {
                "train": [
                    {"image": "/fake/path/img4.jpg", "label": "class_b"},
                    {"image": "/fake/path/img5.jpg", "label": "class_a"},
                ],
                "val": [],
            },
        },
        "global_val": [
            {"image": "/fake/path/val1.jpg", "label": "class_a"},
            {"image": "/fake/path/val2.jpg", "label": "class_b"},
        ],
    }

    split_file = tmp_path / "splits.json"
    with open(split_file, 'w') as f:
        json.dump(mock_splits, f)

    return str(split_file)

@pytest.fixture
def mock_label_map():
    """Create a mock label map for testing."""
    return {"class_a": 0, "class_b": 1}



@pytest.fixture
def simple_transform():
    """Create a simple transform for testing."""
    return T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])


# ============================================================================
# Test ImageDataset
# ============================================================================

class TestImageDataset:
    """Test ImageDataset class."""

    def test_basic_loading(self, temp_images):
        """Test basic image loading."""
        file_dataset = [{"image": path} for path in temp_images]
        dataset = ImageDataset(file_dataset)

        assert len(dataset) == 5

        # Load first image
        img = dataset[0]
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_with_transform(self, temp_images, simple_transform):
        """Test loading with transforms."""
        file_dataset = [{"image": path} for path in temp_images]
        dataset = ImageDataset(file_dataset, transform=simple_transform)

        img_tensor = dataset[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 32, 32)

    def test_with_processor(self, temp_images):
        """Test loading with processor (Hugging Face style)."""
        file_dataset = [{"image": path} for path in temp_images[:2]]

        # Mock processor
        mock_processor = Mock()
        mock_processor.return_value = Mock(
            pixel_values=torch.randn(1, 3, 224, 224)
        )

        dataset = ImageDataset(file_dataset, processor=mock_processor)
        result = dataset[0]

        assert isinstance(result, torch.Tensor)
        assert mock_processor.called

    def test_transform_and_processor_error(self, temp_images):
        """Test that providing both transform and processor raises error."""
        file_dataset = [{"image": temp_images[0]}]

        with pytest.raises(ValueError, match="Provide either transform or processor"):
            ImageDataset(
                file_dataset,
                transform=simple_transform,
                processor=Mock()
            )

    def test_corrupt_image_handling(self, corrupt_image_file):
        """Test that corrupt images are handled gracefully."""
        file_dataset = [{"image": corrupt_image_file}]
        dataset = ImageDataset(file_dataset)

        with pytest.warns(RuntimeWarning, match="Skipping unreadable image"):
            result = dataset[0]

        # Should return None for corrupt images
        assert result is None

    def test_missing_image_handling(self):
        """Test that missing images are handled gracefully."""
        file_dataset = [{"image": "/nonexistent/path.jpg"}]
        dataset = ImageDataset(file_dataset)

        with pytest.warns(RuntimeWarning, match="Skipping unreadable image"):
            result = dataset[0]

        # Should return None for missing images
        assert result is None


# ============================================================================
# Test ClassificationDataset
# ============================================================================

class TestClassificationDataset:
    """Test ClassificationDataset class."""
    def __init__(self):
        self.label_map = {"cat": 0, "dog": 1}

    def test_basic_loading(self, temp_images):
        """Test basic image-label loading."""
        split_data = [
            {"image": temp_images[0], "label": "cat"},
            {"image": temp_images[1], "label": "dog"},
            {"image": temp_images[2], "label": "cat"},
        ]
        dataset = ClassificationDataset(split_data, label_map=self.label_map)

        assert len(dataset) == 3

        img, label = dataset[0]
        assert isinstance(img, Image.Image)
        assert label == 0  # Mapped label

    def test_with_transform(self, temp_images, simple_transform):
        """Test loading with transforms."""
        split_data = [
            {"image": temp_images[0], "label": "cat"},
            {"image": temp_images[1], "label": "dog"},
        ]
        dataset = ClassificationDataset(split_data, label_map=self.label_map, transform=simple_transform)

        img_tensor, label = dataset[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 32, 32)
        assert label == 0


class _DummyDataModule(BaseDataModule):
    def get_train_transform(self):
        return None

    def get_val_transform(self):
        return None

    def setup(self):
        self.train_dataset = [0, 1, 2]
        self.val_dataset = [0]


def test_train_dataloader_respects_train_drop_last():
    dm = _DummyDataModule(
        data_dir=".",
        split_file="/tmp/dummy.json",
        batch_size=2,
        num_workers=0,
        train_drop_last=True,
    )
    dm.setup()

    loader = dm.train_dataloader

    assert loader.drop_last is True


# ============================================================================
# Test SegmentationDataset
# ============================================================================

class TestSegmentationDataset:
    """Test SegmentationDataset class."""

    def test_basic_loading(self, temp_images, temp_mask_files):
        """Test basic image-mask loading."""
        split_data = [
            {"image": temp_images[0], "mask_path": temp_mask_files[0]},
            {"image": temp_images[1], "mask_path": temp_mask_files[1]},
        ]
        dataset = SegmentationDataset(split_data)

        assert len(dataset) == 2

        img, mask = dataset[0]
        assert isinstance(img, Image.Image)
        assert isinstance(mask, Image.Image)

    def test_with_transform(self, temp_images, temp_mask_files):
        """Test loading with synchronized transforms."""
        split_data = [
            {"image": temp_images[0], "mask_path": temp_mask_files[0]},
        ]

        # Mock transform that processes both image and mask
        def mock_transform(img):
            return T.ToTensor()(img)
        
        def mock_mask_transform(mask):
            return T.ToTensor()(mask)

        dataset = SegmentationDataset(split_data, transform=mock_transform, mask_transform=mock_mask_transform)

        img_tensor, mask_tensor = dataset[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert isinstance(mask_tensor, torch.Tensor)

    def test_rgb_ade_mask_is_decoded_to_single_channel_indices(self, tmp_path):
        """ADE-style RGB *_seg masks are decoded from channel-0 into a single channel."""
        img_path = tmp_path / "img.jpg"
        mask_path = tmp_path / "sample_seg.png"

        Image.new("RGB", (2, 1), color=(10, 20, 30)).save(img_path)
        ade_mask = Image.new("RGB", (2, 1))
        ade_mask.putpixel((0, 0), (7, 5, 0))
        ade_mask.putpixel((1, 0), (120, 7, 255))
        ade_mask.save(mask_path)

        dataset = SegmentationDataset(
            split_data=[{"image": str(img_path), "mask_path": str(mask_path)}],
            transform=T.PILToTensor(),
            mask_transform=T.PILToTensor(),
        )

        _, mask_tensor = dataset[0]
        assert mask_tensor.shape == (1, 1, 2)
        assert int(mask_tensor[0, 0, 0]) == 7
        assert int(mask_tensor[0, 0, 1]) == 120

    def test_accepts_legacy_label_key(self, temp_images, temp_mask_files):
        """Legacy split schema with 'label' mask paths remains supported."""
        split_data = [
            {"image": temp_images[0], "label": temp_mask_files[0]},
        ]
        dataset = SegmentationDataset(split_data)

        img, mask = dataset[0]
        assert isinstance(img, Image.Image)
        assert isinstance(mask, Image.Image)

    def test_sync_horizontal_flip_keeps_image_mask_aligned(self, tmp_path):
        """Synchronized horizontal flip must be applied to both image and mask."""
        img_path = tmp_path / "img.png"
        mask_path = tmp_path / "mask.png"

        image = Image.new("RGB", (2, 1))
        image.putpixel((0, 0), (255, 0, 0))
        image.putpixel((1, 0), (0, 0, 255))
        image.save(img_path)

        mask = Image.new("L", (2, 1))
        mask.putpixel((0, 0), 0)
        mask.putpixel((1, 0), 255)
        mask.save(mask_path)

        dataset = SegmentationDataset(
            split_data=[{"image": str(img_path), "mask_path": str(mask_path)}],
            transform=T.PILToTensor(),
            mask_transform=T.PILToTensor(),
            sync_hflip_prob=1.0,
        )

        image_tensor, mask_tensor = dataset[0]

        # After deterministic flip (p=1), left pixel should be original right pixel.
        assert image_tensor[:, 0, 0].tolist() == [0, 0, 255]
        assert int(mask_tensor[0, 0, 0]) == 255


# ============================================================================
# Test BaseDataset Registry Pattern
# ============================================================================

class TestBaseDatasetRegistry:
    """Test BaseDataset registry pattern."""

    def test_register_custom_dataset(self, tmp_path):
        """Test registering a custom dataset loader."""
        # Create test data structure
        data_dir = tmp_path / "custom_dataset"
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)

        for i in range(3):
            img = Image.new('RGB', (32, 32), color=(i * 80, 0, 0))
            img.save(train_dir / f"img_{i}.jpg")

        # Register custom loader
        @BaseDataset.register_dataset("test_custom")
        def load_custom(self):
            train_path = self.base_path / "custom_dataset" / self.split
            for img_file in sorted(train_path.glob("*.jpg")):
                self.data.append(str(img_file))
                self.labels.append(img_file.stem)

        # Load dataset
        dataset = BaseDataset("test_custom", str(tmp_path), "train")

        assert len(dataset) == 3
        assert "test_custom" in BaseDataset._loaders

    def test_unknown_dataset_error(self, tmp_path):
        """Test that unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            BaseDataset("nonexistent_dataset", str(tmp_path), "train")

    def test_missing_base_path_error(self):
        """Test that missing base path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            BaseDataset("tiny_imagenet", "/nonexistent/path", "train")

    def test_pascal_voc_alias_normalization(self, tmp_path):
        """Dataset aliases for Pascal VOC should normalize to canonical registry key."""
        images_dir = tmp_path / "VOC2012" / "images" / "train"
        labels_dir = tmp_path / "VOC2012" / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        image_path = images_dir / "sample_0.jpg"
        Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_path)
        (labels_dir / "sample_0.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")

        dataset = BaseDataset("pascal_voc", str(tmp_path), "train")
        assert dataset.dataset_name == "PascalVOC"
        assert len(dataset) == 1

    def test_bdd100k_alias_normalization(self, tmp_path):
        """Dataset aliases for BDD100K should normalize to canonical registry key."""
        images_dir = tmp_path / "bdd100k" / "images" / "train"
        masks_dir = tmp_path / "bdd100k" / "segmentation" / "train"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        image_path = images_dir / "sample_0.jpg"
        Image.new("RGB", (16, 16), color=(0, 0, 0)).save(image_path)
        Image.new("L", (16, 16), color=1).save(masks_dir / "sample_0.png")

        dataset = BaseDataset("bdd100k", str(tmp_path), "train")
        assert dataset.dataset_name == "BDD100K"
        assert len(dataset) == 1

    def test_food101_missing_image_raises(self, tmp_path):
        """Food101 loader fails fast when metadata references missing images."""
        food101_root = tmp_path / "food101"
        meta_dir = food101_root / "meta"
        images_dir = food101_root / "images"
        meta_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)

        split_file = meta_dir / "train.json"
        split_payload = {"class_a": ["missing_img"]}
        with open(split_file, "w", encoding="utf-8") as handle:
            json.dump(split_payload, handle)

        with pytest.raises(FileNotFoundError, match="Food-101 image"):
            BaseDataset("food101", str(tmp_path), "train")

    def test_food101_loader_accepts_hyphenated_directory_name(self, tmp_path):
        """Food101 loader supports the canonical food-101 directory layout."""
        food101_root = tmp_path / "food-101"
        meta_dir = food101_root / "meta"
        images_dir = food101_root / "images" / "class_a"
        meta_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)

        split_file = meta_dir / "train.json"
        with open(split_file, "w", encoding="utf-8") as handle:
            json.dump({"class_a": ["class_a/sample_0"]}, handle)

        image_path = images_dir / "sample_0.jpg"
        Image.new("RGB", (16, 16), color=(0, 0, 0)).save(image_path)

        dataset = BaseDataset("food101", str(tmp_path), "train")
        assert len(dataset) == 1
        assert Path(dataset.data[0]) == image_path

    def test_get_summary(self, tmp_path):
        """Test dataset summary generation."""
        # Create mock dataset
        data_dir = tmp_path / "mock_data"
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)

        for i in range(5):
            img = Image.new('RGB', (32, 32))
            img.save(train_dir / f"img_{i}.jpg")

        # Register and load
        @BaseDataset.register_dataset("test_summary")
        def load_test(self):
            train_path = self.base_path / "mock_data" / self.split
            for img_file in sorted(train_path.glob("*.jpg")):
                self.data.append(str(img_file))
                # Assign labels with 2 unique classes
                self.labels.append("class_a" if int(img_file.stem.split("_")[1]) % 2 == 0 else "class_b")

        dataset = BaseDataset("test_summary", str(tmp_path), "train")
        summary = dataset.get_summary()

        assert summary["dataset_name"] == "test_summary"
        assert summary["split"] == "train"
        assert summary["total_samples"] == 5
        assert summary["unique_labels"] == 2
        assert str(tmp_path) in summary["base_path"]

    def test_getitem(self, tmp_path):
        """Test __getitem__ returns path and label."""
        data_dir = tmp_path / "test_data"
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)

        img = Image.new('RGB', (32, 32))
        img_path = train_dir / "test.jpg"
        img.save(img_path)

        @BaseDataset.register_dataset("test_getitem")
        def load_test(self):
            train_path = self.base_path / "test_data" / self.split
            for img_file in sorted(train_path.glob("*.jpg")):
                self.data.append(str(img_file))
                self.labels.append("test_label")

        dataset = BaseDataset("test_getitem", str(tmp_path), "train")

        path, label = dataset[0]
        assert isinstance(path, str)
        assert isinstance(label, str)
        assert label == "test_label"


# ============================================================================
# Test BaseDataModule
# ============================================================================

class TestBaseDataModule:
    """Test BaseDataModule class."""

    def test_load_splits(self, mock_split_file, mock_label_map):
        """Test loading splits from JSON file."""
        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file=mock_split_file,
            label_map=mock_label_map,
            client_idx=0,
            batch_size=2,
        )

        splits = dm.load_splits()

        assert "clients" in splits
        assert "global_val" in splits
        assert len(splits["clients"]) == 2

    def test_get_client_data_federated(self, mock_split_file, mock_label_map):
        """Test getting client data in federated mode."""
        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file=mock_split_file,
            label_map=mock_label_map,
            client_idx=0,
            central=False,
            batch_size=2,
        )

        splits = dm.load_splits()
        train_data, val_data = dm.get_client_data(splits)

        assert len(train_data) == 2  # client_0 has 2 train samples
        assert len(val_data) == 2     # global_val has 2 samples

    def test_get_client_data_central(self, mock_split_file, mock_label_map):
        """Test getting data in central mode (all clients)."""
        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file=mock_split_file,
            label_map=mock_label_map,
            central=True,
            batch_size=2,
        )

        splits = dm.load_splits()
        train_data, val_data = dm.get_client_data(splits)

        # Should aggregate all clients
        assert len(train_data) == 4  # 2 + 2 from both clients
        assert len(val_data) == 2    # Using global_val

    def test_repr_federated(self, mock_split_file, mock_label_map):
        """Test string representation in federated mode."""
        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file=mock_split_file,
            label_map=mock_label_map,
            client_idx=5,
            central=False,
            batch_size=64,
        )

        repr_str = str(dm)
        assert "Client 5" in repr_str
        assert "batch_size=64" in repr_str

    def test_repr_central(self, mock_split_file, mock_label_map):
        """Test string representation in central mode."""
        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file=mock_split_file,
            label_map=mock_label_map,
            central=True,
            batch_size=128,
        )

        repr_str = str(dm)
        assert "Central" in repr_str
        assert "batch_size=128" in repr_str

    def test_tiny_imagenet_transforms_resize_to_224(self):
        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file="unused.json",
            client_idx=0,
        )
        image = Image.new("RGB", (64, 64), color=(64, 128, 192))

        train_tensor = dm.get_train_transform()(image)
        val_tensor = dm.get_val_transform()(image)

        assert tuple(train_tensor.shape) == (3, 224, 224)
        assert tuple(val_tensor.shape) == (3, 224, 224)

    def test_segmentation_datamodule_normalizes_legacy_label_schema(self, tmp_path):
        """ADE20KDataModule converts legacy label masks into canonical mask_path."""
        train_image = tmp_path / "train_img.jpg"
        train_mask = tmp_path / "train_mask.png"
        val_image = tmp_path / "val_img.jpg"
        val_mask = tmp_path / "val_mask.png"

        Image.new("RGB", (32, 32), color=(0, 128, 255)).save(train_image)
        Image.new("L", (32, 32), color=10).save(train_mask)
        Image.new("RGB", (32, 32), color=(255, 128, 0)).save(val_image)
        Image.new("L", (32, 32), color=20).save(val_mask)

        split_file = tmp_path / "seg_splits.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "clients": {
                        "client_0": {
                            "train": [{"image": str(train_image), "label": str(train_mask)}],
                            "val": [],
                        }
                    },
                    "global_val": [{"image": str(val_image), "label": str(val_mask)}],
                },
                f,
            )

        dm = ADE20KDataModule(
            data_dir=str(tmp_path),
            split_file=str(split_file),
            client_idx=0,
            batch_size=1,
            num_workers=0,
            image_size=32,
        )
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.train_dataset.samples[0]["mask_path"] == str(train_mask)
        assert dm.val_dataset.samples[0]["mask_path"] == str(val_mask)

    def test_basedataset_ade20k_root_data_dir_works(self, tmp_path):
        ade_root = tmp_path / "ADE20K" / "ADE20K_2021_17_01" / "images" / "ADE"
        train_dir = ade_root / "training" / "unit"
        val_dir = ade_root / "validation" / "unit"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        train_img = train_dir / "ADE_train_00000001.jpg"
        train_mask = train_dir / "ADE_train_00000001_seg.png"
        val_img = val_dir / "ADE_val_00000001.jpg"
        val_mask = val_dir / "ADE_val_00000001_seg.png"
        Image.new("RGB", (16, 16), color=(1, 2, 3)).save(train_img)
        Image.new("L", (16, 16), color=1).save(train_mask)
        Image.new("RGB", (16, 16), color=(4, 5, 6)).save(val_img)
        Image.new("L", (16, 16), color=2).save(val_mask)

        train_ds = BaseDataset("ade20k", str(tmp_path), "train")
        val_ds = BaseDataset("ade20k", str(tmp_path), "val")
        assert len(train_ds) == 1
        assert len(val_ds) == 1

    def test_basedataset_ade20k_nested_data_dir_fails_fast(self, tmp_path):
        (tmp_path / "ade20k").mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="Images directory .*ADE20K_2021_17_01/images/ADE/training"):
            BaseDataset("ade20k", str(tmp_path / "ade20k"), "train")


# ============================================================================
# Test Factory Function
# ============================================================================

class TestFactoryFunction:
    """Test get_datamodule factory function."""

    def test_create_tiny_imagenet(self, mock_split_file, mock_label_map):
        """Test creating TinyImageNet data module."""
        dm = get_datamodule(
            dataset_name="tiny_imagenet",
            data_dir="./data",
            split_file=mock_split_file,
            label_map=mock_label_map,
            client_idx=0,
            image_size=64,
            batch_size=32,
        )

        assert isinstance(dm, TinyImageNetDataModule)

    def test_create_food101(self, mock_split_file):
        """Test creating Food101 data module."""
        dm = get_datamodule(
            dataset_name="food101",
            data_dir="./data",
            split_file=mock_split_file,
            label_map=None,
            client_idx=0,
            image_size=128,
            batch_size=32,
        )

        assert isinstance(dm, Food101DataModule)

    def test_create_ade20k(self, mock_split_file):
        """Test creating ADE20K data module."""
        dm = get_datamodule(
            dataset_name="ade20k",
            data_dir="./data",
            split_file=mock_split_file,
            label_map=None,
            client_idx=0,
            image_size=256,
            batch_size=32,
        )

        assert isinstance(dm, ADE20KDataModule)

    def test_unknown_dataset_error(self, mock_split_file):
        """Test that unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_datamodule(
                dataset_name="nonexistent",
                data_dir="./data",
                split_file=mock_split_file,
                client_idx=0,
            )


# ============================================================================
# Test Transform Pipelines
# ============================================================================

class TestTransformPipelines:
    """Test transform pipeline functionality."""

    def test_transforms_applied_correctly(self, temp_images):
        """Test that transforms are applied correctly."""
        # Create custom transform
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        split_data = [{"image": temp_images[0], "label": "test"}]
        dataset = ClassificationDataset(split_data, transform=transform, label_map={"test": 0})

        img_tensor, _ = dataset[0]

        # Check shape after transform
        assert img_tensor.shape == (3, 224, 224)

        # Check normalization applied (values should not be in [0, 1])
        assert img_tensor.min() < 0 or img_tensor.max() > 1

    def test_no_transform(self, temp_images):
        """Test loading without transform returns PIL Image."""
        split_data = [{"image": temp_images[0], "label": "test"}]
        dataset = ClassificationDataset(split_data, transform=None, label_map={"test": 0})

        img, _ = dataset[0]

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_multiple_transform_types(self, temp_images):
        """Test different transform configurations."""
        split_data = [{"image": temp_images[0], "label": "test"}]

        # Resize only
        transform1 = T.Resize((128, 128))
        dataset1 = ClassificationDataset(split_data, transform=transform1, label_map={"test": 0})
        img1, _ = dataset1[0]
        assert isinstance(img1, Image.Image)
        assert img1.size == (128, 128)

        # Full pipeline
        transform2 = T.Compose([
            T.Resize((64, 64)),
            T.RandomHorizontalFlip(p=0),  # p=0 for deterministic testing
            T.ToTensor(),
        ])
        dataset2 = ClassificationDataset(split_data, transform=transform2, label_map={"test": 0})
        img2, _ = dataset2[0]
        assert isinstance(img2, torch.Tensor)
        assert img2.shape == (3, 64, 64)


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_missing_image_file(self):
        """Test handling of missing image files."""
        split_data = [{"image": "/nonexistent/image.jpg", "label": "test"}]
        dataset = ClassificationDataset(split_data)

        # Should raise error when trying to load
        with pytest.raises(FileNotFoundError):
            _ = dataset[0]

    def test_corrupt_image_in_imagedataset(self, corrupt_image_file):
        """Test that ImageDataset handles corrupt images gracefully."""
        file_dataset = [{"image": corrupt_image_file}]
        dataset = ImageDataset(file_dataset)

        with pytest.warns(RuntimeWarning, match="Skipping unreadable image"):
            result = dataset[0]

        # Should return None
        assert result is None

    def test_mixed_valid_and_corrupt_images(self, temp_images, corrupt_image_file):
        """Test dataset with mix of valid and corrupt images."""
        file_dataset = [
            {"image": temp_images[0]},
            {"image": corrupt_image_file},
            {"image": temp_images[1]},
        ]
        dataset = ImageDataset(file_dataset)

        # Valid images should load
        assert dataset[0] is not None
        assert dataset[2] is not None

        # Corrupt image should return None
        assert dataset[1] is None

    def test_empty_split_file(self, tmp_path):
        """Test handling of empty split file."""
        empty_splits = {"clients": {}, "global_val": None}

        split_file = tmp_path / "empty.json"
        with open(split_file, 'w') as f:
            json.dump(empty_splits, f)

        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file=str(split_file),
            label_map={},
            central=True,
        )

        splits = dm.load_splits()
        train_data, val_data = dm.get_client_data(splits)

        # Should handle gracefully (return empty lists)
        assert len(train_data) == 0

    def test_missing_split_file(self):
        """Test handling of missing split file."""
        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file="/nonexistent/splits.json",
            label_map={},
            client_idx=0,
        )

        # Should raise error when trying to load
        with pytest.raises(FileNotFoundError):
            dm.load_splits()

    def test_load_splits_rejects_path_only_schema(self, tmp_path):
        """Training datamodules require mapping samples with at least an image key."""
        split_file = tmp_path / "path_only.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "clients": {
                        "client_0": {
                            "train": ["/tmp/img_0.jpg"],
                            "val": [],
                        }
                    },
                    "global_val": None,
                },
                f,
            )

        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file=str(split_file),
            label_map={},
            client_idx=0,
        )

        with pytest.raises(ValueError, match="Invalid split sample schema"):
            dm.load_splits()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
