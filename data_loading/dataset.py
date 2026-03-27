import csv
import os
import json
import warnings
import logging
import random
import numpy as np
from torch.utils.data import Dataset
import torch
from numbers import Integral
from typing import Callable, List, Tuple, Dict, Any, Optional
from pathlib import Path
from urllib.parse import quote, unquote, urlparse, parse_qs
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)
_FEMNIST_SPLIT_CACHE: Dict[Tuple[str, str], Tuple[Any, Any, Any]] = {}


class ImageDataset(Dataset):
    """Generic image dataset with flexible preprocessing support.

    Loads images from file paths and applies either torchvision transforms or
    Hugging Face processors (for models like CLIP, DINO). Supports graceful
    handling of corrupt images.

    Args:
        file_dataset: List of dicts with 'image' key containing file paths
        transform: Optional torchvision transform pipeline
        processor: Optional Hugging Face processor (mutually exclusive with transform)

    Raises:
        ValueError: If both transform and processor are provided

    Examples:
        >>> from torchvision import transforms as T
        >>> transform = T.Compose([T.Resize(224), T.ToTensor()])
        >>> dataset = ImageDataset(
        ...     file_dataset=[{"image": "img1.jpg"}, {"image": "img2.jpg"}],
        ...     transform=transform
        ... )
        >>> img = dataset[0]  # Returns transformed tensor

    Notes:
        - Images are converted to RGB mode
        - Returns None for corrupt/missing images with a warning
        - Lazy loading: images loaded only when accessed
    """
    def __init__(self, file_dataset, transform: Callable = None, processor: Any = None):
        self.file_dataset = file_dataset
        if transform and processor:
            raise ValueError("Provide either transform or processor, not both.")
        self.transform = transform if transform else None
        self.processor = processor if processor else None

    def __len__(self) -> int:
        return len(self.file_dataset)

    def __getitem__(self, idx: int):
        path = self.file_dataset[idx]["image"]
        try:
            if isinstance(path, str) and path.startswith("femnist://"):
                image = _load_virtual_femnist_image(path).convert("RGB")
            else:
                with Image.open(path) as img:
                    image = img.convert("RGB")
            if self.transform:
                return self.transform(image)
            elif self.processor:
                inputs = self.processor(images=image, return_tensors="pt")
                return inputs.pixel_values.squeeze(0)
            else:
                return image

        except (UnidentifiedImageError, OSError, ValueError, TypeError) as exc:
            warnings.warn(
                f"Skipping unreadable image {path}: {exc}",
                RuntimeWarning,
            )
            return None

class ClassificationDataset(Dataset):
    def __init__(
            self, 
            split_data: List[Dict[str, Any]],
            label_map: Dict[str, int]|None = None,
            transform=None):
        self.samples = split_data
        self.label_map = label_map
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        raw_label = sample["label"]

        if self.label_map:
            label_key = raw_label if raw_label in self.label_map else str(raw_label)
            if label_key not in self.label_map:
                raise KeyError(f"Label '{raw_label}' missing from label_map.")
            label = self.label_map[label_key]
        else:
            if isinstance(raw_label, bool):
                raise ValueError("Boolean labels are not supported for classification.")
            if isinstance(raw_label, Integral):
                label = int(raw_label)
            elif isinstance(raw_label, str) and raw_label.lstrip("-").isdigit():
                label = int(raw_label)
            else:
                raise ValueError(
                    "Non-numeric labels require a label_map. "
                    f"Got label '{raw_label}' of type {type(raw_label).__name__}."
                )

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset for image-mask pairs.

    Loads synchronized image and segmentation mask pairs for pixel-level
    classification tasks. Applies transforms to both images and masks.

    Args:
        split_data: List of dicts with 'image' (path) and either:
            - 'mask_path' (canonical key), or
            - 'label' (legacy compatibility key)
        transform: Optional transform pipeline applied to both image and mask

    Returns:
        Tuple of (image_tensor, mask_tensor)

    Examples:
        >>> dataset = SegmentationDataset(
        ...     split_data=[
        ...         {"image": "img1.jpg", "mask_path": "mask1.png"},
        ...         {"image": "img2.jpg", "mask_path": "mask2.png"}
        ...     ],
        ...     transform=my_transform
        ... )
        >>> image, mask = dataset[0]

    Notes:
        - Transform should handle both images and masks (e.g., synchronized crops)
        - Masks should use nearest-neighbor interpolation for resizing
        - Images are converted to RGB, masks loaded as-is
        - Legacy split files that store segmentation targets under ``label`` are
          accepted for backward compatibility.
    """
    def __init__(
        self,
        split_data: List[Dict[str, Any]],
        transform=None,
        mask_transform=None,
        sync_hflip_prob: float = 0.0,
    ):
        self.samples = split_data  # [{"image": img_path, "label": mask_path}, ...]
        self.transform = transform
        self.mask_transform = mask_transform
        self.sync_hflip_prob = float(sync_hflip_prob)
    
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _decode_rgb_mask(mask: Image.Image, mask_path: str) -> Image.Image:
        """Decode RGB-encoded segmentation masks into a single-channel index map.

        ADE20K ``*_seg.png`` masks in this local layout expose semantic ids in
        channel-0. Preserve that channel directly to avoid creating synthetic
        out-of-range labels.

        For other RGB masks, fall back to channel-0 to enforce the expected
        single-channel mask contract downstream.
        """
        rgb = np.asarray(mask.convert("RGB"), dtype=np.int32)
        if str(mask_path).endswith("_seg.png"):
            return Image.fromarray(rgb[..., 0].astype(np.uint8), mode="L")

        return Image.fromarray(rgb[..., 0].astype(np.uint8), mode="L")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        mask_path = sample.get("mask_path", sample.get("label"))
        if mask_path is None:
            raise KeyError("Segmentation sample must include 'mask_path' or legacy 'label'.")
        mask = Image.open(mask_path)
        if mask.mode in {"RGB", "RGBA"}:
            mask = self._decode_rgb_mask(mask, str(mask_path))

        # Apply synchronized geometric augmentation before individual transforms.
        if self.sync_hflip_prob > 0.0 and random.random() < self.sync_hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


class DetectionDataset(Dataset):
    """Object detection dataset for image-annotation pairs.

    The dataset supports two sample formats:
    1. File-based annotations, where a sample contains annotation path keys
       such as ``label`` or ``annotation_path``.
    2. Inline annotations, where a sample already contains ``boxes`` and ``labels``.

    File-based annotations are expected as whitespace-separated text rows with
    at least 5 values per row. Supported coordinate formats:
    - ``class xc yc w h`` normalized to [0, 1] (YOLO-style)
    - ``class x1 y1 x2 y2`` absolute pixel coordinates
    - ``class x y w h`` absolute pixel coordinates
    """

    _annotation_keys = (
        "annotation",
        "annotation_path",
        "label_path",
        "bbox_path",
        "target_path",
        "label",
    )
    _pascal_voc_label_to_id = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
    }

    def __init__(
        self,
        split_data: List[Dict[str, Any]],
        transform=None,
        label_offset: int = 1,
        box_format: str = "auto",
        sync_hflip_prob: float = 0.0,
        resize_size: Optional[Tuple[int, int]] = None,
    ):
        self.samples = split_data
        self.transform = transform
        self.label_offset = int(label_offset)
        self.box_format = str(box_format).lower()
        self.sync_hflip_prob = float(sync_hflip_prob)
        self.resize_size = self._normalize_resize_size(resize_size)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _normalize_resize_size(resize_size: Optional[Any]) -> Optional[Tuple[int, int]]:
        if resize_size is None:
            return None
        if isinstance(resize_size, Integral):
            size = int(resize_size)
            if size <= 0:
                raise ValueError("Detection resize_size must be positive.")
            return (size, size)
        if isinstance(resize_size, (tuple, list)) and len(resize_size) == 2:
            height = int(resize_size[0])
            width = int(resize_size[1])
            if height <= 0 or width <= 0:
                raise ValueError("Detection resize_size must contain positive height/width.")
            return (height, width)
        raise ValueError("Detection resize_size must be an int or a 2-tuple of ints.")

    @staticmethod
    def _empty_target_tensors() -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    def _parse_box(
        self,
        values: List[float],
        image_width: int,
        image_height: int,
    ) -> Optional[List[float]]:
        fmt = self.box_format
        if fmt == "auto":
            if all(0.0 <= value <= 1.0 for value in values):
                fmt = "yolo_xywh_norm"
            elif values[2] > values[0] and values[3] > values[1]:
                fmt = "xyxy_abs"
            else:
                fmt = "xywh_abs"

        if fmt == "yolo_xywh_norm":
            xc, yc, w, h = values
            x1 = (xc - (w / 2.0)) * image_width
            y1 = (yc - (h / 2.0)) * image_height
            x2 = (xc + (w / 2.0)) * image_width
            y2 = (yc + (h / 2.0)) * image_height
        elif fmt == "xyxy_abs":
            x1, y1, x2, y2 = values
        elif fmt == "xywh_abs":
            x, y, w, h = values
            x1, y1 = x, y
            x2, y2 = x + w, y + h
        else:
            raise ValueError(f"Unsupported detection box format: {self.box_format}")

        x1 = min(max(float(x1), 0.0), float(image_width))
        y1 = min(max(float(y1), 0.0), float(image_height))
        x2 = min(max(float(x2), 0.0), float(image_width))
        y2 = min(max(float(y2), 0.0), float(image_height))

        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _resolve_annotation_path(self, sample: Dict[str, Any], image_path: Path) -> Path:
        for key in self._annotation_keys:
            raw_path = sample.get(key)
            if isinstance(raw_path, str) and raw_path.strip():
                ann_path = Path(raw_path)
                if not ann_path.is_absolute():
                    ann_path = (image_path.parent / ann_path).resolve()
                return ann_path

        raise KeyError(
            "Detection sample missing annotation path. Expected one of: "
            f"{self._annotation_keys}."
        )

    def _parse_annotation_file(
        self,
        annotation_path: Path,
        image_width: int,
        image_height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not annotation_path.exists():
            raise FileNotFoundError(f"Detection annotation file not found: {annotation_path}")

        boxes: List[List[float]] = []
        labels: List[int] = []

        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                tokens = stripped.split()
                if len(tokens) < 5:
                    continue

                try:
                    coords = [float(value) for value in tokens[1:5]]
                except ValueError:
                    continue

                class_token = tokens[0].strip().lower()
                try:
                    class_id = int(float(class_token))
                except ValueError:
                    if class_token not in self._pascal_voc_label_to_id:
                        continue
                    class_id = self._pascal_voc_label_to_id[class_token]
                class_id += self.label_offset

                box = self._parse_box(coords, image_width=image_width, image_height=image_height)
                if box is None:
                    continue

                boxes.append(box)
                labels.append(class_id)

        if not boxes:
            return self._empty_target_tensors()

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    def _parse_inline_annotations(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = sample.get("boxes")
        labels = sample.get("labels")

        if boxes is None or labels is None:
            return self._empty_target_tensors()

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
        if self.label_offset != 0 and labels_tensor.numel() > 0:
            labels_tensor = labels_tensor + self.label_offset
        return boxes_tensor, labels_tensor

    @staticmethod
    def _hflip_boxes(boxes: torch.Tensor, image_width: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        flipped = boxes.clone()
        flipped[:, 0] = float(image_width) - boxes[:, 2]
        flipped[:, 2] = float(image_width) - boxes[:, 0]
        return flipped

    @staticmethod
    def _resize_boxes(
        boxes: torch.Tensor,
        old_width: int,
        old_height: int,
        new_width: int,
        new_height: int,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        resized = boxes.clone()
        scale_x = float(new_width) / float(old_width)
        scale_y = float(new_height) / float(old_height)
        resized[:, [0, 2]] = resized[:, [0, 2]] * scale_x
        resized[:, [1, 3]] = resized[:, [1, 3]] * scale_y
        return resized

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = Path(sample["image"])
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        orig_width, orig_height = image_width, image_height

        if "boxes" in sample and "labels" in sample:
            boxes, labels = self._parse_inline_annotations(sample)
        else:
            annotation_path = self._resolve_annotation_path(sample, image_path=image_path)
            boxes, labels = self._parse_annotation_file(
                annotation_path=annotation_path,
                image_width=image_width,
                image_height=image_height,
            )

        if self.sync_hflip_prob > 0.0 and random.random() < self.sync_hflip_prob:
            image = TF.hflip(image)
            boxes = self._hflip_boxes(boxes, image_width=image_width)

        if self.resize_size is not None:
            new_height, new_width = self.resize_size
            image = TF.resize(image, [new_height, new_width])
            boxes = self._resize_boxes(
                boxes,
                old_width=image_width,
                old_height=image_height,
                new_width=new_width,
                new_height=new_height,
            )
            image_width, image_height = new_width, new_height

        area = torch.zeros((boxes.shape[0],), dtype=torch.float32)
        if boxes.numel() > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
            "area": area,
            "orig_size": torch.tensor([orig_height, orig_width], dtype=torch.int64),
            "size": torch.tensor([image_height, image_width], dtype=torch.int64),
        }

        if self.transform:
            image = self.transform(image)

        return image, target


class BaseDataset(Dataset):
    """Registry-based dataset loader for standard benchmarks.

    Provides a unified interface for loading common datasets using a registry
    pattern. New datasets can be added via the `register_dataset` decorator.

    Attributes:
        _loaders: Class-level registry mapping dataset names to loader functions

    Supported Datasets:
        - **tiny_imagenet**: Tiny ImageNet (200 classes, 64x64 images)
        - **food101**: Food-101 (101 food categories, variable size)
        - **ade20k**: ADE20K (semantic segmentation, 150 classes)

    Examples:
        >>> # Load existing dataset
        >>> dataset = BaseDataset("tiny_imagenet", "/data/datasets", "train")
        >>> f"Size: {len(dataset)}"
        >>> img_path, label = dataset[0]
        >>>
        >>> # Get dataset summary
        >>> summary = dataset.get_summary()
        >>> f"Classes: {summary['unique_labels']}"
        >>>
        >>> # Register custom dataset
        >>> @BaseDataset.register_dataset("cifar10")
        ... def load_cifar10(self):
        ...     # Custom loading logic
        ...     for img_path in (self.base_path / "images").glob("*.png"):
        ...         self.data.append(str(img_path))
        ...         self.labels.append(img_path.stem.split("_")[0])

    Notes:
        - Data and labels are stored as lists of strings (file paths and label strings)
        - Lazy evaluation: actual image loading happens in downstream datasets
        - Loader functions are registered at module import time
    """

    _loaders = {}
    _dataset_aliases = {
        "tinyimagenet": "tiny_imagenet",
        "tiny-imagenet": "tiny_imagenet",
        "pascal_voc": "PascalVOC",
        "pascal-voc": "PascalVOC",
        "pascalvoc": "PascalVOC",
        "bdd100k": "BDD100K",
        "bdd_100k": "BDD100K",
        "bdd-100k": "BDD100K",
        "fedisic2019": "fed_isic2019",
        "fed_isic_2019": "fed_isic2019",
        "fed-isic-2019": "fed_isic2019",
    }

    def __init__(self, dataset_name: str, base_path: str, split="train"):
        """Initialize dataset loader.

        Args:
            dataset_name: Name of the dataset (must be registered)
            base_path: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')

        Raises:
            FileNotFoundError: If base_path doesn't exist
            ValueError: If dataset_name is not registered
        """
        self.requested_dataset_name = dataset_name
        self.dataset_name = self._normalize_dataset_name(dataset_name)
        self.base_path = Path(base_path)
        self.split = split
        self.data: List[str] = []
        self.labels: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

        self._validate_inputs()
        self._load_dataset()
        if not self.metadata:
            self.metadata = [{} for _ in self.data]
        if len(self.metadata) != len(self.data):
            raise ValueError(
                f"Dataset loader for '{self.dataset_name}' populated metadata with "
                f"{len(self.metadata)} entries for {len(self.data)} samples."
            )

    def _validate_inputs(self):
        """Validate constructor parameters."""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path {self.base_path} does not exist")

        if self.dataset_name not in self._loaders:
            raise ValueError(
                f"Dataset '{self.requested_dataset_name}' not supported. "
                f"Available: {list(self._loaders.keys())}"
            )

    def _load_dataset(self):
        loader_func = self._loaders[self.dataset_name]
        loader_func(self)

    @classmethod
    def _normalize_dataset_name(cls, dataset_name: str) -> str:
        """Resolve dataset aliases and case-insensitive registry keys."""
        candidate = dataset_name.strip()
        if candidate in cls._loaders:
            return candidate

        lowered = candidate.lower()
        if lowered in cls._dataset_aliases:
            return cls._dataset_aliases[lowered]

        for registered in cls._loaders:
            if registered.lower() == lowered:
                return registered

        return candidate

    @classmethod
    def register_dataset(cls, name: str):
        """Decorator to register dataset loader functions.

        Args:
            name: Unique identifier for the dataset

        Returns:
            Decorator function that registers the loader

        Examples:
            >>> @BaseDataset.register_dataset("my_dataset")
            ... def load_my_dataset(self):
            ...     # self has attributes: base_path, split, data, labels
            ...     for img in (self.base_path / self.split).glob("*.jpg"):
            ...         self.data.append(str(img))
            ...         self.labels.append(img.stem)
            >>>
            >>> # Now can use it
            >>> dataset = BaseDataset("my_dataset", "/path/to/data", "train")

        Notes:
            - Loader function receives self (BaseDataset instance)
            - Must populate self.data (list of image paths) and self.labels
            - Called automatically during __init__ after validation
        """

        def decorator(func):
            cls._loaders[name] = func
            return func

        return decorator

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:

        return self.data[idx], self.labels[idx]

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        return dict(self.metadata[idx])

    def get_record(self, idx: int) -> Dict[str, Any]:
        record = {"image": self.data[idx], "label": self.labels[idx]}
        record.update(self.get_metadata(idx))
        return record

    def get_summary(self) -> dict:
        """Get dataset summary statistics.

        Returns:
            Dictionary containing:
                - dataset_name: Name of the dataset
                - split: Split name (train/val/test)
                - total_samples: Number of samples
                - unique_labels: Number of unique classes
                - base_path: Root directory path

        Examples:
            >>> dataset = BaseDataset("food101", "/data", "train")
            >>> summary = dataset.get_summary()
            >>> f"Dataset: {summary['dataset_name']}"
            >>> f"Samples: {summary['total_samples']}"
            >>> f"Classes: {summary['unique_labels']}"
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "total_samples": len(self),
            "unique_labels": len(set(self.labels)),
            "base_path": str(self.base_path),
            # "label_distribution": {label: self.labels.count(label) for label in set(self.labels)},
        }

# Helper function converting image references into an ImageDataset
def to_image_dataset(
    images: List[Any],
    transform: Callable = None,
    processor: Any = None,
) -> ImageDataset:
    file_dataset = []
    for item in images:
        if isinstance(item, dict):
            if "image" not in item:
                raise KeyError("Image records passed to to_image_dataset() must include an 'image' key.")
            file_dataset.append({"image": item["image"]})
        else:
            file_dataset.append({"image": item})
    return ImageDataset(file_dataset=file_dataset, transform=transform, processor=processor)


def _require_file(path: Path, *, context: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    return path

@BaseDataset.register_dataset("tiny_imagenet")
def _load_tiny_imagenet(self) -> None:
    """Load Tiny ImageNet dataset."""
    valid_splits = ["train", "val"]
    # there's a test split but no labels provided
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")
    split_path = self.base_path / "tiny-imagenet-200" / self.split

    if not split_path.exists():
        raise FileNotFoundError(f"Split path {split_path} does not exist")

    # need to handle train and val differently
    if self.split == "train":
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                label = class_dir.name
                img_path = class_dir / "images"
                for img_file in sorted(img_path.glob("*.JPEG")):
                    self.data.append(str(img_file))
                    self.labels.append(label)
    else:
        # val annotations are in a text file
        val_annotations_file = split_path / "val_annotations.txt"
        if not val_annotations_file.exists():
            raise FileNotFoundError(f"Validation annotations file {val_annotations_file} does not exist")

        img_to_label = {}
        with open(val_annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_to_label[parts[0]] = parts[1]

        img_path = split_path / "images"
        for img_file in sorted(img_path.glob("*.JPEG")):
            img_name = img_file.name
            if img_name in img_to_label:
                self.data.append(str(img_file))
                self.labels.append(img_to_label[img_name])


@BaseDataset.register_dataset("food101")
def _load_food101(self):
    """Load Food-101 dataset."""
    valid_splits = ["train", "val"]
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")

    candidate_paths = [
        self.base_path / "food101",
        self.base_path / "food-101",
    ]
    data_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])

    split_file = data_path / "meta" / f"{self.split}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file {split_file} does not exist")

    images_dir = data_path / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist")

    # this json contains the label as a dict key and list of image paths as values
    with open(split_file, "r") as f:
        split_data = json.load(f)

    for label, img_list in split_data.items():
        for img_rel_path in img_list:
            img_path = images_dir / f"{img_rel_path}.jpg"
            _require_file(img_path, context=f"Food-101 image for label '{label}'")
            self.data.append(str(img_path))
            self.labels.append(label)


@BaseDataset.register_dataset("ade20k")
def _load_ade20k(self):
    """Load ADE20K dataset."""
    valid_splits = ["train", "val"]
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")

    # convert to ADE20K naming
    self.split = "training" if self.split == "train" else "validation"

    images_dir = self.base_path / "ADE20K" / "ADE20K_2021_17_01" / "images" / "ADE" / self.split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist")

    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith(".jpg"):
                img_file = Path(root) / file
                mask_file = _require_file(
                    img_file.with_name(img_file.stem + "_seg.png"),
                    context=f"ADE20K mask for image '{img_file.name}'",
                )
                self.data.append(str(img_file))
                self.labels.append(str(mask_file))

@BaseDataset.register_dataset("PascalVOC")
def _load_pascal_voc(self):
    """Load Pascal VOC 2012 dataset."""
    valid_splits = ["train", "val"]
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")

    data_path: Path = self.base_path / "VOC2012"

    images_dir = data_path / "images" / self.split
    labels_dir = data_path / "labels" / self.split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Masks directory {labels_dir} does not exist")
    
    # image names are read from the names inside the image directory directly
    img_names = [img_file.stem for img_file in sorted(images_dir.glob("*.jpg"))]

    for img_name in img_names:
        img_path = images_dir / f"{img_name}.jpg"
        mask_path = labels_dir / f"{img_name}.txt"
        _require_file(img_path, context=f"Pascal VOC image '{img_name}'")
        _require_file(mask_path, context=f"Pascal VOC annotation '{img_name}'")
        self.data.append(str(img_path))
        self.labels.append(str(mask_path))

@BaseDataset.register_dataset("BDD100K") # Berkeley DeepDrive 100K
def _load_bdd100k(self):
    """Load BDD100K dataset."""
    valid_splits = ["train", "val", "test"]
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")

    data_path: Path = self.base_path / "bdd100k"

    images_dir = data_path / "images" / self.split
    masks_dir = data_path / "segmentation" / self.split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory {masks_dir} does not exist")

    for img_file in sorted(images_dir.glob("*.jpg")):
        mask_file = masks_dir / (img_file.stem + ".png")
        mask_file = _require_file(
            masks_dir / (img_file.stem + ".png"),
            context=f"BDD100K mask for image '{img_file.name}'",
        )
        self.data.append(str(img_file))
        self.labels.append(str(mask_file))


def _canonicalize_split_name(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized in {"val", "valid", "validation"}:
        return "val"
    if normalized in {"test", "testing"}:
        return "test"
    return "train"


def _first_existing_path(base_path: Path, candidates: List[str]) -> Path:
    for candidate in candidates:
        path = base_path / candidate
        if path.exists():
            return path
    return base_path / candidates[0]


def _resolve_femnist_root(base_path: Path) -> Path:
    candidates = [
        base_path,
        base_path / "FEMNIST",
    ]
    for candidate in candidates:
        processed_dir = candidate / "processed"
        train_file = processed_dir / "femnist_train.pt"
        test_file = processed_dir / "femnist_test.pt"
        users_file = processed_dir / "femnist_user_keys.pt"
        if train_file.exists() and test_file.exists() and users_file.exists():
            return candidate
    return candidates[0]


def _femnist_cache_key(root: Path, split: str) -> Tuple[str, str]:
    return (str(root.resolve()), str(split))


def _load_femnist_split_tensors(root: Path, split: str) -> Tuple[Any, Any, Any]:
    cache_key = _femnist_cache_key(root, split)
    if cache_key in _FEMNIST_SPLIT_CACHE:
        return _FEMNIST_SPLIT_CACHE[cache_key]

    processed_dir = root / "processed"
    data_file = _require_file(
        processed_dir / f"femnist_{split}.pt",
        context=f"FEMNIST processed {split} tensor file",
    )
    _require_file(
        processed_dir / "femnist_user_keys.pt",
        context="FEMNIST processed user key file",
    )
    data_targets_users = torch.load(data_file, map_location="cpu", weights_only=False)
    if not isinstance(data_targets_users, (tuple, list)) or len(data_targets_users) != 3:
        raise ValueError(
            f"FEMNIST processed file {data_file} must contain a 3-tuple of (data, targets, users)."
        )
    data, targets, users = data_targets_users
    if len(data) != len(targets) or len(data) != len(users):
        raise ValueError(
            f"FEMNIST processed file {data_file} has mismatched lengths: "
            f"data={len(data)} targets={len(targets)} users={len(users)}."
        )
    _FEMNIST_SPLIT_CACHE[cache_key] = (data, targets, users)
    return _FEMNIST_SPLIT_CACHE[cache_key]


def _normalize_femnist_pixels(pixels: Any) -> np.ndarray:
    array = np.asarray(pixels)
    if array.ndim == 1:
        side = int(round(float(array.size) ** 0.5))
        if side * side != array.size:
            raise ValueError(f"FEMNIST sample has non-square flattened image of length {array.size}.")
        array = array.reshape(side, side)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"FEMNIST sample must decode to a 2D grayscale array, got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("FEMNIST sample contains no finite pixel values.")
        max_value = float(finite.max())
        min_value = float(finite.min())
        if 0.0 <= min_value and max_value <= 1.0:
            array = array * 255.0
        array = np.clip(np.rint(array), 0.0, 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def _build_femnist_image_ref(root: Path, split: str, sample_idx: int) -> str:
    encoded_root = quote(str(root.resolve()), safe="")
    return f"femnist://sample?root={encoded_root}&split={split}&index={int(sample_idx)}"


def _parse_femnist_image_ref(ref: str) -> Tuple[Path, str, int]:
    parsed = urlparse(ref)
    if parsed.scheme != "femnist":
        raise ValueError(f"Unsupported virtual image ref scheme for FEMNIST: {ref}")
    query = parse_qs(parsed.query)
    root_values = query.get("root")
    split_values = query.get("split")
    index_values = query.get("index")
    if not root_values or not split_values or not index_values:
        raise ValueError(f"Invalid FEMNIST image ref: {ref}")
    root = Path(unquote(root_values[0]))
    split = str(split_values[0])
    sample_idx = int(index_values[0])
    return root, split, sample_idx


def _load_virtual_femnist_image(ref: str) -> Image.Image:
    root, split, sample_idx = _parse_femnist_image_ref(ref)
    data, _, _ = _load_femnist_split_tensors(root, split)
    pixels = data[sample_idx]
    if isinstance(pixels, torch.Tensor):
        pixels = pixels.detach().cpu().numpy()
    array = _normalize_femnist_pixels(pixels)
    return Image.fromarray(array, mode="L")


def _normalize_fed_isic_row(row: Dict[str, Any]) -> Dict[str, str]:
    alias_groups = {
        "image": ("image", "image_path", "img_path", "path", "filepath", "file_path"),
        "label": ("label", "target", "class", "diagnosis", "y"),
        "center": ("center", "site", "client", "hospital", "center_id"),
        "split": ("split", "partition", "subset", "fold"),
    }
    normalized: Dict[str, str] = {}
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for canonical, aliases in alias_groups.items():
        value = None
        for alias in aliases:
            if alias in lowered:
                value = lowered[alias]
                break
        if value is None:
            raise ValueError(f"FedISIC manifest row is missing required field '{canonical}'.")
        normalized[canonical] = str(value).strip()
    return normalized


@BaseDataset.register_dataset("fed_isic2019")
def _load_fed_isic2019(self):
    valid_splits = ["train", "val", "test"]
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")

    manifest_path = _first_existing_path(
        self.base_path,
        [
            "metadata.csv",
            "fed_isic2019.csv",
            "fed_isic_2019.csv",
            "manifest.csv",
            "labels.csv",
        ],
    )
    _require_file(manifest_path, context="FedISIC-2019 manifest")

    canonical_split = _canonicalize_split_name(self.split)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = _normalize_fed_isic_row(row)
            row_split = _canonicalize_split_name(normalized["split"])
            if canonical_split == "val":
                if row_split != "val":
                    continue
            elif row_split != canonical_split:
                continue

            image_path = Path(normalized["image"])
            if not image_path.is_absolute():
                image_path = self.base_path / image_path
            _require_file(image_path, context="FedISIC-2019 image")

            self.data.append(str(image_path))
            self.labels.append(normalized["label"])
            self.metadata.append({"natural_client_id": normalized["center"], "split": row_split})


@BaseDataset.register_dataset("femnist")
def _load_femnist(self):
    valid_splits = ["train", "val", "test"]
    if self.split not in valid_splits:
        raise ValueError(f"Invalid split '{self.split}'. Valid options: {valid_splits}")

    requested_split = "test" if self.split in {"val", "test"} else "train"
    femnist_root = _resolve_femnist_root(self.base_path)
    data, targets, users = _load_femnist_split_tensors(femnist_root, requested_split)

    for sample_idx, (pixels, label, user) in enumerate(zip(data, targets, users)):
        if isinstance(label, torch.Tensor):
            label = label.item()
        image_ref = _build_femnist_image_ref(femnist_root, requested_split, sample_idx)
        self.data.append(image_ref)
        self.labels.append(str(int(label)))
        self.metadata.append({"natural_client_id": str(user), "split": requested_split})
