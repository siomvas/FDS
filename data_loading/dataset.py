import os
import json
from torch.utils.data import Dataset
import torch
from typing import Callable, List, Tuple, Dict, Any
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, UnidentifiedImageError


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
            with Image.open(path) as img:
                image = img.convert("RGB")
            if self.transform:
                return self.transform(image)
            elif self.processor:
                inputs = self.processor(images=image, return_tensors="pt")
                return inputs.pixel_values.squeeze(0)
            else:
                return image

        except (UnidentifiedImageError, OSError, Exception) as e:
            print(f"[Warning] Skipping corrupt image {path}: {e}")
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
        label = self.label_map[sample["label"]] if self.label_map else sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class SegmentationDataset(Dataset):
    """Semantic segmentation dataset for image-mask pairs.

    Loads synchronized image and segmentation mask pairs for pixel-level
    classification tasks. Applies transforms to both images and masks.

    Args:
        split_data: List of dicts with 'image' (path) and 'mask_path' (mask path) keys
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
    """
    def __init__(self, split_data: List[Dict[str, Any]], transform=None, mask_transform=None):
        self.samples = split_data  # [{"image": img_path, "label": mask_path}, ...]
        self.transform = transform
        self.mask_transform = mask_transform
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask_path"])

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


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
        >>> print(f"Size: {len(dataset)}")
        >>> img_path, label = dataset[0]
        >>>
        >>> # Get dataset summary
        >>> summary = dataset.get_summary()
        >>> print(f"Classes: {summary['unique_labels']}")
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
        self.dataset_name = dataset_name
        self.base_path = Path(base_path)
        self.split = split
        self.data: List[str] = []
        self.labels: List[str] = []

        self._validate_inputs()
        self._load_dataset()

    def _validate_inputs(self):
        """Validate constructor parameters."""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path {self.base_path} does not exist")

        if self.dataset_name not in self._loaders:
            raise ValueError(f"Dataset '{self.dataset_name}' not supported. " f"Available: {list(self._loaders.keys())}")

    def _load_dataset(self):
        loader_func = self._loaders[self.dataset_name]
        loader_func(self)

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
            >>> print(f"Dataset: {summary['dataset_name']}")
            >>> print(f"Samples: {summary['total_samples']}")
            >>> print(f"Classes: {summary['unique_labels']}")
        """
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "total_samples": len(self),
            "unique_labels": len(set(self.labels)),
            "base_path": str(self.base_path),
            # "label_distribution": {label: self.labels.count(label) for label in set(self.labels)},
        }

# Helper function converting from list of strings to ImageDataset
def to_image_dataset(images: List[str], transform: Callable = None, processor: Any = None) -> ImageDataset:
    file_dataset = [{"image": path} for path in images]
    return ImageDataset(file_dataset=file_dataset, transform=transform, processor=processor)

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

    data_path: Path = self.base_path / "food101"

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
            if img_path.exists():
                self.data.append(str(img_path))
                self.labels.append(label)
            else:
                raise Warning(f"Image path {img_path} does not exist")


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
                self.data.append(str(img_file))
                mask_file = img_file.with_name(img_file.stem + "_seg.png")
                if mask_file.exists():
                    self.labels.append(str(mask_file))
                else:
                    raise Warning(f"Mask file {mask_file} does not exist")


if __name__ == "__main__":

    dataset = BaseDataset("ade20k", "/path/to/gtai/FDS", "train")
    print(f"Dataset size: {len(dataset)}")
    print(f"Summary: {dataset.get_summary()}")

    img_path, label = dataset[0]
    print(f"First sample - Path: {img_path}, Label: {label}")
