"""
Data modules for federated learning training.

Provides PyTorch Lightning-inspired data modules that load split datasets,
apply transforms, and create dataloaders for FL training.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from data_loading.dataset import (
    ClassificationDataset,
    SegmentationDataset,
    DetectionDataset,
)


class BaseDataModule(ABC):
    """
    Abstract base class for federated learning data modules.

    Inspired by PyTorch Lightning's LightningDataModule.
    """

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        label_map: Optional[Dict[str, int]] = None,
        client_idx: Optional[int] = None,
        image_size: Optional[int] = None,
        central: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Root directory containing datasets
            split_file: Path to JSON/PKL split file from splitter
            client_idx: Which client's data to load (None for central mode)
            central: If True, use all training data (ignore client_idx)
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.split_file = split_file
        self.label_map = label_map
        self.client_idx = client_idx
        self.image_size = image_size
        self.central = central
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Will be set by subclasses
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        # Dataloaders
        self._train_dataloader: Optional[DataLoader] = None
        self._val_dataloader: Optional[DataLoader] = None

    def load_splits(self) -> Dict:
        """Load split data from JSON or PKL file."""
        split_path = Path(self.split_file)

        if split_path.suffix == ".json":
            with open(split_path, "r") as f:
                return json.load(f)
        elif split_path.suffix == ".pkl":
            with open(split_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported split file format: {split_path.suffix}")

    @staticmethod
    def _is_int_like(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, str):
            return value.lstrip("-").isdigit()
        return False

    def _resolve_label_map(self, splits: Dict) -> Optional[Dict[str, int]]:
        """Build a deterministic global label map when labels are non-numeric."""
        if self.label_map is not None:
            return self.label_map

        labels: List[Any] = []
        for client_split in splits.get("clients", {}).values():
            for key in ("train", "val"):
                for sample in client_split.get(key, []):
                    labels.append(sample.get("label"))
        for sample in splits.get("global_val") or []:
            labels.append(sample.get("label"))

        labels = [label for label in labels if label is not None]
        if not labels:
            return None

        unique_labels = sorted(set(labels), key=lambda x: str(x))
        if all(self._is_int_like(label) for label in unique_labels):
            return None

        return {str(label): idx for idx, label in enumerate(unique_labels)}

    def get_client_data(self, splits: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract train and validation data for the specified client or central mode.

        Args:
            splits: DatasetSplits dictionary

        Returns:
            (train_data, val_data) lists
        """
        if self.central:
            # Central mode: concatenate all client training data
            train_data = []
            for client_id, client_split in splits["clients"].items():
                train_data.extend(client_split["train"])

            # Use global validation if available, otherwise aggregate local val
            if splits.get("global_val"):
                val_data = splits["global_val"]
            else:
                val_data = []
                for client_id, client_split in splits["clients"].items():
                    val_data.extend(client_split["val"])

            return train_data, val_data

        else:
            # Federated mode: get specific client's data
            if self.client_idx is None:
                raise ValueError("client_idx must be specified when central=False")

            client_id = f"client_{self.client_idx}"
            if client_id not in splits["clients"]:
                raise ValueError(f"Client {client_id} not found in splits")

            client_split = splits["clients"][client_id]
            train_data = client_split["train"]

            # Use global val if available, otherwise use client's local val
            if splits.get("global_val"):
                val_data = splits["global_val"]
            else:
                val_data = client_split["val"]

            return train_data, val_data

    @abstractmethod
    def get_train_transform(self) -> T.Compose:
        """Return training transforms (with augmentation)."""
        pass

    @abstractmethod
    def get_val_transform(self) -> T.Compose:
        """Return validation transforms (no augmentation)."""
        pass

    @abstractmethod
    def setup(self):
        """Load data and create datasets."""
        pass

    def get_collate_fn(self):
        """Optional dataloader collate function override."""
        return None

    @property
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self._train_dataloader is None:
            if self.train_dataset is None:
                raise RuntimeError("Must call setup() before accessing dataloaders")

            self._train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=self.num_workers > 0,
                collate_fn=self.get_collate_fn(),
            )

        return self._train_dataloader

    @property
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None

        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=self.num_workers > 0,
                collate_fn=self.get_collate_fn(),
            )

        return self._val_dataloader

    def teardown(self):
        """Cleanup resources."""
        self.train_dataset = None
        self.val_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None

    def __repr__(self) -> str:
        mode = "Central" if self.central else f"Client {self.client_idx}"
        return f"{self.__class__.__name__}({mode}, batch_size={self.batch_size})"


# ============================================================================
# DATASET-SPECIFIC DATA MODULES
# ============================================================================


class TinyImageNetDataModule(BaseDataModule):

    def get_train_transform(self) -> T.Compose:
        """Training transforms with augmentation."""
        return T.Compose([
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def get_val_transform(self) -> T.Compose:
        """Validation transforms without augmentation."""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def setup(self):
        """Load data and create datasets."""
        splits = self.load_splits()
        self.label_map = self._resolve_label_map(splits)
        train_data, val_data = self.get_client_data(splits)
        train_transform = self.get_train_transform()
        val_transform = self.get_val_transform()

        # Create datasets
        self.train_dataset = ClassificationDataset(
            split_data=train_data,
            label_map=self.label_map,
            transform=train_transform,
        )

        if val_data:
            self.val_dataset = ClassificationDataset(
                split_data=val_data,
                label_map=self.label_map,
                transform=val_transform,
            )


class Food101DataModule(BaseDataModule):
    """
    Data module for Food-101 dataset.

    Food-101: Variable size RGB images, 101 food classes
    Standard preprocessing: 256x256 resize, 224x224 crop
    """

    def get_train_transform(self) -> T.Compose:
        """Training transforms with augmentation."""
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def get_val_transform(self) -> T.Compose:
        """Validation transforms without augmentation."""
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def setup(self):
        """Load data and create datasets."""

        splits = self.load_splits()
        self.label_map = self._resolve_label_map(splits)
        train_data, val_data = self.get_client_data(splits)
        train_transform = self.get_train_transform()
        val_transform = self.get_val_transform()

        # Create datasets
        self.train_dataset = ClassificationDataset(
            split_data=train_data,
            label_map=self.label_map,
            transform=train_transform,
        )

        if val_data:
            self.val_dataset = ClassificationDataset(
                split_data=val_data,
                label_map=self.label_map,
                transform=val_transform,
            )


class ADE20KDataModule(BaseDataModule):
    """
    Data module for ADE20K semantic segmentation dataset.

    ADE20K: Variable size RGB images with pixel-level semantic labels
    150 semantic classes
    """

    def setup(self):
        """Load data and create segmentation datasets."""

        splits = self.load_splits()
        train_data, val_data = self.get_client_data(splits)
        train_transform = self.get_train_transform()
        val_transform = self.get_val_transform()
        mask_transform = self.get_mask_transform()

        self.train_dataset = SegmentationDataset(
            split_data=train_data,
            transform=train_transform,
            mask_transform=mask_transform,
        )

        if val_data:
            self.val_dataset = SegmentationDataset(
                split_data=val_data,
                transform=val_transform,
                mask_transform=mask_transform,
            )

    def get_train_transform(self) -> T.Compose:
        """Training transforms for images."""
        image_size = int(self.image_size) if self.image_size is not None else 512
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def get_val_transform(self) -> T.Compose:
        """Validation transforms for images."""
        image_size = int(self.image_size) if self.image_size is not None else 512
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def get_mask_transform(self) -> T.Compose:
        """Transforms for segmentation masks (no normalization)."""
        image_size = int(self.image_size) if self.image_size is not None else 512
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            # Keep discrete class ids (0-255) without scaling to [0, 1].
            T.PILToTensor(),
        ])


class PascalVOCDataModule(BaseDataModule):
    """Data module for Pascal VOC object detection."""

    def __init__(
        self,
        *args,
        detection_label_offset: int = 1,
        detection_box_format: str = "auto",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.detection_label_offset = int(detection_label_offset)
        self.detection_box_format = str(detection_box_format).lower()

    def setup(self):
        splits = self.load_splits()
        train_data, val_data = self.get_client_data(splits)
        train_transform = self.get_train_transform()
        val_transform = self.get_val_transform()

        self.train_dataset = DetectionDataset(
            split_data=train_data,
            transform=train_transform,
            label_offset=self.detection_label_offset,
            box_format=self.detection_box_format,
        )

        if val_data:
            self.val_dataset = DetectionDataset(
                split_data=val_data,
                transform=val_transform,
                label_offset=self.detection_label_offset,
                box_format=self.detection_box_format,
            )

    def get_train_transform(self) -> T.Compose:
        # Keep tensor conversion only. Torchvision detection backends handle
        # normalization and multi-scale image sizes internally.
        return T.Compose([T.ToTensor()])

    def get_val_transform(self) -> T.Compose:
        return T.Compose([T.ToTensor()])

    @staticmethod
    def _detection_collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

    def get_collate_fn(self):
        return self._detection_collate_fn


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def get_datamodule(
    dataset_name: str,
    data_dir: str,
    split_file: str,
    label_map: Optional[Dict[str, int]] = None,
    client_idx: Optional[int] = None,
    image_size: Optional[int] = None,
    central: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    **kwargs,
) -> BaseDataModule:
    """
    Factory function to create data modules by dataset name.

    Args:
        dataset_name: Name of dataset ("tiny_imagenet", "food101", "ade20k", "pascal_voc")
        data_dir: Root data directory
        split_file: Path to split JSON/PKL file
        client_idx: Client index for federated mode
        central: If True, use all training data
        batch_size: Batch size
        num_workers: Number of dataloader workers
        seed: Random seed
        **kwargs: Dataset-specific parameters

    Returns:
        Appropriate DataModule instance

    Example:
        >>> dm = get_datamodule(
        ...     "tiny_imagenet",
        ...     data_dir="./data",
        ...     split_file="./splits/random_split.json",
        ...     client_idx=0,
        ...     batch_size=64
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader
    """
    datamodules = {
        "tiny_imagenet": TinyImageNetDataModule,
        "food101": Food101DataModule,
        "ade20k": ADE20KDataModule,
        "pascal_voc": PascalVOCDataModule,
        "pascalvoc": PascalVOCDataModule,
    }

    if dataset_name.lower() not in datamodules:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(datamodules.keys())}"
        )

    DataModuleClass = datamodules[dataset_name.lower()]

    return DataModuleClass(
        data_dir=data_dir,
        split_file=split_file,
        label_map=label_map,
        client_idx=client_idx,
        image_size=image_size,
        central=central,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        **kwargs,
    )
