"""FDSDataFactory: Converts DatasetSplits to PyTorch DataLoaders for NVFlare 2.7.

Bridges the gap between sophisticated partitioning strategies (IID, Dirichlet,
embedding-based, VLM-based) and federated learning training infrastructure by
consuming DatasetSplits JSON files and producing task-specific DataLoaders.

Supports:
- Classification: ClassificationDataset with ImageNet transforms
- Segmentation: SegmentationDataset with segmentation transforms
- Detection: Stubbed for Phase 5 implementation

Architecture Decision: Direct DatasetSplits consumption (not legacy data_modules)
to avoid intermediate file I/O and align with new partitioning API.
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data_loading.dataset import ClassificationDataset, SegmentationDataset


class FDSDataFactory:
    """Factory for creating PyTorch DataLoaders from DatasetSplits.

    Consumes DatasetSplits JSON files produced by any splitter (RandomSplitter,
    DirichletSplitter, EmbeddingSplitter, MultivariateDirichletSplitter) and
    generates task-specific DataLoaders ready for federated training.

    Examples:
        >>> # Load DatasetSplits from JSON
        >>> with open("splits.json") as f:
        ...     splits = json.load(f)
        >>>
        >>> # Create loaders for client 0 (classification)
        >>> train_loader, val_loader = FDSDataFactory.create_dataloaders(
        ...     splits_dict=splits,
        ...     client_idx=0,
        ...     task_type="classification",
        ...     batch_size=32,
        ...     num_workers=4
        ... )
        >>>
        >>> # Use in training
        >>> for images, labels in train_loader:
        ...     # Train model
        ...     pass

    Notes:
        - Global validation set takes priority over local validation
        - Empty validation sets return val_loader=None
        - Task-specific transforms are hardcoded (ImageNet stats for classification)
        - Detection support stubbed for Phase 5
    """

    @staticmethod
    def create_dataloaders(
        splits_dict: Dict[str, Any],
        client_idx: int,
        task_type: str,
        batch_size: int = 32,
        num_workers: int = 4,
        label_map: Optional[Dict[str, int]] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create PyTorch DataLoaders for a specific client and task type.

        Args:
            splits_dict: DatasetSplits dictionary with 'clients' and 'global_val' keys
            client_idx: Client index (0-indexed, maps to "client_0", "client_1", etc.)
            task_type: Task type - "classification", "segmentation", or "detection"
            batch_size: Batch size for DataLoaders (default: 32)
            num_workers: Number of data loading workers (default: 4)
            label_map: Optional mapping from string labels to integer indices
                      (auto-created from training data if not provided)

        Returns:
            Tuple of (train_loader, val_loader)
            - train_loader: DataLoader for client's training data
            - val_loader: DataLoader for validation (global_val if available,
                         else client's local val, or None if no validation data)

        Raises:
            ValueError: If task_type is invalid or DatasetSplits is malformed
            KeyError: If client_idx not found in splits_dict

        Examples:
            >>> # Classification with global validation
            >>> train_loader, val_loader = FDSDataFactory.create_dataloaders(
            ...     splits_dict=splits,
            ...     client_idx=0,
            ...     task_type="classification",
            ...     batch_size=64
            ... )
            >>>
            >>> # Segmentation with local validation
            >>> train_loader, val_loader = FDSDataFactory.create_dataloaders(
            ...     splits_dict=splits_no_global,
            ...     client_idx=1,
            ...     task_type="segmentation",
            ...     num_workers=8
            ... )
        """
        # Validate inputs
        FDSDataFactory._validate_inputs(splits_dict, client_idx, task_type)

        # Extract client's data
        client_key = f"client_{client_idx}"
        if client_key not in splits_dict["clients"]:
            available_clients = list(splits_dict["clients"].keys())
            raise KeyError(
                f"Client '{client_key}' not found in splits. "
                f"Available clients: {available_clients}"
            )

        client_split = splits_dict["clients"][client_key]
        train_data = client_split["train"]
        local_val_data = client_split.get("val", [])

        # Use global validation if available, otherwise local validation
        global_val_data = splits_dict.get("global_val", None)
        val_data = global_val_data if global_val_data is not None else local_val_data

        # Create task-specific datasets and loaders
        if task_type == "classification":
            train_loader, val_loader = FDSDataFactory._create_classification_loaders(
                train_data, val_data, batch_size, num_workers, label_map
            )
        elif task_type == "segmentation":
            train_loader, val_loader = FDSDataFactory._create_segmentation_loaders(
                train_data, val_data, batch_size, num_workers
            )
        elif task_type == "detection":
            raise NotImplementedError(
                "Detection task support will be implemented in Phase 5. "
                "Currently only 'classification' and 'segmentation' are supported."
            )
        else:
            raise ValueError(
                f"Invalid task_type '{task_type}'. "
                f"Valid options: ['classification', 'segmentation', 'detection']"
            )

        return train_loader, val_loader

    @staticmethod
    def _validate_inputs(splits_dict: Dict[str, Any], client_idx: int, task_type: str):
        """Validate DatasetSplits structure and parameters.

        Args:
            splits_dict: DatasetSplits dictionary to validate
            client_idx: Client index to validate
            task_type: Task type to validate

        Raises:
            ValueError: If inputs are invalid or DatasetSplits malformed
        """
        # Validate splits_dict structure
        if not isinstance(splits_dict, dict):
            raise ValueError(
                f"splits_dict must be a dictionary, got {type(splits_dict)}"
            )

        if "clients" not in splits_dict:
            raise ValueError(
                "DatasetSplits malformed: missing 'clients' key. "
                "Expected structure: {'clients': {...}, 'global_val': [...]}"
            )

        if not isinstance(splits_dict["clients"], dict):
            raise ValueError(
                f"DatasetSplits['clients'] must be a dictionary, "
                f"got {type(splits_dict['clients'])}"
            )

        # Validate client_idx
        if not isinstance(client_idx, int) or client_idx < 0:
            raise ValueError(
                f"client_idx must be a non-negative integer, got {client_idx}"
            )

        # Validate task_type
        valid_tasks = ["classification", "segmentation", "detection"]
        if task_type not in valid_tasks:
            raise ValueError(
                f"Invalid task_type '{task_type}'. Valid options: {valid_tasks}"
            )

    @staticmethod
    def _create_classification_loaders(
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        batch_size: int,
        num_workers: int,
        label_map: Optional[Dict[str, int]] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create classification DataLoaders with ImageNet transforms.

        Args:
            train_data: List of training samples with 'image' and 'label' keys
            val_data: List of validation samples (may be empty)
            batch_size: Batch size for DataLoaders
            num_workers: Number of data loading workers
            label_map: Optional mapping from string labels to integer indices

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # ImageNet normalization statistics
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # Training transforms with augmentation
        train_transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        # Validation transforms (no augmentation)
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        # Create label map if not provided (extract unique labels from training data)
        if label_map is None:
            unique_labels = sorted(set(sample["label"] for sample in train_data))
            label_map = {label: idx for idx, label in enumerate(unique_labels)}

        # Create datasets
        train_dataset = ClassificationDataset(
            split_data=train_data,
            label_map=label_map,
            transform=train_transform
        )

        # Create validation dataset if data available
        val_dataset = None
        if val_data and len(val_data) > 0:
            val_dataset = ClassificationDataset(
                split_data=val_data,
                label_map=label_map,
                transform=val_transform
            )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

        return train_loader, val_loader

    @staticmethod
    def _create_segmentation_loaders(
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        batch_size: int,
        num_workers: int
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create segmentation DataLoaders with segmentation transforms.

        Args:
            train_data: List of training samples with 'image' and 'mask_path' keys
            val_data: List of validation samples (may be empty)
            batch_size: Batch size for DataLoaders
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_loader, val_loader)

        Notes:
            - Images resized to 512x512 for segmentation tasks
            - Masks use nearest-neighbor interpolation (no smoothing)
            - ImageNet normalization applied to images
        """
        # ImageNet normalization statistics
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # Image transforms
        image_transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        # Mask transforms (nearest-neighbor to preserve class labels)
        mask_transform = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

        # Create datasets
        train_dataset = SegmentationDataset(
            split_data=train_data,
            transform=image_transform,
            mask_transform=mask_transform
        )

        # Create validation dataset if data available
        val_dataset = None
        if val_data and len(val_data) > 0:
            val_dataset = SegmentationDataset(
                split_data=val_data,
                transform=image_transform,
                mask_transform=mask_transform
            )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

        return train_loader, val_loader
