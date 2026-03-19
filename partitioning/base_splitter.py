"""Base classes and types for dataset splitting across federated learning clients.

This module provides the abstract base class and type definitions for splitting
datasets across multiple clients in federated learning scenarios. Supports various
validation set configurations and provides utilities for serialization and validation.
"""

from data_loading.dataset import BaseDataset
import logging
from typing import Dict, List, Optional, TypedDict, Any
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import torch
from abc import ABC, abstractmethod
import json
import pickle

logger = logging.getLogger(__name__)


class ClientSplit(TypedDict):
    """Type definition for a single client's data split.

    Attributes:
        train: List of training samples for the client.
        val: List of validation samples for the client (may be empty).
    """
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]


class DatasetSplits(TypedDict):
    """Type definition for complete dataset splits across all clients.

    Attributes:
        clients: Dictionary mapping client IDs to their data splits.
        global_val: Optional global validation set shared across all clients.
    """
    clients: Dict[str, ClientSplit]
    global_val: Optional[List[Dict[str, Any]]]


class BaseDatasetSplitter(ABC):
    """Abstract base class for splitting datasets across multiple clients.

    Provides a common interface and utilities for distributing data across
    federated learning clients. Subclasses implement specific splitting
    strategies (e.g., IID, Dirichlet, embedding-based) while inheriting
    common functionality for validation set handling, serialization, and
    split validation.

    This class supports three types of validation set configurations:
    - "local": Each client gets its own validation subset
    - "global": Single shared validation set for all clients
    - "none": No validation sets

    Attributes:
        train_dataset: The training dataset to split across clients.
        val_dataset: Optional validation dataset.
        n_clients: Number of clients in the federation.
        heterogeneity: Degree of data heterogeneity (interpretation varies by subclass).
        valset_type: Type of validation set configuration ("local", "global", "none").
        seed: Random seed for reproducibility.
        splits: Computed data splits (None until first access).
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset],
        n_clients: int,
        heterogeneity: float,
        valset_type: str,
        seed: Optional[int] = None,
    ):
        """Initialize the dataset splitter.

        Args:
            train_dataset: The training dataset instance to split.
            val_dataset: Optional validation dataset. Required if valset_type
                is "local" or "global".
            n_clients: Number of clients to split data across. Must be >= 1.
            heterogeneity: Degree of data heterogeneity. Interpretation depends
                on the specific splitter implementation.
            valset_type: Type of validation set - "local", "global", or "none".
                "local" creates per-client validation sets, "global" creates a
                single shared set, "none" creates no validation sets.
            seed: Random seed for reproducibility. If None, results may vary.

        Raises:
            ValueError: If valset_type is invalid, or if val_dataset requirements
                don't match valset_type.
        """

        valset_type_valid_options = ["local", "global", "none"]
        if valset_type not in valset_type_valid_options:
            raise ValueError(f"Invalid valset_type '{valset_type}'. Valid options: {valset_type_valid_options}")

        if valset_type in ["local", "global"] and val_dataset is None:
            raise ValueError(f"val_dataset must be provided when valset_type is '{valset_type}'")
        if n_clients < 1:
            raise ValueError(f"n_clients must be >= 1, got {n_clients}")

        if valset_type == "none" and val_dataset is not None:
            logger.warning("val_dataset provided while valset_type='none'; val_dataset will be ignored.")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_clients = n_clients
        self.heterogeneity = heterogeneity
        self.valset_type = valset_type
        self.seed = seed
        self.splits: Optional[DatasetSplits] = None

        self._set_random_seed(seed)

    def _set_random_seed(self, seed: Optional[int]) -> None:
        """Set random seed for NumPy and PyTorch.

        Args:
            seed: Random seed value. If None, no seeding is performed.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    @abstractmethod
    def _compute_client_splits(self) -> None:
        """Compute the actual data splits across clients.

        Must be implemented by subclasses to define the specific splitting
        strategy (e.g., random, Dirichlet, embedding-based).

        Returns:
            DatasetSplits: Dictionary containing client data splits and optional
                global validation set.

        Raises:
            NotImplementedError: This is an abstract method that must be
                implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_client_splits(self) -> DatasetSplits:
        """Get the computed client data splits.

        Lazily computes splits on first call by invoking _compute_client_splits().
        Subsequent calls return the cached splits unless invalidated by changing
        heterogeneity or seed.

        Returns:
            DatasetSplits: Dictionary with structure:
                {
                    "clients": {
                        "client_0": {"train": [...], "val": [...]},
                        "client_1": {"train": [...], "val": [...]},
                        ...
                    },
                    "global_val": [...] or None
                }

        Example:
            >>> splitter = RandomSplitter(dataset, n_clients=3, ...)
            >>> splits = splitter.get_client_splits()
            >>> client_0_train = splits["clients"]["client_0"]["train"]
            >>> f"Client 0 has {len(client_0_train)} training samples"
        """
        if self.splits is None:
            # Re-seed on each fresh computation so two splitter instances with the
            # same seed remain reproducible regardless of call order.
            self._set_random_seed(self.seed)
            self.splits = self._compute_client_splits()
        return self.splits  # type: ignore

    def store_client_splits(self, filename: str, format: str = "json") -> None:
        """Store the client data splits to a file.

        Serializes the computed splits to disk for later use or analysis.
        Supports JSON (human-readable) and pickle (preserves Python objects) formats.

        Args:
            filename: Path to the output file.
            format: Serialization format - "json" or "pkl". JSON is recommended
                for portability and human readability. Pickle preserves exact
                Python objects but is less portable.

        Raises:
            ValueError: If format is not "json" or "pkl".

        Example:
            >>> splitter = DirichletSplitter(...)
            >>> splitter.get_client_splits()
            >>> splitter.store_client_splits("splits.json", format="json")
        """
        splits = self.get_client_splits()
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        normalized_format = format.lower()
        if normalized_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                serializable_splits = self._make_serializable(splits)
                json.dump(serializable_splits, f, indent=2)
        elif normalized_format in {"pkl", "pickle"}:
            with open(output_path, "wb") as f:
                pickle.dump(splits, f)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'pkl'/'pickle'.")

    def _make_serializable(self, splits: DatasetSplits) -> DatasetSplits:
        """Convert any non-serializable objects in splits to serializable forms.

        Ensures all image paths are strings and data is JSON-compatible.

        Args:
            splits: The DatasetSplits dictionary to serialize.

        Returns:
            DatasetSplits: A serializable version of the splits.
        """
        serializable_splits: DatasetSplits = {
            "clients": {},
            "global_val": None,
        }

        if splits["global_val"] is not None:
            serializable_splits["global_val"] = [
                self._normalize_split_sample(sample) for sample in splits["global_val"]
            ]

        for client_id, client_split in splits["clients"].items():
            serializable_train = [self._normalize_split_sample(sample) for sample in client_split["train"]]
            serializable_val = [self._normalize_split_sample(sample) for sample in client_split["val"]]
            serializable_splits["clients"][client_id] = ClientSplit(
                train=serializable_train,
                val=serializable_val,
            )

        return serializable_splits

    @staticmethod
    def _is_probable_segmentation_mask(path_like: str) -> bool:
        suffix = Path(path_like).suffix.lower()
        return suffix in {".png", ".bmp", ".gif", ".tif", ".tiff"}

    @staticmethod
    def _is_probable_detection_annotation(path_like: str) -> bool:
        suffix = Path(path_like).suffix.lower()
        return suffix in {".txt", ".xml", ".json"}

    def _normalize_split_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a split sample while preserving task-specific fields.

        Keeps additional keys (e.g., boxes/labels for detection) and adds
        compatibility aliases where possible.
        """
        normalized: Dict[str, Any] = {}
        for key, value in sample.items():
            if isinstance(value, Path):
                normalized[key] = str(value)
            elif isinstance(value, np.generic):
                normalized[key] = value.item()
            elif isinstance(value, torch.Tensor):
                normalized[key] = value.tolist()
            else:
                normalized[key] = value

        if "image" not in normalized:
            raise KeyError("Split sample is missing required 'image' key.")

        normalized["image"] = str(normalized["image"])

        if "mask_path" in normalized:
            normalized["mask_path"] = str(normalized["mask_path"])
            # Keep label for backward compatibility with older split consumers.
            normalized.setdefault("label", normalized["mask_path"])

        label_value = normalized.get("label")
        if isinstance(label_value, Path):
            label_value = str(label_value)
            normalized["label"] = label_value

        if isinstance(label_value, str):
            if self._is_probable_segmentation_mask(label_value):
                normalized.setdefault("mask_path", label_value)
            if self._is_probable_detection_annotation(label_value):
                normalized.setdefault("annotation_path", label_value)

        return normalized

    def _coerce_dataset_item_to_sample(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, Mapping):
            raw_sample = dict(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_sample = {"image": item[0], "label": item[1]}
        else:
            raise TypeError(
                "Dataset item must be a mapping or a tuple/list of (image, label). "
                f"Got type: {type(item).__name__}"
            )

        return self._normalize_split_sample(raw_sample)

    def _extract_label(self, dataset: BaseDataset, idx: int) -> Any:
        sample = self._coerce_dataset_item_to_sample(dataset[idx])
        if "label" not in sample:
            raise KeyError(f"Dataset sample at index {idx} is missing 'label'.")
        return sample["label"]

    def _load_dataset_samples(self, dataset: BaseDataset, indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Load samples from the dataset.

        Args:
            dataset: The dataset to load from.
            indices: Optional list of indices to load. If None, loads all samples.

        Returns:
            List of dictionaries with "image" (path) and "label" keys.
        """
        if indices is not None:
            return [self._coerce_dataset_item_to_sample(dataset[idx]) for idx in indices]
        else:
            return [self._coerce_dataset_item_to_sample(dataset[idx]) for idx in range(len(dataset))]

    def load_client_splits(self, filename: str, format: str = "json") -> DatasetSplits:
        """Load previously saved client splits from a file.

        Loads splits from disk and stores them in self.splits. The loaded
        splits can be accessed via get_client_splits().

        Args:
            filename: Path to the file containing saved splits.
            format: Format of the saved file - "json" or "pkl".

        Raises:
            ValueError: If format is not "json" or "pkl".
            FileNotFoundError: If the specified file doesn't exist.

        Example:
            >>> splitter = DirichletSplitter(...)
            >>> splitter.load_client_splits("splits.json")
            >>> splits = splitter.get_client_splits()
        """
        normalized_format = format.lower()
        if normalized_format == "json":
            with open(filename, "r", encoding="utf-8") as f:
                self.splits = json.load(f)
        elif normalized_format in {"pkl", "pickle"}:
            with open(filename, "rb") as f:
                self.splits = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'pkl'/'pickle'.")

        self.validate_splits(self.splits)

        return self.splits  # type: ignore[return-value]

    # Backward-compatible aliases used by older scripts and tests.
    def store_splits(self, filename: str, format: str = "json") -> None:
        self.store_client_splits(filename, format=format)

    def load_splits(self, filename: str, format: str = "json") -> DatasetSplits:
        return self.load_client_splits(filename, format=format)

    def set_heterogeneity(self, heterogeneity: float) -> None:
        """Update the heterogeneity parameter and invalidate cached splits.

        Changes the heterogeneity level for the splitting strategy. This
        invalidates any previously computed splits, requiring recomputation
        on the next call to get_client_splits().

        Args:
            heterogeneity: New heterogeneity value. Must be between 0.0 and 1.0.

        Raises:
            ValueError: If heterogeneity is not in [0.0, 1.0].
        """
        if heterogeneity < 0.0:
            raise ValueError(f"Heterogeneity must be non-negative, got {heterogeneity}")
        self.heterogeneity = heterogeneity
        self.splits = None  # Invalidate existing splits

    def set_seed(self, seed: int) -> None:
        """Update the random seed and invalidate cached splits.

        Changes the random seed for the splitting strategy. This invalidates
        any previously computed splits and ensures reproducibility with the new seed.

        Args:
            seed: New random seed value.
        """
        self._set_random_seed(seed)
        self.splits = None  # Invalidate existing splits

    def get_split_summary(self) -> Dict[str, Any]:
        """Get a statistical summary of the data splits.

        Provides statistics about the split distribution including total samples,
        per-client sample counts, and configuration parameters.

        Returns:
            Dictionary containing:
                - n_clients: Number of clients
                - heterogeneity: Heterogeneity parameter value
                - valset_type: Validation set configuration type
                - total_train_samples: Total training samples across all clients
                - total_val_samples: Total validation samples (local sets)
                - client_sizes: Per-client sample counts (train/val)
                - seed: Random seed used
                - global_val_samples: Global validation set size (if applicable)

        Example:
            >>> splitter = RandomSplitter(...)
            >>> summary = splitter.get_split_summary()
            >>> f"Total samples: {summary['total_train_samples']}"
            >>> f"Samples per client: {summary['client_sizes']}"
        """
        splits = self.get_client_splits()
        client_sizes = {}
        total_train = 0
        total_val = 0

        for client_id, split in splits["clients"].items():
            n_train = len(split["train"])
            n_val = len(split["val"])
            client_sizes[client_id] = {"train_samples": n_train, "val_samples": n_val}
            total_train += n_train
            total_val += n_val

        summary = {
            "n_clients": self.n_clients,
            "heterogeneity": self.heterogeneity,
            "valset_type": self.valset_type,
            "total_train_samples": total_train,
            "total_val_samples": total_val,
            "client_sizes": client_sizes,
            "seed": self.seed,
        }

        if self.valset_type == "global" and splits["global_val"] is not None:
            summary["global_val_samples"] = len(splits["global_val"])

        return summary

    def validate_splits(self, splits: Optional[DatasetSplits] = None) -> bool:
        """Validate that the splits cover the entire dataset without overlap.

        Performs comprehensive validation checks:
        - Verifies correct number of clients
        - Ensures no overlap in training data between clients
        - Confirms all training data is assigned
        - Validates validation set coverage (local or global)

        Returns:
            True if all validation checks pass.

        Raises:
            ValueError: If validation fails with details about the specific issue:
                - Wrong number of clients
                - Overlapping training data
                - Missing or extra samples
                - Validation set coverage issues

        Example:
            >>> splitter = DirichletSplitter(...)
            >>> splitter.get_client_splits()
            >>> try:
            ...     splitter.validate_splits()
            ...     "Splits are valid!"
            ... except ValueError as e:
            ...     f"Validation failed: {e}"
        """
        if splits is None:
            splits = self.get_client_splits()
        if len(splits["clients"]) != self.n_clients:
            raise ValueError(
                f"Expected {self.n_clients} clients, but got {len(splits['clients'])} in provided splits."
            )
        
        all_train_paths = set()
        for client_id, split in splits["clients"].items():
            client_train_paths = {sample["image"] for sample in split["train"]}
            if all_train_paths.intersection(client_train_paths):
                raise ValueError(f"Detected overlap in training samples for client {client_id}.")
            all_train_paths.update(client_train_paths)
        expected_train_paths = {
            self._coerce_dataset_item_to_sample(self.train_dataset[idx])["image"]
            for idx in range(len(self.train_dataset))
        }
        if all_train_paths != expected_train_paths:
            missing = expected_train_paths - all_train_paths
            extra = all_train_paths - expected_train_paths
            raise ValueError(
                "Expected training splits to cover all samples exactly once. "
                f"Missing: {missing}, Extra: {extra}"
            )
        
        all_val_paths = set()
        if self.valset_type == "global" and splits["global_val"] is not None and self.val_dataset is not None:
            all_val_paths = {sample["image"] for sample in splits["global_val"]}
            expected_val_paths = {
                self._coerce_dataset_item_to_sample(self.val_dataset[idx])["image"]
                for idx in range(len(self.val_dataset))
            }
            if all_val_paths != expected_val_paths:
                missing = expected_val_paths - all_val_paths
                extra = all_val_paths - expected_val_paths
                raise ValueError(f"Global validation data does not cover the entire validation dataset. Missing: {missing}, Extra: {extra}")
        elif self.valset_type == "local" and self.val_dataset is not None:
            for client_id, split in splits["clients"].items():
                client_val_paths = {sample["image"] for sample in split["val"]}
                if all_val_paths.intersection(client_val_paths):
                    raise ValueError(f"Detected overlap in validation samples for client {client_id}.")
                all_val_paths.update(client_val_paths)
            expected_val_paths = {
                self._coerce_dataset_item_to_sample(self.val_dataset[idx])["image"]
                for idx in range(len(self.val_dataset))
            }
            if all_val_paths != expected_val_paths:
                missing = expected_val_paths - all_val_paths
                extra = all_val_paths - expected_val_paths
                raise ValueError(f"Local validation data splits do not cover the entire validation dataset. Missing: {missing}, Extra: {extra}")
        
        logger.info("All splits are valid.")
        return True
