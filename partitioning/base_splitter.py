"""Base classes and types for dataset splitting across federated learning clients.

This module provides the abstract base class and type definitions for splitting
datasets across multiple clients in federated learning scenarios. Supports various
validation set configurations and provides utilities for serialization and validation.
"""

from data_loading.dataset import BaseDataset
from typing import Dict, List, Optional, TypedDict, Any
import numpy as np
import torch
from abc import ABC, abstractmethod
import json
import pickle


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

        if valset_type == "none" and val_dataset is not None:
            print("Warning: val_dataset provided but valset_type is 'none'. The val_dataset will be ignored.")

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
            >>> print(f"Client 0 has {len(client_0_train)} training samples")
        """
        if self.splits is None:
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

        if format == "json":
            with open(filename, "w") as f:
                serializable_splits = self._make_serializable(splits)
                json.dump(serializable_splits, f, indent=2)
        elif format == "pkl":
            with open(filename, "wb") as f:
                pickle.dump(splits, f)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'pkl'.")

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
                {"image": str(sample["image"]), "label": sample["label"]} for sample in splits["global_val"]
            ]

        for client_id, client_split in splits["clients"].items():
            serializable_train = [{"image": str(sample["image"]), "label": sample["label"]} for sample in client_split["train"]]
            serializable_val = [{"image": str(sample["image"]), "label": sample["label"]} for sample in client_split["val"]]
            serializable_splits["clients"][client_id] = ClientSplit(
                train=serializable_train,
                val=serializable_val,
            )

        return serializable_splits

    def _load_dataset_samples(self, dataset: BaseDataset, indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Load samples from the dataset.

        Args:
            dataset: The dataset to load from.
            indices: Optional list of indices to load. If None, loads all samples.

        Returns:
            List of dictionaries with "image" (path) and "label" keys.
        """
        if indices is not None:
            return [{"image": str(dataset[idx][0]), "label": dataset[idx][1]} for idx in indices]
        else:
            return [{"image": str(dataset[idx][0]), "label": dataset[idx][1]} for idx in range(len(dataset))]

    def load_client_splits(self, filename: str, format: str = "json") -> None:
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
        if format == "json":
            with open(filename, "r") as f:
                self.splits = json.load(f)
        elif format == "pkl":
            with open(filename, "rb") as f:
                self.splits = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'json' or 'pkl'.")

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
        if not (0.0 <= heterogeneity <= 1.0):
            raise ValueError(f"Heterogeneity must be between 0.0 and 1.0, got {heterogeneity}")
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
            >>> print(f"Total samples: {summary['total_train_samples']}")
            >>> print(f"Samples per client: {summary['client_sizes']}")
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

    def validate_splits(self) -> bool:
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
            ...     print("Splits are valid!")
            ... except ValueError as e:
            ...     print(f"Validation failed: {e}")
        """
        splits = self.get_client_splits()
        if len(splits["clients"]) != self.n_clients:
            raise ValueError(f"Number of clients in splits ({len(splits['clients'])}) does not match n_clients ({self.n_clients})")
        
        all_train_paths = set()
        for client_id, split in splits["clients"].items():
            client_train_paths = {sample["image"] for sample in split["train"]}
            if all_train_paths.intersection(client_train_paths):
                raise ValueError(f"Overlap detected in training data for client {client_id}")
            all_train_paths.update(client_train_paths)
        expected_train_paths = {str(self.train_dataset[idx][0]) for idx in range(len(self.train_dataset))}
        if all_train_paths != expected_train_paths:
            missing = expected_train_paths - all_train_paths
            extra = all_train_paths - expected_train_paths
            raise ValueError(f"Training data splits do not cover the entire dataset. Missing: {missing}, Extra: {extra}")
        
        all_val_paths = set()
        if self.valset_type == "global" and splits["global_val"] is not None and self.val_dataset is not None:
            all_val_paths = {sample["image"] for sample in splits["global_val"]}
            expected_val_paths = {str(self.val_dataset[idx][0]) for idx in range(len(self.val_dataset))}
            if all_val_paths != expected_val_paths:
                missing = expected_val_paths - all_val_paths
                extra = all_val_paths - expected_val_paths
                raise ValueError(f"Global validation data does not cover the entire validation dataset. Missing: {missing}, Extra: {extra}")
        elif self.valset_type == "local" and self.val_dataset is not None:
            for client_id, split in splits["clients"].items():
                client_val_paths = {sample["image"] for sample in split["val"]}
                if all_val_paths.intersection(client_val_paths):
                    raise ValueError(f"Overlap detected in validation data for client {client_id}")
                all_val_paths.update(client_val_paths)
            expected_val_paths = {str(self.val_dataset[idx][0]) for idx in range(len(self.val_dataset))}
            if all_val_paths != expected_val_paths:
                missing = expected_val_paths - all_val_paths
                extra = all_val_paths - expected_val_paths
                raise ValueError(f"Local validation data splits do not cover the entire validation dataset. Missing: {missing}, Extra: {extra}")
        
        print("All splits are valid.")
        return True
