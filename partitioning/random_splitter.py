"""Random IID data splitting for federated learning.

This module implements random (IID - Independent and Identically Distributed)
splitting of datasets across federated learning clients. Data is distributed
uniformly at random, creating balanced, homogeneous client datasets.
"""

import numpy as np
from typing import List
from .base_splitter import BaseDatasetSplitter, DatasetSplits, ClientSplit
from data_loading.dataset import BaseDataset


class RandomSplitter(BaseDatasetSplitter):
    """Randomly split dataset across clients for IID data distribution.

    Creates Independent and Identically Distributed (IID) splits by randomly
    shuffling and partitioning data across clients. Each client receives
    approximately equal amounts of data with similar label distributions.

    This splitting strategy is useful for:
    - Baseline experiments (comparing against non-IID scenarios)
    - Scenarios where data is naturally homogeneous across clients
    - Testing federated algorithms under ideal conditions

    The heterogeneity parameter is ignored for RandomSplitter since splits
    are always uniform/IID.

    Validation set modes:
    - "local": Each client gets random subsets from both train and val datasets
    - "global": Single shared validation set (all of val_dataset)
    - "none": No validation sets created

    Examples:
        >>> from data_loading.dataset import BaseDataset
        >>> train_data = BaseDataset("cifar10", "/data/", "train")
        >>> val_data = BaseDataset("cifar10", "/data/", "val")
        >>>
        >>> # Create IID splits with global validation
        >>> splitter = RandomSplitter(
        ...     train_dataset=train_data,
        ...     val_dataset=val_data,
        ...     n_clients=10,
        ...     valset_type="global",
        ...     seed=42
        ... )
        >>> splits = splitter.get_client_splits()
        >>> summary = splitter.get_split_summary()
        >>> print(f"Client 0 has {summary['client_sizes']['client_0']['train_samples']} samples")
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset | None,
        n_clients: int,
        heterogeneity: float = 0.0,
        valset_type: str = "global",
        seed: int | None = None,
    ):
        """Initialize the random splitter.

        Args:
            train_dataset: Training dataset to split.
            val_dataset: Optional validation dataset. Required if valset_type
                is "local" or "global".
            n_clients: Number of clients to distribute data across.
            heterogeneity: Ignored for RandomSplitter (kept for interface
                consistency). Random splits are always IID.
            valset_type: Validation set configuration - "local", "global", or "none".
            seed: Random seed for reproducibility.
        """
        super().__init__(train_dataset, val_dataset, n_clients, heterogeneity, valset_type, seed)

    def _compute_client_splits(self) -> DatasetSplits:
        """Perform random IID splitting of the dataset.

        Randomly shuffles training data and distributes it evenly across clients.
        Validation sets are handled according to valset_type configuration.

        Returns:
            DatasetSplits: Dictionary containing:
                - clients: Per-client train/val splits
                - global_val: Global validation set (if valset_type="global")
        """
        print("Performing random splitting of the dataset...")
        # Seed already set in base class init

        splits: DatasetSplits = {
            "clients": {},
            "global_val": None,
        }

        train_indices = np.arange(len(self.train_dataset))
        np.random.shuffle(train_indices)
        client_train_indices = np.array_split(train_indices, self.n_clients)

        if self.valset_type == "global":
            return self._split_with_global_val(client_train_indices, splits)
        elif self.valset_type == "local":
            return self._split_with_local_val(client_train_indices, splits)
        else:  # valset_type == "none"
            return self._split_without_val(client_train_indices, splits)

    def _split_with_global_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:
        """Create splits with a single global validation set.

        Args:
            client_train_indices: Pre-shuffled training indices for each client.
            splits: DatasetSplits structure to populate.

        Returns:
            DatasetSplits: Populated splits with global validation set.

        Raises:
            ValueError: If val_dataset is None.
        """
        if self.val_dataset is None:
            raise ValueError("val_dataset must be provided for global validation splitting.")

        splits["global_val"] = self._load_dataset_samples(self.val_dataset)

        for i, indices in enumerate(client_train_indices):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, indices.tolist())

            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=[])

        print("Random splitting with global validation completed.")
        return splits

    def _split_with_local_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:
        """Create splits with per-client local validation sets.

        Randomly distributes validation data across clients in addition to
        training data.

        Args:
            client_train_indices: Pre-shuffled training indices for each client.
            splits: DatasetSplits structure to populate.

        Returns:
            DatasetSplits: Populated splits with local validation sets.

        Raises:
            ValueError: If val_dataset is None.
        """
        if self.val_dataset is None:
            raise ValueError("val_dataset must be provided for local validation splitting.")

        val_indices = np.arange(len(self.val_dataset))
        np.random.shuffle(val_indices)
        client_val_indices = np.array_split(val_indices, self.n_clients)

        for i, (train_indices, val_indices) in enumerate(zip(client_train_indices, client_val_indices)):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, train_indices.tolist())
            client_val_data = self._load_dataset_samples(self.val_dataset, val_indices.tolist())

            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=client_val_data)
        print("Random splitting with local validation completed.")
        return splits

    def _split_without_val(self, client_train_indices: List[np.ndarray], splits: DatasetSplits) -> DatasetSplits:
        """Create splits without any validation sets.

        Args:
            client_train_indices: Pre-shuffled training indices for each client.
            splits: DatasetSplits structure to populate.

        Returns:
            DatasetSplits: Populated splits with no validation data.
        """
        for i, train_indices in enumerate(client_train_indices):
            client_id = f"client_{i}"
            client_train_data = self._load_dataset_samples(self.train_dataset, train_indices.tolist())
            splits["clients"][client_id] = ClientSplit(train=client_train_data, val=[])
        print("Random splitting without validation completed.")
        return splits

    
if __name__ == "__main__":
    # Test the RandomSplitter with Tiny ImageNet
    train_dataset = BaseDataset("tiny_imagenet", "/path/to/gtai/FDS/", "train")
    val_dataset = BaseDataset("tiny_imagenet", "/path/to/gtai/FDS/", "val")
    print("Testing with global validation set:")
    splitter = RandomSplitter(train_dataset, val_dataset, n_clients=5, heterogeneity=0.5, valset_type="global", seed=42)
    print(splitter.get_split_summary())
    print("\nTesting with local validation sets:")
    splitter = RandomSplitter(train_dataset, val_dataset, n_clients=5, heterogeneity=0.5, valset_type="local", seed=42)
    print(splitter.get_split_summary())
    print("\nTesting without validation sets:")
    splitter = RandomSplitter(train_dataset, None, n_clients=5, heterogeneity=0.5, valset_type="none", seed=42)
    print(splitter.get_split_summary())
    