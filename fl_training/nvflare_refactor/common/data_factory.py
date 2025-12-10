"""
Data Factory for NVFlare FL Training

Provides unified interface to existing data_modules for all datasets.
Supports all splitting methods: IID, Dirichlet, VLM-guided, Embeddings+Clustering.
"""

from typing import Optional, Tuple
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data_modules.dm import (
    TinyImageNetDataModule,
    Food101DataModule,
    ADE20KDataModule,
    BaseDataModule
)


class DatasetConfig:
    """Configuration for dataset loading"""

    # Default data paths (update for your environment)
    DEFAULT_PATHS = {
        "tiny_imagenet": "/vol/research/tubui/tiny-imagenet-200/",
        "food101": "/vol/research/tubui/food-101/",
        "ade20k": "/vol/research/tubui/ADEChallengeData2016/",
    }

    # Image sizes for each dataset
    IMAGE_SIZES = {
        "tiny_imagenet": 64,
        "food101": 224,
        "ade20k": 512,
    }

    # Number of classes
    NUM_CLASSES = {
        "tiny_imagenet": 200,
        "food101": 101,
        "ade20k": 150,  # ADE20K has 150 classes
    }


def get_datamodule(
    dataset_name: str,
    data_dir: Optional[str] = None,
    split_file: Optional[str] = None,
    client_idx: int = 0,
    central: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    **kwargs
) -> BaseDataModule:
    """
    Factory function to get the appropriate DataModule for a dataset.

    Args:
        dataset_name: Name of dataset ('tiny_imagenet', 'food101', 'ade20k')
        data_dir: Path to dataset directory (uses default if None)
        split_file: Path to client split JSON/PKL file (from data_partitioning)
                   Can be from any splitter: RandomSplitter, DirichletSplitter,
                   MultivariateDirichletSplitter, EmbeddingSplitter
        client_idx: Index of client (0-based)
        central: If True, aggregate all training data (for central baseline)
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility
        **kwargs: Additional dataset-specific arguments

    Returns:
        Initialized DataModule ready for use

    Examples:
        # Load client 0's data with Dirichlet split
        >>> dm = get_datamodule(
        ...     dataset_name="tiny_imagenet",
        ...     split_file="data_partitioning/dirichlet_split/dirichlet_split_0.1.json",
        ...     client_idx=0,
        ...     batch_size=64
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader

        # Load with VLM-guided multivariate split
        >>> dm = get_datamodule(
        ...     dataset_name="food101",
        ...     split_file="data_partitioning/multivariate_splits/split_10clients.json",
        ...     client_idx=5
        ... )

        # Central baseline (all training data)
        >>> dm = get_datamodule(
        ...     dataset_name="ade20k",
        ...     central=True,
        ...     batch_size=16
        ... )
    """
    dataset_name = dataset_name.lower()

    # Get default data directory if not provided
    if data_dir is None:
        if dataset_name in DatasetConfig.DEFAULT_PATHS:
            data_dir = DatasetConfig.DEFAULT_PATHS[dataset_name]
        else:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(DatasetConfig.DEFAULT_PATHS.keys())}"
            )

    # Create appropriate DataModule
    if dataset_name == "tiny_imagenet":
        datamodule = TinyImageNetDataModule(
            data_dir=data_dir,
            split_file=split_file,
            client_idx=client_idx,
            central=central,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            **kwargs
        )

    elif dataset_name == "food101":
        datamodule = Food101DataModule(
            data_dir=data_dir,
            split_file=split_file,
            client_idx=client_idx,
            central=central,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            **kwargs
        )

    elif dataset_name == "ade20k":
        datamodule = ADE20KDataModule(
            data_dir=data_dir,
            split_file=split_file,
            client_idx=client_idx,
            central=central,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: tiny_imagenet, food101, ade20k"
        )

    return datamodule


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get metadata about a dataset.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary with num_classes, image_size, default_path
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DatasetConfig.NUM_CLASSES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DatasetConfig.NUM_CLASSES.keys())}"
        )

    return {
        "num_classes": DatasetConfig.NUM_CLASSES[dataset_name],
        "image_size": DatasetConfig.IMAGE_SIZES[dataset_name],
        "default_path": DatasetConfig.DEFAULT_PATHS[dataset_name],
    }


def validate_split_file(split_file: str) -> bool:
    """
    Validate that a split file exists and is readable.

    Args:
        split_file: Path to split file

    Returns:
        True if valid, raises FileNotFoundError otherwise
    """
    split_path = Path(split_file)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Generate splits using one of:\n"
            f"  - RandomSplitter\n"
            f"  - DirichletSplitter\n"
            f"  - MultivariateDirichletSplitter\n"
            f"  - EmbeddingSplitter"
        )

    if split_path.suffix not in ['.json', '.pkl', '.pickle']:
        raise ValueError(
            f"Split file must be .json or .pkl, got: {split_path.suffix}"
        )

    return True


# Convenience function for FL simulation setup
def setup_fl_data(
    dataset_name: str,
    split_file: str,
    n_clients: int,
    batch_size: int = 32,
    **kwargs
) -> list:
    """
    Set up data modules for all clients in FL simulation.

    Args:
        dataset_name: Dataset name
        split_file: Path to split file
        n_clients: Number of clients
        batch_size: Batch size
        **kwargs: Additional arguments for get_datamodule

    Returns:
        List of initialized DataModules, one per client

    Example:
        >>> data_modules = setup_fl_data(
        ...     dataset_name="tiny_imagenet",
        ...     split_file="dirichlet_split_0.1.json",
        ...     n_clients=10,
        ...     batch_size=64
        ... )
        >>> for dm in data_modules:
        ...     dm.setup()
    """
    validate_split_file(split_file)

    data_modules = []
    for client_idx in range(n_clients):
        dm = get_datamodule(
            dataset_name=dataset_name,
            split_file=split_file,
            client_idx=client_idx,
            central=False,
            batch_size=batch_size,
            **kwargs
        )
        data_modules.append(dm)

    return data_modules


if __name__ == "__main__":
    # Example usage and testing
    print("Testing data_factory.py...")

    # Get dataset info
    for dataset in ["tiny_imagenet", "food101", "ade20k"]:
        info = get_dataset_info(dataset)
        print(f"\n{dataset.upper()}:")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Image size: {info['image_size']}")
        print(f"  Default path: {info['default_path']}")

    print("\n✓ Data factory module ready!")
