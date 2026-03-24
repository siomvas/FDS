"""Data loading infrastructure for federated learning.

This package provides PyTorch dataset loaders and data modules for federated
learning training. It includes both general-purpose dataset loaders with a
registry pattern and specialized data modules inspired by PyTorch Lightning.

Components
----------

**Dataset Loaders** (`dataset.py`)
    Core PyTorch Dataset implementations with flexible loading:

    - **ImageDataset**: Generic image loader with transform/processor support
    - **ClassificationDataset**: Image classification with labels
    - **SegmentationDataset**: Semantic segmentation with image-mask pairs
    - **BaseDataset**: Registry-based loader for standard benchmarks
      (Tiny ImageNet, Food-101, ADE20K, Pascal VOC, BDD100K) and
      natural-federated evaluation datasets (Fed-ISIC2019, FEMNIST)

**Data Modules** (`data_modules.py`)
    PyTorch Lightning-inspired data modules for FL training:

    - **BaseDataModule**: Abstract base class for federated data loading
    - **TinyImageNetDataModule**: 64x64 RGB, 200 classes
    - **Food101DataModule**: Variable size RGB, 101 food categories
    - **ADE20KDataModule**: Semantic segmentation, 150 classes
    - **PascalVOCDataModule**: Object detection dataset module

Key Features
------------

- **Registry Pattern**: `BaseDataset.register_dataset()` decorator for adding
  new datasets dynamically
- **Federated/Central Modes**: Data modules support both federated (single client)
  and centralized (all clients' data) training
- **Split File Integration**: Load data splits from JSON/PKL files produced
  by partitioning splitters
- **Transform Pipelines**: Separate train (with augmentation) and validation
  (without augmentation) transforms
- **Lazy Loading**: Images loaded on-the-fly to reduce memory footprint
- **Error Handling**: Graceful handling of corrupt/missing images

Usage Examples
--------------

Basic dataset loading with registry:

    >>> from data_loading.dataset import BaseDataset
    >>>
    >>> # Load Tiny ImageNet training split
    >>> dataset = BaseDataset(
    ...     dataset_name="tiny_imagenet",
    ...     base_path="/data/datasets",
    ...     split="train"
    ... )
    >>> f"Dataset size: {len(dataset)}"
    >>> img_path, label = dataset[0]
    >>> f"First sample: {img_path}, Label: {label}"

Register custom dataset:

    >>> from data_loading.dataset import BaseDataset
    >>>
    >>> @BaseDataset.register_dataset("my_dataset")
    ... def load_my_dataset(self):
    ...     # Custom loading logic
    ...     for img_path in Path(self.base_path / "images").glob("*.jpg"):
    ...         self.data.append(str(img_path))
    ...         self.labels.append(img_path.stem)
    >>>
    >>> # Now use it
    >>> dataset = BaseDataset("my_dataset", "/path/to/data", "train")

Federated training with data modules:

    >>> from data_loading.data_modules import get_datamodule
    >>>
    >>> # Client 0 in federated mode
    >>> dm = get_datamodule(
    ...     dataset_name="food101",
    ...     data_dir="./data",
    ...     split_file="./splits/dirichlet_alpha0.5.json",
    ...     client_idx=0,
    ...     batch_size=64,
    ...     num_workers=4
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader
    >>> for images, labels in train_loader:
    ...     # Training loop
    ...     pass

Centralized training mode:

    >>> # Use all clients' data (centralized baseline)
    >>> dm = get_datamodule(
    ...     dataset_name="tiny_imagenet",
    ...     data_dir="./data",
    ...     split_file="./splits/dirichlet_alpha0.5.json",
    ...     central=True,  # Pool all clients' training data
    ...     batch_size=128
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader

Custom transforms with data module:

    >>> from data_loading.data_modules import Food101DataModule
    >>> import torchvision.transforms as T
    >>>
    >>> class CustomFood101DataModule(Food101DataModule):
    ...     def get_train_transform(self):
    ...         return T.Compose([
    ...             T.Resize(256),
    ...             T.RandomCrop(224),
    ...             T.RandomHorizontalFlip(),
    ...             T.RandAugment(num_ops=2, magnitude=9),
    ...             T.ToTensor(),
    ...             T.Normalize(mean=[0.485, 0.456, 0.406],
    ...                         std=[0.229, 0.224, 0.225])
    ...         ])
    >>>
    >>> dm = CustomFood101DataModule(
    ...     data_dir="./data",
    ...     split_file="./splits/embedding_split.json",
    ...     client_idx=0
    ... )

Dataset Registry Pattern
------------------------

The `BaseDataset` class uses a registry pattern to discover and load datasets:

    >>> # List available datasets
    >>> BaseDataset._loaders.keys()
    dict_keys([
    ...     'tiny_imagenet', 'food101', 'ade20k', 'PascalVOC', 'BDD100K',
    ...     'fed_isic2019', 'femnist'
    ... ])
    >>>
    >>> # Register new dataset
    >>> @BaseDataset.register_dataset("cifar10")
    ... def load_cifar10(self):
    ...     # Loading implementation
    ...     pass

Split File Format
-----------------

Data modules expect split files in the format produced by partitioning splitters:

.. code-block:: json

    {
        "clients": {
            "client_0": {
                "train": [
                    {"image": "/path/img1.jpg", "label": "class_0"},
                    {"image": "/path/img2.jpg", "label": "class_1"},
                    ...
                ],
                "val": [...]
            },
            "client_1": {...},
            ...
        },
        "global_val": [...]  # Optional global validation set
    }

Notes
-----

- All data modules normalize images with ImageNet statistics by default
- Segmentation datasets use nearest-neighbor interpolation for masks
- DataLoaders use `pin_memory=True` when CUDA is available
- Number of workers > 0 enables persistent workers for efficiency

See Also
--------

- partitioning: Dataset splitting strategies for federated learning
- evaluation.metrics: Metrics for measuring data heterogeneity
- models: Model implementations for classification and segmentation
"""
