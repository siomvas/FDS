"""Data partitioning package for federated learning.

This package provides tools for splitting datasets across federated learning clients
with different heterogeneity strategies. It implements various data partitioning
approaches ranging from IID (Independent and Identically Distributed) to highly
non-IID splits based on labels, semantic features, or multiple criteria.

Available Splitters
-------------------

**BaseDatasetSplitter** (Abstract Base Class)
    Provides the common interface and utilities for all splitting strategies.
    Handles validation set configurations, serialization, and split validation.

**RandomSplitter** (IID Distribution)
    Randomly distributes data uniformly across clients, creating balanced,
    homogeneous datasets. Useful for baseline experiments and ideal conditions.

**DirichletSplitter** (Label-based Non-IID)
    Creates label heterogeneity using Dirichlet distribution. The alpha
    (heterogeneity) parameter controls the degree of imbalance:
    - Low alpha (0.1-0.5): Highly heterogeneous, each client specializes
    - Medium alpha (1.0-5.0): Moderately heterogeneous, realistic imbalance
    - High alpha (10.0+): Nearly IID, approaches uniform distribution

**EmbeddingSplitter** (Semantic Non-IID)
    Partitions data based on semantic similarity using foundation model embeddings
    (DINOv2, CLIP, SigLIP). Clusters semantically similar samples and assigns
    clusters to clients, creating natural visual heterogeneity.

**MultivariateDirichletSplitter** (VLM-based Multi-criterion)
    Advanced splitter using Vision-Language Model (VLM) annotations to create
    heterogeneity across multiple criteria simultaneously (e.g., style, color,
    composition, lighting). Controlled by alpha parameter and preference weights.

Validation Set Modes
--------------------

All splitters support three validation set configurations:

- **"local"**: Each client receives its own validation subset, proportional
  to their training data size.

- **"global"**: Single shared validation set used by all clients, enabling
  consistent evaluation across the federation.

- **"none"**: No validation sets created, useful when evaluation is done
  separately or not needed.

Type Definitions
----------------

**ClientSplit**: TypedDict defining a single client's data with 'train' and 'val' keys.

**DatasetSplits**: TypedDict defining the complete split structure with 'clients'
    dict and optional 'global_val' list.

Examples
--------

IID splitting with global validation:

    >>> from partitioning import RandomSplitter
    >>> from data_loading.dataset import BaseDataset
    >>>
    >>> train_data = BaseDataset("cifar10", "/data/", "train")
    >>> val_data = BaseDataset("cifar10", "/data/", "val")
    >>>
    >>> splitter = RandomSplitter(
    ...     train_dataset=train_data,
    ...     val_dataset=val_data,
    ...     n_clients=10,
    ...     valset_type="global",
    ...     seed=42
    ... )
    >>> splits = splitter.get_client_splits()
    >>> print(f"Client 0 has {len(splits['clients']['client_0']['train'])} samples")

Non-IID splitting with Dirichlet:

    >>> from partitioning import DirichletSplitter
    >>>
    >>> splitter = DirichletSplitter(
    ...     train_dataset=train_data,
    ...     val_dataset=val_data,
    ...     n_clients=10,
    ...     heterogeneity=0.5,  # Medium heterogeneity
    ...     valset_type="local",
    ...     seed=42
    ... )
    >>> splits = splitter.get_client_splits()
    >>> summary = splitter.get_split_summary()
    >>> print(summary)

Semantic clustering-based splitting:

    >>> from partitioning import EmbeddingSplitter
    >>> from partitioning.embedding_utils.config import EmbeddingConfig
    >>>
    >>> config = EmbeddingConfig(
    ...     model_name="dinov2-base",
    ...     pooling_strategy="cls",
    ...     clustering_method="hdbscan"
    ... )
    >>>
    >>> splitter = EmbeddingSplitter(
    ...     train_dataset=train_data,
    ...     val_dataset=val_data,
    ...     n_clients=10,
    ...     heterogeneity=0.7,
    ...     valset_type="global",
    ...     config=config,
    ...     seed=42
    ... )
    >>> splits = splitter.get_client_splits()

Notes
-----

- All splitters ensure no overlap between clients' training data
- Splits can be serialized to JSON or pickle for reproducibility
- The heterogeneity parameter is interpreted differently by each splitter
- Random seeds ensure reproducible splits across runs
"""

# Import all splitter classes and types for easy access
from .base_splitter import (
    BaseDatasetSplitter,
    ClientSplit,
    DatasetSplits,
)
from .random_splitter import RandomSplitter
from .dirichlet_splitter import DirichletSplitter
from .embedding_splitter import EmbeddingSplitter
from .multivariate_dirichlet_splitter import MultivariateDirichletSplitter

__all__ = [
    'BaseDatasetSplitter',
    'ClientSplit',
    'DatasetSplits',
    'RandomSplitter',
    'DirichletSplitter',
    'EmbeddingSplitter',
    'MultivariateDirichletSplitter',
]
