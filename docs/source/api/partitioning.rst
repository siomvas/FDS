Partitioning Package
====================

.. automodule:: partitioning
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The partitioning package provides tools for splitting datasets across federated learning
clients with different heterogeneity strategies. Supports IID, label-based non-IID,
semantic clustering-based, and multi-criterion partitioning.

**Key Features:**

- Multiple heterogeneity strategies (IID, label-based, semantic, multi-criterion)
- Flexible validation set modes (local, global, none)
- Reproducible splits with random seeds
- JSON/pickle serialization for split persistence
- Integration with foundation models (DINOv2, CLIP, SigLIP, etc.)

Splitter Classes
----------------

.. autosummary::
   :toctree: generated/

   partitioning.BaseDatasetSplitter
   partitioning.RandomSplitter
   partitioning.DirichletSplitter
   partitioning.EmbeddingSplitter
   partitioning.MultivariateDirichletSplitter

Base Splitter
~~~~~~~~~~~~~

.. autoclass:: partitioning.BaseDatasetSplitter
   :members:
   :special-members: __init__
   :show-inheritance:
   :member-order: bysource
   :exclude-members: __weakref__

   Abstract base class providing common interface for all splitting strategies.

   **Key Methods:**

   - ``get_client_splits()``: Returns the complete split structure
   - ``get_split_summary()``: Returns statistics about the splits
   - ``save_splits(path, format='json')``: Saves splits to file
   - ``validate_splits(splits)``: Validates split integrity

IID Splitter
~~~~~~~~~~~~

.. autoclass:: partitioning.RandomSplitter
   :members:
   :special-members: __init__
   :show-inheritance:
   :member-order: bysource
   :exclude-members: __weakref__

   Randomly distributes data uniformly across clients (IID distribution).

   **Use Cases:**

   - Baseline experiments
   - Testing FL algorithms in ideal conditions
   - Benchmarking against non-IID scenarios

   **Example:**

   .. code-block:: python

      from partitioning import RandomSplitter

      splitter = RandomSplitter(
          train_dataset=train_data,
          val_dataset=val_data,
          n_clients=10,
          valset_type="global",
          seed=42
      )
      splits = splitter.get_client_splits()
      print(f"Client 0: {len(splits['clients']['client_0']['train'])} samples")

Label-based Non-IID Splitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: partitioning.DirichletSplitter
   :members:
   :special-members: __init__
   :show-inheritance:
   :member-order: bysource
   :exclude-members: __weakref__

   Creates label heterogeneity using Dirichlet distribution.

   **Heterogeneity Parameter (alpha):**

   - **Low (0.1-0.5)**: Highly heterogeneous, clients specialize in few classes
   - **Medium (1.0-5.0)**: Moderate heterogeneity, realistic imbalance
   - **High (10.0+)**: Nearly IID, approaches uniform distribution

   **Example:**

   .. code-block:: python

      from partitioning import DirichletSplitter

      splitter = DirichletSplitter(
          train_dataset=train_data,
          val_dataset=val_data,
          n_clients=10,
          heterogeneity=0.5,  # Medium heterogeneity
          valset_type="local",
          seed=42
      )
      splits = splitter.get_client_splits()
      summary = splitter.get_split_summary()
      print(summary)

Semantic Clustering Splitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: partitioning.EmbeddingSplitter
   :members:
   :special-members: __init__
   :show-inheritance:
   :member-order: bysource
   :exclude-members: __weakref__

   Partitions data based on semantic similarity using foundation model embeddings.

   **Supported Models:**

   - **DINOv2**: dinov2-small, dinov2-base, dinov2-large, dinov2-giant
   - **CLIP**: clip-vit-b-32, clip-vit-b-16, clip-vit-l-14
   - **SigLIP**: siglip-base-patch16-224, siglip-large-patch16-256
   - **Others**: InceptionV3, MAE

   **Clustering Methods:**

   - **K-Means**: Balanced clusters, requires exact n_clusters
   - **HDBSCAN**: Density-based, automatic cluster count

   **Example:**

   .. code-block:: python

      from partitioning import EmbeddingSplitter
      from partitioning.embedding_utils.config import EmbeddingConfig

      config = EmbeddingConfig(
          model_name="dinov2-base",
          pooling_strategy="cls",  # or "avg", "max"
          clustering_method="hdbscan"
      )

      splitter = EmbeddingSplitter(
          train_dataset=train_data,
          val_dataset=val_data,
          n_clients=10,
          heterogeneity=0.7,
          config=config,
          cache_dir="./.embedding_cache",
          seed=42
      )
      splits = splitter.get_client_splits()

Multi-Criterion VLM Splitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: partitioning.MultivariateDirichletSplitter
   :members:
   :special-members: __init__
   :show-inheritance:
   :member-order: bysource
   :exclude-members: __weakref__

   Advanced splitter using Vision-Language Model annotations for multi-criterion heterogeneity.

   **Features:**

   - Heterogeneity across multiple visual criteria (style, color, composition, lighting)
   - Requires VLM annotations (GPT-4V, Gemini, LLaVA)
   - Controlled by alpha parameter and preference weights

   **Example:**

   .. code-block:: python

      from partitioning import MultivariateDirichletSplitter

      # Requires VLM annotation file (JSONL format)
      splitter = MultivariateDirichletSplitter(
          train_dataset=train_data,
          val_dataset=val_data,
          n_clients=10,
          alpha=0.5,
          annotation_file="vlm_annotations.jsonl",
          criteria=["style", "color", "composition"],
          seed=42
      )
      splits = splitter.get_client_splits()

Type Definitions
----------------

.. autoclass:: partitioning.ClientSplit
   :members:
   :undoc-members:

   TypedDict defining a single client's data partition.

   **Structure:**

   .. code-block:: python

      {
          "train": [  # Training samples
              {"image": "path/to/img1.jpg", "label": 0},
              {"image": "path/to/img2.jpg", "label": 1},
          ],
          "val": [  # Validation samples (if local mode)
              {"image": "path/to/val1.jpg", "label": 0},
          ]
      }

.. autoclass:: partitioning.DatasetSplits
   :members:
   :undoc-members:

   TypedDict defining the complete split structure across all clients.

   **Structure:**

   .. code-block:: python

      {
          "clients": {
              "client_0": {"train": [...], "val": [...]},
              "client_1": {"train": [...], "val": [...]},
              ...
          },
          "global_val": [...]  # Shared validation (if global mode)
      }

Validation Set Modes
--------------------

All splitters support three validation set configurations:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``"local"``
     - Each client receives its own validation subset, proportional to training size
   * - ``"global"``
     - Single shared validation set used by all clients
   * - ``"none"``
     - No validation sets created

Usage Examples
--------------

Basic Workflow
~~~~~~~~~~~~~~

.. code-block:: python

   from partitioning import DirichletSplitter
   from data_loading.dataset import BaseDataset

   # Load dataset
   train_data = BaseDataset("tiny_imagenet", "/data/datasets", "train")
   val_data = BaseDataset("tiny_imagenet", "/data/datasets", "val")

   # Create splitter
   splitter = DirichletSplitter(
       train_dataset=train_data,
       val_dataset=val_data,
       n_clients=10,
       heterogeneity=0.5,
       valset_type="global",
       seed=42
   )

   # Generate splits
   splits = splitter.get_client_splits()

   # Access client data
   client_0_train = splits['clients']['client_0']['train']
   global_val = splits['global_val']

   print(f"Client 0 has {len(client_0_train)} training samples")
   print(f"Global validation has {len(global_val)} samples")

Saving and Loading Splits
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from partitioning import DirichletSplitter
   import json

   # Create and save splits
   splitter = DirichletSplitter(
       train_dataset=train_data,
       n_clients=10,
       heterogeneity=0.5
   )
   splits = splitter.get_client_splits()

   # Save to JSON (human-readable)
   splitter.save_splits("splits.json", format="json")

   # Save to pickle (faster, preserves Python objects)
   splitter.save_splits("splits.pkl", format="pickle")

   # Load splits later
   with open("splits.json", "r") as f:
       loaded_splits = json.load(f)

Analyzing Split Quality
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from partitioning import DirichletSplitter

   splitter = DirichletSplitter(
       train_dataset=train_data,
       n_clients=10,
       heterogeneity=0.5
   )
   splits = splitter.get_client_splits()

   # Get summary statistics
   summary = splitter.get_split_summary()
   print(f"Total samples: {summary['total_samples']}")
   print(f"Clients: {summary['n_clients']}")
   print(f"Validation mode: {summary['valset_type']}")

   # Validate split integrity
   is_valid = splitter.validate_splits(splits)
   assert is_valid, "Splits validation failed!"

Combining with Evaluation Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from partitioning import DirichletSplitter
   from evaluation.metrics import JS_divergence, FID_distance

   # Create splits
   splitter = DirichletSplitter(
       train_dataset=train_data,
       n_clients=10,
       heterogeneity=0.5
   )
   splits = splitter.get_client_splits()

   # Prepare client datasets
   client_datasets = [
       [train_data[i] for i in splits['clients'][f'client_{c}']['train']]
       for c in range(10)
   ]

   # Measure label heterogeneity
   jsd = JS_divergence(client_datasets, num_classes=200)
   print(f"Label divergence (JSD): {jsd:.4f}")

   # Measure feature heterogeneity
   fid = FID_distance(
       client_datasets,
       model_name="inception_v3",
       use_cache=True
   )
   print(f"Feature distance (FID): {fid:.4f}")

Advanced: Custom Validation Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from partitioning import RandomSplitter

   # Create IID splits with custom validation ratio
   splitter = RandomSplitter(
       train_dataset=train_data,
       val_dataset=val_data,
       n_clients=10,
       valset_type="local",
       val_ratio=0.2,  # 20% of each client's data for validation
       seed=42
   )
   splits = splitter.get_client_splits()

   # Each client now has local validation set
   for client_id in range(10):
       train_size = len(splits['clients'][f'client_{client_id}']['train'])
       val_size = len(splits['clients'][f'client_{client_id}']['val'])
       print(f"Client {client_id}: {train_size} train, {val_size} val")

See Also
--------

- :doc:`evaluation` - Metrics for measuring data heterogeneity
- :doc:`embedding` - Foundation models for feature extraction
- :doc:`data_loading` - Dataset loading utilities
