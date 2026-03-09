Quickstart
==========

This quickstart gets you from clone to basic split generation and FL data-module
loading with a reproducible environment preflight.

1. Install package + extras
---------------------------

.. code-block:: bash

   cd /path/to/FDS
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[partitioning,embedding,evaluation,fl,docs,dev]"

2. Run environment preflight
----------------------------

.. code-block:: bash

   ./scripts/preflight_env.sh

3. Generate a simple split
--------------------------

.. code-block:: bash

   python -m scripts.generate_embedding_partitions \
     --dataset tiny_imagenet \
     --base_path /path/to/data \
     --output ./results/splits.json \
     --num_clients 10

4. Load split in FL training
----------------------------

.. code-block:: python

   from data_loading.data_modules import get_datamodule

   dm = get_datamodule(
       dataset_name="tiny_imagenet",
       data_dir="/path/to/data",
       split_file="./results/splits.json",
       client_idx=0,
       batch_size=32,
   )
   dm.setup()
   batch = next(iter(dm.train_dataloader))
   print(type(batch))
