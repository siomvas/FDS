Installation
============

Python Requirements
-------------------

- Python `>=3.10`
- `pip` and virtual environment support

Recommended Setup
-----------------

.. code-block:: bash

   cd /path/to/FDS
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[partitioning,embedding,evaluation,fl,docs,dev]"

Dependency Profiles
-------------------

- `partitioning`: splitters and clustering dependencies
- `embedding`: foundation-model embedding extraction dependencies
- `evaluation`: metric and plotting dependencies
- `fl`: FL training dependencies (Torch/NVFlare/model adapters)
- `docs`: documentation tooling
- `dev`: testing and local quality tooling

Environment Preflight
---------------------

Run the preflight script to confirm imports and baseline test collection:

.. code-block:: bash

   ./scripts/preflight_env.sh

The script checks:

- core Python import health
- package importability for data loading/partitioning/evaluation/embedding
- `pytest --collect-only` baseline readiness
