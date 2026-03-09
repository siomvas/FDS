# FDS

FDS is a federated-learning data simulation and evaluation toolkit with:

- dataset loading/data modules for FL training
- IID and non-IID partitioning strategies
- embedding-based partitioning support
- partition evaluation metrics and reporting
- FL training integration modules

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[partitioning,embedding,evaluation,fl,docs,dev]"
```

## Environment Preflight

```bash
./scripts/preflight_env.sh
```

This checks import health and runs `pytest --collect-only` as a baseline gate.

## Docs

Sphinx source lives under `docs/source`.

- Start from [`docs/source/index.rst`](docs/source/index.rst)
- Data loading API: [`docs/source/api/data_loading.rst`](docs/source/api/data_loading.rst)

## Core Entry Points

- Data modules: `data_loading/data_modules.py`
- Dataset registry/loaders: `data_loading/dataset.py`
- Splitters: `partitioning/`
- Model factory: `fl_training/models/factory.py`
- FL launch entrypoint: `fl_training/run_experiment.py`
