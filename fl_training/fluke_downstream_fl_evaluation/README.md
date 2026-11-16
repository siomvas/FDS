# fluke adapter

This package runs aggregation methods other than FedAvg on FDS partitions through
[fluke](https://github.com/makgyver/fluke). The paper compares seven of them on Tiny
ImageNet (Table 4): FedAvg, FedDyn, FedNova, FedProx, FedSAM, SCAFFOLD and FedLC. The rest
of the repository trains with NVFlare.

The adapter turns an FDS split file into a fluke federation with one fluke client per FDS
client, and evaluates the global model on the split's global validation set. A config
defines a matrix of cells (method, partition and number of local epochs), and one process
runs one cell, so the cells can run as an array job.

## Install

```bash
pip install -e ".[fluke]"
```

The extra pins fluke to a commit of its main branch, because the adapter relies on fixes
made after fluke's latest release.

## Configs

| Config | Model |
| --- | --- |
| `configs/tiny_imagenet_resnet18gn.yaml` | ResNet-18 with GroupNorm, trained from scratch |
| `configs/tiny_imagenet_resnet18_imagenet_bn.yaml` | ResNet-18 with BatchNorm, from ImageNet weights |

Both use 10 clients, 100 rounds, one local epoch, batch size 64 and SGD with learning rate
0.01, momentum 0.9 and no weight decay, on three partitions: Dirichlet label skew with
α = 0.1, the geometric partition, and the semantic partition with α = 0.1. They read four
environment variables:

| Variable | Meaning |
| --- | --- |
| `FDS_DATASETS_ROOT` | directory that holds `tiny-imagenet-200/` |
| `FDS_SPLITS_ROOT` | directory that holds the split files under `tiny_imagenet/` |
| `FDS_RUNS_ROOT` | directory for the results |
| `FDS_FLUKE_SEED` | seed of the run |

## Split files

The partition wrapper writes the Dirichlet and semantic splits under
`<output-root>/splits/`, with the names the configs expect:

```bash
python scripts/reproduce.py partitions \
  --data-root "$FDS_DATASETS_ROOT" \
  --datasets tiny_imagenet \
  --methods dirichlet,vlm \
  --n-clients 10 \
  --alphas 0.1 \
  --output-root "$FDS_OUTPUT_ROOT/reproduction"
export FDS_SPLITS_ROOT="$FDS_OUTPUT_ROOT/reproduction/splits"
```

The geometric split is an `EmbeddingSplitter` partition into 10 clients with a global
validation set, built with the representation and clustering settings given in the paper.
Save it as `$FDS_SPLITS_ROOT/tiny_imagenet/geometric_n10.json`.

## Run

```bash
CONFIG=fl_training/fluke_downstream_fl_evaluation/configs/tiny_imagenet_resnet18gn.yaml
python -m fl_training.fluke_downstream_fl_evaluation --config "$CONFIG" --list-tasks
python -m fl_training.fluke_downstream_fl_evaluation --config "$CONFIG" --task-index 0
```

`--list-tasks` prints the cells with their indices, and `--dry-run` checks the selected
cell without training. A cell that already finished with the same configuration is
skipped unless `--no-resume` is given. Each cell writes its results to its own directory
under `$FDS_RUNS_ROOT`.

## Changes to fluke's methods

`fluke_compat.py` corrects three of fluke's methods at the pinned commit, and every run
records the changes applied to its method:

- FedDyn: the dual variable accumulates with the sign and scale of the FedDyn paper.
- FedNova: the normalization counts the steps each client executed, with SGD momentum,
  and the server applies the normalized update to the previous global model.
- SCAFFOLD (option II): the control update is normalized by the effective number of steps
  under SGD momentum, and the momentum restarts at every local solve. Every client takes
  the same number of local steps, the local epochs times the median number of batches per
  epoch across clients, and the server weights each client's control update by its
  number of samples times its effective steps.

The last two SCAFFOLD changes were made after the results of the paper, so SCAFFOLD on
partitions with clients of unequal size, such as the geometric partition, can differ from
Table 4.
