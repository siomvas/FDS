# FDS with NVIDIA FLARE

FDS trains its federations with [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) (NVFlare). Any FDS partition, whether random, label skew, geometric or semantic, becomes an NVFlare federation with one client per partition, without code changes. This page shows the integration on one short run and records its validation.

## What the integration uses

| Component | NVFlare functionality |
|---|---|
| `fl_training/fl_job.py` | builds a recipe from the Hydra configuration in `fl_training/conf/` and runs it in the simulation environment (`SimEnv`) |
| `fl_training/fed_methods.py` | NVFlare's `FedAvgRecipe`, which aggregates the clients' models with FedAvg |
| `fl_training/client_entry.py` | the training script of every client, written against the Client API (`nvflare.client`): it receives the global model, trains the task engine on the client's partition and sends the update back |
| `fl_training/engines/` | task engines for classification, semantic segmentation (ADE20K) and object detection (Pascal VOC) |

Each client reads its samples from an FDS split file (`clients.<id>.train`), and every client validates on the split's shared `global_val` set.

## Setup

Python 3.10 or newer, and PyTorch for CPU or CUDA.

```bash
git clone https://github.com/siomvas/fds.git && cd fds
pip install -e ".[fl,partitioning]"
python scripts/download_datasets.py --dataset tiny_imagenet --output-root /path/to/datasets
```

## Run

1. Build a semantic partition of Tiny ImageNet into four clients from the released criterion labels in `resources/vlm_labels/`:

```bash
python scripts/reproduce.py partitions --data-root /path/to/datasets --datasets tiny_imagenet \
  --methods vlm --n-clients 4 --alphas 1 --output-root /path/to/out
```

2. For a short CPU run, keep five validation images per class:

```bash
python examples/nvflare/subsample_global_val.py \
  /path/to/out/splits/tiny_imagenet/vlm_n4_alpha1_seed42.json \
  --output /path/to/out/splits/tiny_imagenet/vlm_n4_alpha1_seed42_val1k.json --per-class 5 --seed 42
```

3. Train five rounds of FedAvg, with 20 local steps per client and round:

```bash
DATASETS_ROOT=/path/to/datasets \
tiny_imagenet_chosen_split_path=/path/to/out/splits/tiny_imagenet/vlm_n4_alpha1_seed42_val1k.json \
python -m fl_training.run_experiment dataset=tiny_imagenet model=resnet18 \
  fed.n_clients=4 fed.rounds=5 training.steps_or_epochs=steps training.local_steps=20 \
  training.lr=0.05 training.amp_enabled=false 'system.gpu_ids=null' seed=42 \
  +dataset.args.num_workers=0 +fed.nvflare.num_threads=1 paths.output_dir=/path/to/fl_run
```

The learning rate is raised from the configured 0.01 so that five short rounds show learning. `system.gpu_ids=null`, `+dataset.args.num_workers=0` and `+fed.nvflare.num_threads=1` keep the run on CPU, in little memory, with one client at a time. On a GPU host, leave them out. The paper's experiments ran 100 rounds with one local epoch per round.

4. Evaluate the final global model on all 10,000 validation images:

```bash
DATASETS_ROOT=/path/to/datasets \
tiny_imagenet_chosen_split_path=/path/to/out/splits/tiny_imagenet/vlm_n4_alpha1_seed42.json \
python examples/nvflare/evaluate_global_model.py \
  /path/to/fl_run/tiny_imagenet_resnet18_fedavg_seed_42/tiny_imagenet_resnet18_fedavg_seed_42/server/simulate_job/app_server/FL_global_model.pt \
  dataset=tiny_imagenet model=resnet18 fed.n_clients=4 fed.rounds=5 \
  training.steps_or_epochs=steps training.local_steps=20 training.lr=0.05 \
  training.amp_enabled=false 'system.gpu_ids=null' seed=42 +dataset.args.num_workers=0
```

## Validation performed

Run on 16 November 2026, by the authors.

| Item | Value |
|---|---|
| fds commit tested | `5e6ab015179447be0351796a998d75a71f6f6a14`. The commit that adds this page and its two helper scripts changes no training code |
| NVFlare | 2.9.1 |
| Environment | Python 3.10.20, PyTorch 2.5.1 (CPU), torchvision 0.20.1, timm 1.0.31, Linux, two CPU cores, 7 GB of memory, no GPU |
| Data | Tiny ImageNet from `scripts/download_datasets.py`. Semantic partition into four clients (`vlm`, α = 1, seed 42) of 25,005, 25,037, 24,945 and 25,013 training images. Per-round validation on 1,000 of the 10,000 validation images, five per class |
| Commands | steps 1 to 4 above |
| Duration | 38.5 minutes for training, 3.7 minutes for the final evaluation |
| Outcome | `Job Status: FINISHED:COMPLETED`. The server ran five FedAvg rounds, received a result from each of the four sites in every round, and saved the global model as `FL_global_model.pt` |

Validation of each site's model after local training, on the 1,000 validation images, as top-1 accuracy with the validation loss in brackets. Chance is 0.5%, and its loss is ln 200 = 5.298.

| Round | site-1 | site-2 | site-3 | site-4 |
|---|---|---|---|---|
| 0 | 0.7% (5.411) | 1.4% (5.318) | 1.0% (5.365) | 0.7% (5.321) |
| 1 | 0.8% (5.383) | 1.8% (5.271) | 1.3% (5.302) | 0.7% (5.302) |
| 2 | 1.6% (5.314) | 2.0% (5.260) | 1.5% (5.275) | 1.2% (5.269) |
| 3 | 2.6% (5.244) | 3.6% (5.228) | 2.3% (5.246) | 2.2% (5.192) |
| 4 | 3.5% (5.223) | 3.4% (5.058) | 2.8% (5.139) | 3.2% (5.107) |

The final global model reaches a top-1 accuracy of 8.58% on all 10,000 validation images, with a loss of 4.921 (step 4).

## Limitations

- This is a functional check, not a benchmark: five rounds of 20 local steps on two CPU cores, with a learning rate raised for the purpose. It shows that an FDS partition trains end to end through the Recipe API, the Client API and the simulator, and that the global model learns. It says nothing about the accuracy a full run reaches.
- Only FedAvg and the classification engine were exercised. The segmentation and detection engines use the same job builder and client script, but were not part of this validation.
- Real deployments were not tested. The integration runs in NVFlare's simulation environment.

## License and citation

Apache 2.0, as the rest of the repository. Tiny ImageNet and the other datasets keep their own terms. To cite FDS, see the main [README](../../README.md#citation).
