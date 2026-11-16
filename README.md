# FDS: Federated Dataset Simulation

FDS is a federated dataset simulation package for studying visual client
heterogeneity in federated learning. It provides dataset loaders, splitters,
embedding extraction and caching, partition metrics, and FL training entrypoints
for the four datasets used in the paper.

This is the code of *Federated Dataset Simulation: Inducing Label-Free Heterogeneity
Across Tasks* (NeurIPS 2026, Evaluations and Datasets Track,
[OpenReview](https://openreview.net/forum?id=oIgrH55uFj)), released as reviewed,
with the artifacts of the paper's validation experiments and the code behind
them added (`resources/paper_artifacts/`). Replace every placeholder path below with paths on
your machine.

## Package Scope

The public package covers these components:

- `data_loading/`: dataset wrappers for classification, semantic segmentation,
  and object detection, plus task-specific data modules for FL training.
- `data_loading/dataset.py`: `BaseDataset`, a registry-backed dataset factory
  used by splitters and evaluation code.
- `data_loading/data_modules.py`: `get_datamodule(...)`, the FL-facing data
  module factory for Tiny ImageNet, Food-101, ADE20K, and Pascal VOC.
- `partitioning/`: IID random splits, label-skew Dirichlet splits,
  embedding-cluster splits, and VLM multi-criterion Dirichlet splits.
- `partitioning/vlm/`: optional VLM/LLM annotation pipeline used to generate
  criterion labels before they are consumed by the splitters.
- `embedding/`: `CacheManager`, `ModelRegistry`, and embedding extractors for
  foundation-model and non-neural feature representations.
- `evaluation/`: label-distribution and feature-distribution metrics for
  comparing federated partitions.
- `fl_training/`: Hydra/NVFlare-compatible FL training configs, model factory,
  and task engines for classification, segmentation, and detection.
- `resources/vlm_labels/`: the VLM criterion labels and prompts
  for reproducing VLM-based partitions without rerunning the expensive VLM
  annotation pipeline.
- `resources/paper_artifacts/`: Clevr-4 and Food-4c artifacts of the paper's
  validation experiments (Section 3.5, Figure 1 and Appendix F).
- `partitioning/vlm/CLEVR4.md`: the Clevr-4 validation of the semantic
  pipeline, with the code that produced those artifacts
  (`partitioning/vlm/clevr4_pipeline.py` and the `scripts/*clevr4*` entrypoints).
- `scripts/`: portable entrypoints for dataset preparation,
  reproduction, and environment checks.

The package intentionally excludes the old Sphinx docs tree, generated reports,
RunPod/HPC launch scripts, local result folders, cached datasets, and gradient
diversity metrics that were not part of the final public workflow.

## Supported Datasets

Set one dataset root and prepare the expected layout under it:

```bash
export DATASETS_ROOT=/path/to/datasets
export FDS_CACHE_DIR=/path/to/cache
export FDS_OUTPUT_ROOT=/path/to/outputs
```

| Dataset id | Task | Loader/data module | Expected location under `DATASETS_ROOT` |
| --- | --- | --- | --- |
| `tiny_imagenet` | Image classification | `TinyImageNetDataModule` | `tiny-imagenet-200/` |
| `food101` | Image classification | `Food101DataModule` | `food-101/` |
| `ade20k` | Semantic segmentation | `ADE20KDataModule` | `ADE20K/ADE20K_2021_17_01/` |
| `pascal_voc` | Object detection | `PascalVOCDataModule` | `VOC2012/` |

The lower-level `BaseDataset` registry also accepts aliases such as
`tiny-imagenet`, `pascal-voc`, and `PascalVOC`. Split JSON files keep the same
schema throughout the package:

```json
{
  "clients": {
    "client_0": {"train": [], "val": []},
    "client_1": {"train": [], "val": []}
  },
  "global_val": []
}
```

`global_val` is present only when the splitter is configured with a global
validation set.

## Installation

Use Python 3.10 or newer. The full install is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[partitioning,embedding,evaluation,fl,dev]"
```

Run the preflight gate:

```bash
./scripts/preflight_env.sh
```

The install extras are split so lightweight partitioning or evaluation work can
avoid the full FL stack:

```bash
python -m pip install -e ".[partitioning]"
python -m pip install -e ".[embedding,evaluation]"
python -m pip install -e ".[fl]"
```

## Dataset Setup

The downloader prepares Tiny ImageNet, Food-101, and Pascal VOC from public
dataset hosts:

```bash
python scripts/download_datasets.py --dataset all --output-root "$DATASETS_ROOT"
```

ADE20K requires official registration or approval. Download the official archive
manually, then normalize it with:

```bash
python scripts/download_datasets.py \
  --dataset ade20k \
  --output-root "$DATASETS_ROOT" \
  --ade20k-archive /path/to/ADE20K_2021_17_01.zip
```

Use `--dry-run` to inspect the planned work without downloading or writing:

```bash
python scripts/download_datasets.py --dataset all --output-root "$DATASETS_ROOT" --dry-run
```

## Dataset Factory And Data Modules

`BaseDataset(dataset_name, base_path, split)` loads a dataset split into image
paths, labels, and optional metadata. It is intentionally simple because it is
used by the splitters and metrics before model-specific transforms are needed.

`get_datamodule(...)` builds the training-facing data module for a central run
or a single federated client:

```python
from data_loading.data_modules import get_datamodule

dm = get_datamodule(
    "tiny_imagenet",
    data_dir="/path/to/datasets",
    split_file="/path/to/splits/tiny_imagenet/random_n10_seed42.json",
    client_idx=0,
    batch_size=64,
)
dm.setup()
train_loader = dm.train_dataloader()
```

The same factory handles segmentation masks for ADE20K and detection targets for
Pascal VOC, so FL training code can select datasets by config rather than by
task-specific branches.

## Partitioning

The public splitters produce the same client split schema:

- `RandomSplitter`: IID baseline split with optional local, global, or no
  validation set.
- `DirichletSplitter`: class-label non-IID split for classification datasets.
- `EmbeddingSplitter`: semantic split based on extracted image embeddings and
  clustering.
- `MultivariateDirichletSplitter`: VLM criterion split that combines multiple
  annotation dimensions such as object, color, lighting, scene, or style.

Generate the paper-scope splits with:

```bash
python scripts/reproduce.py partitions \
  --data-root "$DATASETS_ROOT" \
  --output-root "$FDS_OUTPUT_ROOT/reproduction"
```

Useful options:

```bash
python scripts/reproduce.py partitions \
  --datasets tiny_imagenet,food101 \
  --methods random,dirichlet,vlm \
  --n-clients 10 \
  --alphas 0.1,0.3,0.5,1.0,5.0 \
  --valset-type global \
  --dry-run
```

## Embeddings And Cache Management

`embedding.CacheManager` stores features in compressed `.npz` files and tracks
metadata in `metadata.json`. It supports two modes:

- Batch mode: cache one ad-hoc image list under a content-derived key.
- Dataset mode: cache an entire named dataset once, then retrieve subsets by
  path.

Dataset mode supports `base_path`, which stores relative paths in the cache so
the cache can move between machines:

```python
from embedding import CacheManager, ModelRegistry

cache = CacheManager(cache_dir="/path/to/cache")
extractor = ModelRegistry.create(
    "dinov2-small",
    device="cuda",
    batch_size=64,
    cache_manager=cache,
)
features = extractor.extract_embeddings(
    images=["/path/to/datasets/tiny-imagenet-200/train/n01443537/images/sample.JPEG"],
    use_cache=True,
    dataset_name="tiny_imagenet_train",
    base_path="/path/to/datasets",
)
```

Registered extractor families include InceptionV3, DINOv2, CLIP, SigLIP, MAE,
raw resized pixels, sparse random projections, an untrained autoencoder, and
wavelet scattering features.

## VLM Labels And Provenance

The VLM labels are stored under:

```text
resources/vlm_labels/
```

Each dataset contains one folder per criterion, for example
`resources/vlm_labels/food101/output_texture_train/`. Retained files are:

- `step3_result.jsonl`: sanitized train-split VLM criterion labels.
- `step2b_result.txt`: criterion class descriptions.
- `step1_prompt.txt`, `step2a_prompt.txt`, `step2b_prompt.txt`, `step3_prompt.txt`:
  prompts used by the VLM annotation process.

The splitter can read both `.jsonl` and `.jsonl.gz` step-3 files. The retained
plain JSONL files are used here so the labels can be opened directly
without archive-tool differences. Step-3 records contain only `image` and
`class`: `image` is a dataset-anchored relative path, and `class` is the
criterion label. Private absolute paths, source annotation paths, generated
descriptions, and local provenance fields are not retained. Regenerating VLM
labels is expensive and optional; the default reproduction path uses
the retained artifacts.

The optional generation pipeline under `partitioning/vlm/` expects local model
checkpoints and dataset roots to be supplied by environment variables or CLI
flags:

```bash
export FDS_VLM_MODEL_ROOT=/path/to/local/vlm_models
export DATASETS_ROOT=/path/to/datasets
python -m partitioning.vlm.full_pipeline \
  --dataset tiny_imagenet \
  --criteria main_object \
  --vlm-data-root /path/to/vlm_prompts \
  --output-root /path/to/vlm_outputs
```

## Evaluation Metrics

Partition metrics live in `evaluation/metrics.py` and operate on the generated
client split files. The current public surface includes:

- Label metrics: Jensen-Shannon divergence, total variation, and client-to-global
  label divergence.
- Feature metrics: FID, MMD, energy distance, sliced Wasserstein distance, KS
  testing, Vendi score, intrinsic dimensionality, k-NN separability, centroid
  distance, and feature variance.

Run the metric wrapper after generating splits:

```bash
python scripts/reproduce.py metrics \
  --data-root "$DATASETS_ROOT" \
  --output-root "$FDS_OUTPUT_ROOT/reproduction" \
  --cache-dir "$FDS_CACHE_DIR"
```

Use `--subsample` for a smaller smoke run, and use `--feature-model` to select a
different embedding extractor.

## Federated Training

`fl_training/` contains the FL runtime surface:

- `fl_training/conf/`: Hydra configs for datasets, models, paths, and FL rounds.
- `fl_training/models/factory.py`: model creation for classification,
  segmentation, and detection.
- `fl_training/engines/`: task engines for classification, ADE20K segmentation,
  and Pascal VOC detection.
- `fl_training/run_experiment.py`: main training entrypoint used by the
  reproduction wrapper.
- `examples/nvflare/`: a short NVFlare run on an FDS partition, with its
  validation record.

Run one FL command per generated split file:

```bash
python scripts/reproduce.py fl \
  --data-root "$DATASETS_ROOT" \
  --split-root "$FDS_OUTPUT_ROOT/reproduction/splits" \
  --output-root "$FDS_OUTPUT_ROOT/reproduction/fl" \
  --rounds 100 \
  --limit 1
```

The wrapper sets `DATASETS_ROOT` and the dataset-specific split-path environment
variable expected by Hydra, such as `tiny_imagenet_chosen_split_path`.

You can also call Hydra directly:

```bash
DATASETS_ROOT=/path/to/datasets \
tiny_imagenet_chosen_split_path=/path/to/splits/tiny_imagenet/random_n10_seed42.json \
python -m fl_training.run_experiment \
  dataset=tiny_imagenet \
  model=resnet18 \
  fed.n_clients=10 \
  fed.rounds=100 \
  paths.output_dir=/path/to/outputs/fl/tiny_imagenet_random
```

## Reproduction Shortcuts

Dry-run all public entrypoints:

```bash
python scripts/download_datasets.py --dataset all --output-root local_data --dry-run
python scripts/reproduce.py partitions --dry-run
python scripts/reproduce.py metrics --dry-run
python scripts/reproduce.py fl --dry-run
```

Run a small end-to-end partition and metric smoke test:

```bash
python scripts/reproduce.py partitions \
  --datasets tiny_imagenet \
  --methods random \
  --output-root "$FDS_OUTPUT_ROOT/smoke"

python scripts/reproduce.py metrics \
  --datasets tiny_imagenet \
  --output-root "$FDS_OUTPUT_ROOT/smoke" \
  --cache-dir "$FDS_CACHE_DIR" \
  --subsample 100
```

## Citation

GitHub's "Cite this repository" button reads `CITATION.cff`.

```bibtex
@inproceedings{siomos2026federated,
  title     = {Federated Dataset Simulation: Inducing Label-Free Heterogeneity Across Tasks},
  author    = {Siomos, Vasilis and Ngo, Lam and Passerat-Palmbach, Jonathan and Tarroni, Giacomo},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://openreview.net/forum?id=oIgrH55uFj}
}
```

## License

Apache License 2.0. See `LICENSE`.
