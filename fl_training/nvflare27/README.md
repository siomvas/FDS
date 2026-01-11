# FDS NVFlare 2.7 Federated Learning

**Complete federated learning system integrating FDS data partitioning with NVFlare 2.7 Client API.**

This system enables researchers to run FL experiments from YAML configuration to results without manual NVFlare setup. It supports all three task types (classification, segmentation, detection) with sophisticated data heterogeneity strategies.

---

## Overview

**Architecture:**
```
YAML Config → Job Generation → NVFlare Execution → Results Collection
     ↓              ↓                  ↓                    ↓
FDSExperimentConfig  JSON Job    train.py (Learners)   TensorBoard + Models
```

**Key Components:**
- **Learners**: Task-specific training logic (ClassificationLearner, SegmentationLearner, DetectionLearner)
- **Config System**: User-friendly YAML → NVFlare JSON translation
- **Experiment Runner**: End-to-end orchestration with results collection
- **Client Script**: Unified entry point for NVFlare ScriptExecutor

**Research Value:**
- Seamless integration with FDS partitioning strategies (IID, Dirichlet, embedding-based, VLM-based multivariate)
- Gradient diversity tracking for analyzing federated learning dynamics
- Reproducible experiments with automatic seeding and configuration management
- Production-ready architecture for HPC deployment

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch (tested with 2.0+)
- CUDA (optional, for GPU acceleration)

### Install FDS Package

```bash
# Clone repository
git clone <repository-url>
cd FDS

# Install in editable mode (includes pyyaml dependency)
pip install -e .
```

### Install NVFlare

```bash
# NVFlare 2.7 for FL execution
pip install nvflare~=2.7.0

# Verify installation
nvflare --version
```

### Additional Dependencies (Task-Specific)

```bash
# For segmentation (ADE20K)
pip install pillow

# For detection (COCO)
pip install pycocotools

# For visualization
pip install tensorboard matplotlib
```

---

## Quick Start

### 1. Create Experiment Configuration

Create a YAML file defining your FL experiment:

```yaml
# my_experiment.yaml
experiment:
  name: "tiny_imagenet_fedavg_baseline"
  task_type: "classification"
  dataset: "tiny_imagenet"
  dataset_splits: "outputs/tiny_imagenet_dirichlet_alpha0.1.json"

federated_learning:
  num_rounds: 10
  num_clients: 10
  aggregator: "fedavg"
  min_clients: 8
  wait_time: 10

training:
  model: "resnet18"
  local_epochs: 1
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  amp_enabled: true
  grad_tracking: true
  seed: 42

resources:
  client_gpu_mem: "4GB"
  server_cpu: 2
```

**Important:** Generate dataset splits first using FDS partitioning:

```python
from partitioning import DirichletSplitter
from data_loading.dataset import BaseDataset

# Load dataset
train_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "train")
val_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "val")

# Create heterogeneous splits
splitter = DirichletSplitter(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_clients=10,
    heterogeneity=0.1,  # alpha=0.1 → high heterogeneity
    valset_type="global",
    seed=42
)

# Save splits for FL training
splitter.store_client_splits("outputs/tiny_imagenet_dirichlet_alpha0.1.json")
```

### 2. Run Experiment

```bash
# Run in simulator mode (local testing)
python fl_training/nvflare27/run_experiment.py \
    --config my_experiment.yaml \
    --mode simulator \
    --workspace ./nvflare_workspace
```

**Output:**
```
================================================================================
FDS Federated Learning Experiment Runner
================================================================================

[1/5] Checking NVFlare installation...
  ✓ NVFlare installed

[2/5] Loading experiment config: my_experiment.yaml
  ✓ Experiment: tiny_imagenet_fedavg_baseline
  ✓ Task type: classification
  ✓ Dataset: tiny_imagenet

[3/5] Generating NVFlare job config...
  ✓ Job config written to: ./nvflare_workspace/jobs/tiny_imagenet_fedavg_baseline_job_config.json

[4/5] Running FL simulation (10 clients, 10 rounds)...
  [NVFlare training logs stream here...]
  ✓ Simulation complete

[5/5] Collecting results...
  ✓ TensorBoard logs: results/tiny_imagenet_fedavg_baseline/2026-01-11_15-30-45/tensorboard
  ✓ Model checkpoint: results/tiny_imagenet_fedavg_baseline/2026-01-11_15-30-45/model/server_model.pt
  ✓ Experiment summary: results/tiny_imagenet_fedavg_baseline/2026-01-11_15-30-45/experiment_summary.json

================================================================================
✓ Experiment complete!
================================================================================
```

### 3. View Results

```bash
# TensorBoard visualization
tensorboard --logdir results/tiny_imagenet_fedavg_baseline/2026-01-11_15-30-45/tensorboard

# Experiment summary
cat results/tiny_imagenet_fedavg_baseline/2026-01-11_15-30-45/experiment_summary.json

# Load final model
python
>>> import torch
>>> model = torch.load("results/.../model/server_model.pt")
```

---

## YAML Configuration

See [`config/examples/README.md`](config/examples/README.md) for detailed documentation and examples for all three task types.

**Configuration Sections:**

| Section | Description | Key Fields |
|---------|-------------|------------|
| `experiment` | Experiment metadata | `name`, `task_type`, `dataset`, `dataset_splits` |
| `federated_learning` | FL parameters | `num_rounds`, `num_clients`, `aggregator` |
| `training` | Training hyperparameters | `model`, `local_epochs`, `batch_size`, `learning_rate` |
| `resources` | Resource hints | `client_gpu_mem`, `server_cpu` |

**Example Configs:**
- [Classification (Tiny ImageNet)](config/examples/classification_example.yaml)
- [Segmentation (ADE20K)](config/examples/segmentation_example.yaml)
- [Detection (COCO)](config/examples/detection_example.yaml)

---

## Running Experiments

### Simulator Mode (Local Testing)

Best for development and debugging:

```bash
python fl_training/nvflare27/run_experiment.py \
    --config my_experiment.yaml \
    --mode simulator \
    --workspace ./nvflare_workspace
```

**Features:**
- Runs all clients on single machine
- Real-time logging output
- Automatic results collection
- Fast iteration for hyperparameter tuning

### Production Mode (Multi-Node HPC)

For large-scale experiments on HPC cluster:

```bash
python fl_training/nvflare27/run_experiment.py \
    --config my_experiment.yaml \
    --mode production \
    --workspace /shared/nvflare_workspace
```

**Output:** Instructions for manual deployment:
1. Provision NVFlare infrastructure
2. Deploy job to server
3. Start server and clients on compute nodes
4. Monitor progress via NVFlare CLI

See [HPC Deployment](#hpc-deployment) section below.

### Override Parameters

```bash
# Override number of clients from YAML
python fl_training/nvflare27/run_experiment.py \
    --config my_experiment.yaml \
    --mode simulator \
    --num_clients 5

# Custom results directory
python fl_training/nvflare27/run_experiment.py \
    --config my_experiment.yaml \
    --mode simulator \
    --results_dir ./my_results
```

---

## Task-Specific Examples

### Classification (Tiny ImageNet, 200 classes)

```yaml
experiment:
  task_type: "classification"
  dataset_splits: "outputs/tiny_imagenet_splits.json"

training:
  model: "resnet18"  # Options: vgg11, vgg13, vgg16, resnet18, resnet34, resnet50
  batch_size: 32
  learning_rate: 0.001
```

**Models Supported:** VGG family, ResNet family (via model_factory)

**Metrics Logged:** `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`

### Segmentation (ADE20K, 151 classes)

```yaml
experiment:
  task_type: "segmentation"
  dataset_splits: "outputs/ade20k_splits.json"

training:
  model: "deeplabv3_resnet50"
  batch_size: 8  # Smaller due to dense predictions
  learning_rate: 0.001
```

**Models Supported:** DeepLabV3 ResNet50 (extendable to other architectures)

**Metrics Logged:** `train_loss`, `val_loss`, `val_mIoU` (mean Intersection over Union)

### Detection (COCO, 80 classes)

```yaml
experiment:
  task_type: "detection"
  dataset_splits: "outputs/coco_splits.json"

training:
  model: "fasterrcnn_resnet50"
  batch_size: 4  # Smallest due to bounding box computations
  learning_rate: 0.001
  amp_enabled: false  # AMP can cause instability in detection
```

**Models Supported:** Faster R-CNN ResNet50 (via model_factory)

**Metrics Logged:** `train_loss`, `val_loss`, `val_mAP` (mean Average Precision)

---

## Understanding Results

### Results Directory Structure

```
results/
└── {experiment_name}/
    └── {timestamp}/
        ├── tensorboard/          # TensorBoard event files
        │   ├── server/
        │   └── site-1/
        │       └── ...
        ├── model/                # Final model checkpoints
        │   └── server_model.pt
        └── experiment_summary.json  # Experiment metadata
```

### TensorBoard Visualization

```bash
tensorboard --logdir results/{experiment_name}/{timestamp}/tensorboard
```

**Available Metrics:**
- **Per-Client:** `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` (or task-specific)
- **Server:** Aggregated validation metrics, client participation
- **Gradient Tracking** (if enabled): `grad_diversity_mean`, `grad_conflict_rate`, `grad_alignment_cosine`

### Experiment Summary JSON

```json
{
  "experiment_name": "tiny_imagenet_fedavg_baseline",
  "task_type": "classification",
  "dataset": "tiny_imagenet",
  "timestamp": "2026-01-11_15-30-45",
  "mode": "simulator",
  "configuration": {
    "experiment": {...},
    "federated_learning": {...},
    "training": {...}
  },
  "results_location": "/path/to/results/..."
}
```

### Loading Final Model

```python
import torch

# Load checkpoint
checkpoint = torch.load("results/.../model/server_model.pt")

# If saved as state_dict
model = get_model("resnet18", num_classes=200)
model.load_state_dict(checkpoint)

# If saved as full model
model = checkpoint
```

---

## HPC Deployment

### Proxy Configuration

City University HPC cluster requires proxy for model downloads:

```bash
# Set environment variable before running experiments
export FDS_HPC_PROXY="http://proxy.example.org:3128"

# Or use custom proxy
export FDS_HPC_PROXY="http://custom-proxy:8080"
```

The `hpc_proxy()` context manager in learners automatically uses this setting.

### Production Deployment Steps

1. **Provision Infrastructure:**

   ```bash
   # Create NVFlare project configuration
   nvflare provision -p project.yml
   ```

2. **Generate Job Config:**

   ```bash
   python fl_training/nvflare27/run_experiment.py \
       --config my_experiment.yaml \
       --mode production \
       --workspace /shared/nvflare_workspace
   ```

3. **Deploy to Server:**

   ```bash
   nvflare job submit /shared/nvflare_workspace/jobs/my_experiment_job_config.json
   ```

4. **Start Server:**

   ```bash
   cd startup && ./start.sh
   ```

5. **Start Clients (on each compute node):**

   ```bash
   # On node 1
   cd startup/site-1 && ./start.sh

   # On node 2
   cd startup/site-2 && ./start.sh

   # Repeat for all clients...
   ```

6. **Monitor Progress:**

   ```bash
   nvflare job list
   nvflare job show <job_id>
   ```

**For detailed production setup, see:**
- [NVFlare Production Deployment Guide](https://nvidia.github.io/NVFlare/user_guide/production_deployment.html)
- [NVFlare Provisioning](https://nvidia.github.io/NVFlare/user_guide/provisioning.html)

---

## Architecture Reference

### Component Overview

```
Client-Side (train.py):
  ├── Argument Parsing → Config Dict
  ├── Learner Instantiation (task-specific)
  │   ├── ClassificationLearner
  │   ├── SegmentationLearner
  │   └── DetectionLearner
  └── NVFlare Client API Integration
      ├── flare.init()
      ├── flare.receive() → global model
      ├── learner.execute() → local training
      └── flare.send() → updated model

Server-Side (NVFlare):
  ├── FDSFedAvgController
  │   ├── Model aggregation (FedAvg)
  │   └── Round coordination
  └── TensorBoard logging

Shared Infrastructure:
  ├── FDSDataFactory → DataLoaders from splits
  ├── ClientGradientTracker → gradient metrics
  └── Model factories (classification, segmentation, detection)
```

### Learner Hierarchy

**BaseLearner** (abstract base class):
- NVFlare Client API lifecycle
- Gradient tracking integration
- TensorBoard logging
- AMP support
- Data loading via FDSDataFactory

**Task-Specific Learners** (inherit from BaseLearner):
- **ClassificationLearner**: Cross-entropy loss, accuracy metrics
- **SegmentationLearner**: Pixel-wise CE loss, mIoU metrics
- **DetectionLearner**: RPN + detection losses, mAP metrics

### Data Flow

1. **Partitioning** (pre-experiment):
   ```
   BaseDataset → Splitter → DatasetSplits JSON
   ```

2. **FL Training** (per round):
   ```
   Server: Aggregate → Broadcast global model
   Clients: Receive → FDSDataFactory → train_loader, val_loader → Train → Evaluate → Send updates
   ```

3. **Results** (post-experiment):
   ```
   NVFlare workspace → Collect (TensorBoard, models) → Organized results directory
   ```

---

## Troubleshooting

### NVFlare Not Found

```
ERROR: NVFlare not found
```

**Solution:**
```bash
pip install nvflare~=2.7.0
nvflare --version
```

### YAML Syntax Error

```
ERROR: Failed to load configuration
```

**Solution:**
- Check YAML syntax (indentation, colons, quotes)
- Verify all required fields present (experiment, federated_learning, training)
- Compare with examples in `config/examples/`

### Dataset Splits Not Found

```
FileNotFoundError: DatasetSplits file not found
```

**Solution:**
- Generate splits first using FDS partitioning (see Quick Start step 1)
- Verify path in YAML config matches actual file location
- Use absolute paths to avoid confusion

### Model Download Fails (HPC)

```
ConnectionError: Failed to download pretrained weights
```

**Solution:**
```bash
# Set HPC proxy
export FDS_HPC_PROXY="http://proxy.example.org:3128"

# Verify proxy is set
echo $FDS_HPC_PROXY

# Re-run experiment
python fl_training/nvflare27/run_experiment.py --config my_experiment.yaml --mode simulator
```

### GPU Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch_size in YAML config
- Enable `amp_enabled: true` for mixed precision
- Use smaller model architecture
- Detection: reduce batch_size to 2-4
- Segmentation: reduce batch_size to 4-8

---

## Next Steps

1. **Generate Dataset Splits:**
   ```bash
   # See partitioning/ for all splitting strategies
   # IID, Dirichlet, embedding-based, VLM-based multivariate
   ```

2. **Create YAML Config:**
   ```bash
   cp fl_training/nvflare27/config/examples/classification_example.yaml my_experiment.yaml
   # Edit paths and hyperparameters
   ```

3. **Run Experiment:**
   ```bash
   python fl_training/nvflare27/run_experiment.py --config my_experiment.yaml --mode simulator
   ```

4. **Analyze Results:**
   ```bash
   tensorboard --logdir results/{experiment_name}/{timestamp}/tensorboard
   ```

5. **Compare Partitioning Strategies:**
   ```python
   # Run experiments with different heterogeneity levels
   # Compare gradient diversity, convergence, final accuracy
   ```

---

## References

- **NVFlare Documentation:** https://nvidia.github.io/NVFlare/
- **NVFlare Client API:** https://nvidia.github.io/NVFlare/programming_guide/client_api.html
- **FDS Partitioning:** See `docs/MULTIVARIATE_USAGE.md` and `docs/README_embeddings.md`
- **FL Training Refactor Summary:** See `docs/FL_TRAINING_REFACTOR_SUMMARY.md`

---

## License

See repository LICENSE file.

---

**Questions or Issues?**
Open an issue on the repository or contact the development team.
