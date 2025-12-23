# Multivariate Dirichlet Client Partitioning - Usage Guide

## Overview

This implementation provides a novel federated learning data partitioning strategy that uses **multiple VLM-generated clustering criteria** (color, main_object, size, location, time) with Dirichlet-distributed preference vectors to create heterogeneous client distributions.

## Algorithm

1. **Preference Sampling**: For each client and each criterion, sample a preference vector from Dirichlet(α, α, ..., α)
   - Lower α (e.g., 0.1) → High heterogeneity (skewed preferences)
   - Higher α (e.g., 10.0) → Low heterogeneity (uniform preferences)

2. **Affinity Computation**: For each sample, compute affinity score to each client:
   ```
   Affinity(sample, client) = Σ(preference[criterion][sample's label]) across all 5 criteria
   ```

3. **Softmax Assignment**: Assign samples using softmax probabilities:
   ```
   P(sample → client_i) = exp(affinity_i) / Σ(exp(affinity_j))
   ```

## Quick Start

### 1. Basic Usage

```python
from data_partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from data_partitioning.dataset import BaseDataset

# Load dataset
train_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "train")
val_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "val")

# Create splitter
splitter = MultivariateDirichletSplitter(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_clients=10,
    heterogeneity=0.1,  # High heterogeneity
    cluster_labels_dir="data_partitioning/legacy_vlm/data/tiny_imagenet",
    valset_type="global",
    seed=42,
)

# Get splits
splits = splitter.get_client_splits()

# Access client data
client_0_train = splits["clients"]["client_0"]["train"]
# Format: [{"image": "/path/to/image.jpg", "label": "n01234567"}, ...]
```

### 2. Testing the Implementation

Run the validation tests to ensure everything works correctly:

```bash
python evaluation/test_multivariate_splitter.py
```

This will:
- Verify preference vectors sum to 1.0
- Test affinity score calculation
- Validate split constraints (no overlaps, complete coverage)
- Test heterogeneity control (α=0.1 vs α=10.0)

### 3. Running the Full Evaluation

Compare multivariate partitioning against baselines:

```bash
python evaluation/compare_partitioning_methods.py
```

This evaluates 3 methods across 8 configurations:
- **Methods**: MultivariateDirichlet, Random (IID), Dirichlet on original labels
- **Configurations**: n_clients ∈ {3, 6, 10, 20} × α ∈ {0.1, 10.0}

**Outputs**:
- `evaluation/results/partitioning_comparison.csv` - Tabular results
- `evaluation/results/partitioning_comparison.json` - Full results with metadata

## Parameters

### MultivariateDirichletSplitter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_dataset` | BaseDataset | - | Training dataset |
| `val_dataset` | BaseDataset | None | Validation dataset |
| `n_clients` | int | - | Number of clients |
| `heterogeneity` | float | - | Alpha parameter for Dirichlet (0.1=high, 10.0=low) |
| `cluster_labels_dir` | str | - | Path to VLM cluster labels |
| `valset_type` | str | "global" | "global", "local", or "none" |
| `seed` | int | None | Random seed |
| `min_samples_per_client` | int | 10 | Minimum samples per client |
| `max_retries` | int | 1000 | Max attempts to satisfy constraints |
| `temperature` | float | 1.0 | Softmax temperature |

## File Structure

```
data_partitioning/
├── multivariate_dirichlet_splitter.py  # Main implementation
├── MULTIVARIATE_USAGE.md               # This file
└── legacy_vlm/data/tiny_imagenet/      # Cluster labels (read-only)
    ├── output_color_test/step3_result.jsonl
    ├── output_main_object_test/step3_result.jsonl
    ├── output_size_test/step3_result.jsonl
    ├── output_location_test/step3_result.jsonl
    └── output_time_test/step3_result.jsonl

evaluation/
├── test_multivariate_splitter.py       # Validation tests
├── compare_partitioning_methods.py     # Full evaluation script
└── results/                            # Output directory
    ├── partitioning_comparison.csv
    └── partitioning_comparison.json
```

## Cluster Label Format

The VLM-generated cluster labels are stored in JSONL format:

```json
{
  "image": "/path/to/image.JPEG",
  "label": "n03444034",
  "description": "A red car driving on a highway",
  "original_index": 0,
  "class": "Vehicle"
}
```

**Available Criteria**:
1. **Color**: Brown/Yellow, Red, Green, Blue, Black, White, Other (7 classes)
2. **Main Object**: Human, Animal, Vehicle, Building, Food, Landscape, Inanimate Object (7 classes)
3. **Size**: Small, Large, Medium, Expansive (4 classes)
4. **Location**: Indoors, Urban Outdoors, Nature Outdoors, Aquatic, Unknown (5 classes)
5. **Time**: Day, Night, Unknown, Sunset/Dusk (4 classes)

## Evaluation Metrics

The evaluation script computes:

### Label Distribution Metrics
- **KL Divergence**: Average pairwise KL divergence between client label distributions
- **MMD**: Maximum Mean Discrepancy on label distributions

### Balance Metrics
- Mean/std/min/max samples per client
- Coefficient of variation (CV) of samples per client

### Cluster Distribution Metrics (Multivariate method only)
- Average entropy of cluster distributions per criterion
- Shows how diverse each client is in terms of cluster membership

## Expected Results

Based on the algorithm design:

1. **MultivariateDirichlet with α=0.1** (high heterogeneity):
   - Higher KL divergence and MMD (more heterogeneous)
   - Each client strongly prefers certain cluster combinations
   - Higher variance in samples per client

2. **MultivariateDirichlet with α=10.0** (low heterogeneity):
   - Lower KL divergence and MMD (approaching IID)
   - More uniform preferences across clients
   - Lower variance in samples per client

3. **Random (IID)**:
   - Lowest KL divergence and MMD
   - Nearly equal samples per client
   - Baseline for comparison

4. **Dirichlet on original labels**:
   - Intermediate heterogeneity
   - Similar to MultivariateDirichlet but uses single criterion (original ImageNet labels)

## HPC Deployment Tips

### 1. Environment Setup

Ensure you have the required dependencies:
```bash
pip install numpy scipy torch torchvision pillow alibi-detect pandas
```

### 2. Path Configuration

Update paths in the evaluation scripts:
- `BASE_PATH`: Path to TinyImageNet data on HPC
- `CLUSTER_LABELS_DIR`: Path to VLM cluster labels (legacy_vlm/data/tiny_imagenet)

### 3. Resource Requirements

- **Memory**: ~8GB for TinyImageNet with 20 clients
- **Time**: ~5-10 minutes per configuration (depends on dataset size)
- **GPU**: Not required for partitioning (only if computing FID metrics)

### 4. Batch Job Example

```bash
#!/bin/bash
#SBATCH --job-name=multivariate_eval
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.9
source /path/to/venv/bin/activate

cd /path/to/FDS
python evaluation/compare_partitioning_methods.py
```

## Troubleshooting

### Issue: "Unable to satisfy min_samples_per_client"
**Solution**: Lower `min_samples_per_client` or increase `max_retries`

### Issue: "Cluster labels not found"
**Solution**: Verify `cluster_labels_dir` path points to correct location with step3_result.jsonl files

### Issue: Skewed client sizes
**Solution**: Increase heterogeneity parameter (higher α) for more balanced splits

## Advanced Usage

### Custom Criterion Weights

To modify which criteria are most important, you can adjust the code in `_compute_affinity_score()`:

```python
# Example: Weight main_object more heavily
criterion_weights = {
    "color": 0.1,
    "main_object": 0.5,
    "size": 0.1,
    "location": 0.2,
    "time": 0.1,
}

affinity = sum(
    criterion_weights[criterion] * preference
    for criterion in self.CRITERIA
)
```

### Analyzing Client Preferences

```python
# Get preference summary
summary = splitter.get_preference_summary()

# Show top preferences for each criterion
for criterion in splitter.CRITERIA:
    print(f"\n{criterion} preferences:")
    for client_data in summary[f"{criterion}_top_preferences"]:
        print(f"  Client {client_data['client']}: {client_data['label']} "
              f"({client_data['preference']:.3f})")
```

## References

- **Base Splitter**: `data_partitioning/base_splitter.py`
- **Dirichlet Splitter**: `data_partitioning/dirichlet_splitter.py`
- **Evaluation Metrics**: `evaluation/metrics.py`
- **Dataset Loader**: `data_partitioning/dataset.py`
