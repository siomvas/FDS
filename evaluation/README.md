# Evaluation Metrics

This directory contains evaluation metrics for measuring heterogeneity in federated learning data partitions.

## Files

- **`metrics.py`** - Core metric implementations
- **`metric_tests.py`** - Comprehensive unit tests (55 tests)
- **`validate_metrics_fixed.py`** - Validation script (use this one!)
- **`metric_validation_fixed.ipynb`** - Interactive notebook for presentations
- **`TESTING.md`** - Test documentation and guide

## ⚠️ Important: Splitter API Compatibility

The data partitioning splitters (`RandomSplitter`, `DirichletSplitter`) have a specific API:

### Splitter Input Format
```python
from data_partitioning.dataset import BaseDataset
from data_partitioning.random_splitter import RandomSplitter

# Load datasets (NOT PyTorch datasets!)
train_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "train")
val_dataset = BaseDataset("tiny_imagenet", "/path/to/data", "val")

# Create splitter
splitter = RandomSplitter(
    train_dataset=train_dataset,
    val_dataset=val_dataset,  # Required
    n_clients=5,
    heterogeneity=0.0,
    valset_type="global",  # or "local" or "none"
    seed=42
)
```

### Splitter Output Format
```python
splits = splitter.get_client_splits()

# Structure:
# {
#     "clients": {
#         "client_0": {
#             "train": [{"image": "/path/img1.jpg", "label": "n01234"}, ...],
#             "val": [...]
#         },
#         "client_1": {...},
#         ...
#     },
#     "global_val": [...] or None
# }

# Access client 0's training data:
client_0_train = splits["clients"]["client_0"]["train"]
# Returns: [{"image": path_string, "label": label_string}, ...]
```



## Testing

### Running Unit Tests

All metrics have comprehensive unit tests to ensure correctness:

```bash
# Install pytest
pip install pytest

# Run all fast tests (label metrics, ~15 seconds)
pytest metric_tests.py -v -m "not slow"

# Run all tests including feature metrics (~80 seconds)
pytest metric_tests.py -v

# Run specific test class
pytest metric_tests.py::TestKLDivergence -v
```

**What's tested**: 55 tests covering:
- Mathematical properties (self-distance=0, non-negativity, symmetry)
- Expected behavior (IID vs non-IID detection)
- Edge cases (empty datasets, single class, extreme skew)
- Numerical stability (zero counts, large samples)
- Integration (metrics agree on ordering)

See `TESTING.md` for detailed documentation.

## Usage

### Using the Validation Script

```bash
# For TinyImageNet
python validate_metrics_fixed.py \
    --dataset tiny_imagenet \
    --data_path /path/to/data \
    --num_clients 2 \
    --dirichlet_alpha 0.1 \
    --seed 42

# For Food101
python validate_metrics_fixed.py \
    --dataset food101 \
    --data_path /path/to/data \
    --num_clients 3 \
    --dirichlet_alpha 0.3

# Skip feature metrics for faster testing
python validate_metrics_fixed.py \
    --dataset tiny_imagenet \
    --data_path /path/to/data \
    --skip_feature_metrics
```

### Using the Interactive Notebook

```bash
jupyter notebook metric_validation_fixed.ipynb
```

Update the configuration cell with your dataset path:
```python
DATASET = 'tiny_imagenet'  # or 'food101', 'ade20k'
DATA_PATH = '/path/to/your/data'
DIRICHLET_ALPHA = 0.1
NUM_CLIENTS = 2
```

## Available Metrics

### Label Distribution Metrics
- **KL Divergence** - Kullback-Leibler divergence between class distributions
- **Kolmogorov-Smirnov Test** - Non-parametric two-sample test
- **MMD (Labels)** - Maximum Mean Discrepancy on label distributions
- **Mahalanobis Distance** - Distance accounting for covariance

### Feature Distribution Metrics
- **FID** - Fréchet Inception Distance (InceptionV3 features)
- **MMD (Features)** - MMD with RBF kernel on deep features
- **Wasserstein Distance** - Earth Mover's Distance averaged across dimensions
- **Cramér-von Mises** - Non-parametric test on feature distributions
- **KS Multivariate** - Multiple KS tests with Bonferroni correction

## Expected Behavior

When comparing Random (IID) vs Dirichlet (α=0.1) splits:

- **Random splits** → LOW heterogeneity scores (similar distributions)
- **Dirichlet splits** → HIGH heterogeneity scores (different distributions)
- **Ratio (Dirichlet/Random)** → Should be >> 1.0 (typically 2-10x or higher)

Example output:
```
✅ KL_divergence        Random: 0.012345    Dirichlet: 0.456789    Ratio: 37.03x
✅ KS_statistic         Random: 0.023456    Dirichlet: 0.234567    Ratio: 10.00x
✅ MMD_labels           Random: 0.034567    Dirichlet: 0.345678    Ratio: 10.00x
```

## Common Issues

### Issue 1: Dataset Not Found
```
FileNotFoundError: Base path /path/to/data does not exist
```
**Solution**: Make sure the dataset is downloaded and the path is correct.

### Issue 2: Wrong Dataset Structure
```
FileNotFoundError: Split path /path/to/tiny-imagenet-200/train does not exist
```
**Solution**: Check that the dataset follows the expected directory structure for your dataset.

### Issue 3: Out of Memory (GPU)
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `--max_samples_features` or use `--skip_feature_metrics`.

### Issue 4: Label Type Mismatch
If you see errors about label types, make sure you're using the `convert_to_metric_format()` function to convert string labels to integers.

## Adding New Metrics

To add a new metric:

1. Add the implementation to `metrics.py`
2. Add a call in the validation script's `run_label_distribution_metrics()` or `run_feature_distribution_metrics()`
3. Update this README

## Dataset Structure

### TinyImageNet
```
tiny-imagenet-200/
├── train/
│   ├── n01443537/
│   │   └── images/
│   │       ├── n01443537_0.JPEG
│   │       └── ...
│   └── ...
└── val/
    ├── images/
    │   └── val_*.JPEG
    └── val_annotations.txt
```

### Food101
```
food101/
├── images/
│   ├── apple_pie/
│   │   └── *.jpg
│   └── ...
└── meta/
    ├── train.json
    └── test.json
```

### ADE20K
```
ADE20K/
└── ADE20K_2021_17_01/
    └── images/
        └── ADE/
            ├── training/
            └── validation/
```

## Citation

If you use these metrics in your research, please cite our paper:

```bibtex
@inproceedings{fds2025,
  title={Federated Dataset Simulation},
  author={Your Name},
  booktitle={NeurIPS},
  year={2025}
}
```
