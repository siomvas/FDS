# Quick Start Guide - Metric Validation

## Prerequisites

Make sure you have one of the supported datasets downloaded:
- **TinyImageNet** - 200 classes, 100K train images
- **Food101** - 101 food categories, 75K train images
- **ADE20K** - Scene segmentation dataset

## Option 1: Python Script (Recommended for Batch Processing)

### Basic Usage

```bash
cd evaluation

# Run with TinyImageNet
python validate_metrics_fixed.py \
    --dataset tiny_imagenet \
    --data_path /path/to/your/data \
    --num_clients 2 \
    --dirichlet_alpha 0.1 \
    --seed 42
```

### Quick Test (Skip Feature Metrics)

```bash
# Much faster - only tests label distribution metrics
python validate_metrics_fixed.py \
    --dataset tiny_imagenet \
    --data_path /path/to/your/data \
    --skip_feature_metrics
```

### With Different Heterogeneity Levels

```bash
# Low heterogeneity (α=0.5) - more balanced
python validate_metrics_fixed.py \
    --dataset tiny_imagenet \
    --data_path /path/to/your/data \
    --dirichlet_alpha 0.5

# High heterogeneity (α=0.01) - very skewed
python validate_metrics_fixed.py \
    --dataset tiny_imagenet \
    --data_path /path/to/your/data \
    --dirichlet_alpha 0.01
```

### All Options

```bash
python validate_metrics_fixed.py \
    --dataset tiny_imagenet \
    --data_path /path/to/your/data \
    --num_clients 3 \
    --dirichlet_alpha 0.1 \
    --seed 42 \
    --device cuda \
    --max_samples_features 200 \
    --skip_feature_metrics
```

### Expected Output

```
================================================================================
METRIC VALIDATION SCRIPT (FIXED)
================================================================================
Dataset: tiny_imagenet
Data path: /path/to/your/data
Number of clients: 2
Dirichlet alpha: 0.1
Seed: 42
Device: cuda
================================================================================

✓ Loaded tiny_imagenet: 100000 train samples, 10000 val samples
✓ Number of classes: 200

...

================================================================================
SUMMARY: RANDOM vs DIRICHLET COMPARISON
================================================================================

Expected: Dirichlet metrics >> Random metrics (more heterogeneous)

Label Distribution Metrics:
Metric                         Random          Dirichlet       Ratio (D/R)
---------------------------------------------------------------------------
✅ KL_divergence               0.012345        0.456789        37.03
✅ KS_statistic                0.023456        0.234567        10.00
✅ MMD_label_distributions     0.034567        0.345678        10.00
✅ mahalanobis_distance        0.045678        0.456789        10.00

...

✓ Results saved to metric_validation_tiny_imagenet_alpha0.1.json
```

## Option 2: Jupyter Notebook (Recommended for Presentations)

### Step 1: Start Jupyter

```bash
cd evaluation
jupyter notebook
```

### Step 2: Open the Notebook

Open `metric_validation_fixed.ipynb` in your browser

### Step 3: Configure

In the first code cell, update these settings:

```python
# Configuration
DATASET = 'tiny_imagenet'  # Options: 'tiny_imagenet', 'food101', 'ade20k'
DATA_PATH = '/path/to/your/data'  # ⚠️ UPDATE THIS!
NUM_CLIENTS = 2
DIRICHLET_ALPHA = 0.1  # Lower = more heterogeneous
SEED = 42
SKIP_FEATURE_METRICS = False  # Set to True for faster execution
MAX_SAMPLES_FEATURES = 100  # Subsample for feature metrics
```

### Step 4: Run All Cells

Click "Cell" → "Run All" or press `Shift+Enter` through each cell

### What You'll See

1. **Dataset loading** - Confirmation that dataset loaded successfully
2. **Split creation** - Random and Dirichlet splits created
3. **Label distribution plots** - Visual comparison of class distributions
4. **Metric results** - Tables showing all metric values
5. **Comparison charts** - Bar charts showing Random vs Dirichlet
6. **Validation summary** - Pass/fail for each metric

## Dataset Path Examples

### TinyImageNet
```bash
/path/to/data/
└── tiny-imagenet-200/
    ├── train/
    │   ├── n01443537/
    │   │   └── images/
    │   │       └── *.JPEG
    │   └── ...
    └── val/
        ├── images/
        │   └── *.JPEG
        └── val_annotations.txt
```

### Food101
```bash
/path/to/data/
└── food101/
    ├── images/
    │   ├── apple_pie/
    │   │   └── *.jpg
    │   └── ...
    └── meta/
        ├── train.json
        └── test.json
```

### ADE20K
```bash
/path/to/data/
└── ADE20K/
    └── ADE20K_2021_17_01/
        └── images/
            └── ADE/
                ├── training/
                └── validation/
```

## Common Issues

### Issue: Dataset Not Found

```
FileNotFoundError: Base path /path/to/data does not exist
```

**Solution**:
1. Check the path is correct
2. Make sure dataset is downloaded
3. Verify directory structure matches expected format

### Issue: GPU Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**:
```bash
# Reduce batch size or samples
python validate_metrics_fixed.py \
    --dataset tiny_imagenet \
    --data_path /path/to/data \
    --max_samples_features 50 \
    --device cpu
```

### Issue: Import Errors

```
ModuleNotFoundError: No module named 'data_partitioning'
```

**Solution**:
```bash
# Make sure you're in the evaluation directory
cd evaluation

# Or run from project root
python evaluation/validate_metrics_fixed.py ...
```

## Interpreting Results

### Good Results ✅
```
✅ KL_divergence        Random: 0.012    Dirichlet: 0.456    Ratio: 38.0x
✅ MMD_labels           Random: 0.034    Dirichlet: 0.345    Ratio: 10.1x
```
- Dirichlet >> Random (ratios > 2x)
- Metrics correctly detect heterogeneity

### Problematic Results ⚠️
```
❌ KL_divergence        Random: 0.456    Dirichlet: 0.123    Ratio: 0.27x
```
- Random > Dirichlet (inverted!)
- Metric may be incorrectly implemented
- Or data issue

## Next Steps

Once validation passes:

1. **Apply to VLM splits**:
   ```python
   # Load VLM-generated splits
   vlm_splits = load_vlm_splits('output_main_object_test.jsonl')

   # Convert to metric format
   vlm_c0, _ = convert_to_metric_format(vlm_splits['client_0'])
   vlm_c1, _ = convert_to_metric_format(vlm_splits['client_1'])

   # Run metrics
   kl_div = metrics.KL_divergence(vlm_c0, vlm_c1, num_classes)
   ```

2. **Compare splitting strategies**:
   - Random (IID baseline)
   - Dirichlet (label-based heterogeneity)
   - VLM-based (feature-based heterogeneity)

3. **Generate paper results**:
   - Quantitative heterogeneity comparison table
   - Visual distribution comparisons
   - Statistical significance tests

## Getting Help

- Check `README.md` for detailed API documentation
- See `COMPATIBILITY_ISSUES.md` for troubleshooting
- Review `metrics.py` for metric implementations
- Open an issue if you find bugs

## Performance Tips

1. **Start small**: Test with 2 clients first
2. **Skip features initially**: Use `--skip_feature_metrics` for quick tests
3. **Use GPU**: Feature metrics are much faster on GPU
4. **Subsample**: Reduce `--max_samples_features` for speed
5. **Batch multiple runs**: Test different α values in one script
