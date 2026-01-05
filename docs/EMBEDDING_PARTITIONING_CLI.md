# Embedding-Based Data Partitioning CLI Guide

This guide explains how to use the `generate_embedding_partitions.py` script to create federated learning data partitions with fully configurable components.

## Quick Start

### Basic Usage with CLI Arguments

```bash
python scripts/generate_embedding_partitions.py \
  --dataset tiny_imagenet \
  --base_path /path/to/tiny-imagenet-200 \
  --extractor dinov2-base \
  --pooling cls \
  --clustering hdbscan \
  --n_clients 10 \
  --output splits.json
```

### Using a Preset Configuration

```bash
python scripts/generate_embedding_partitions.py \
  --preset high_quality \
  --dataset tiny_imagenet \
  --base_path /path/to/tiny-imagenet-200 \
  --n_clients 20 \
  --output splits.json
```

### Using a JSON Config File

```bash
python scripts/generate_embedding_partitions.py \
  --config configs/example_clip_gem_kmeans.json \
  --dataset tiny_imagenet \
  --base_path /path/to/tiny-imagenet-200 \
  --output splits.json
```

---

## Configuration Methods

The script supports three ways to configure the embedding pipeline:

### 1. Individual CLI Arguments

Specify each component separately. This gives you full control over every parameter.

**Required arguments:**
- `--dataset`: Dataset name
- `--base_path`: Path to dataset directory
- `--extractor`: Embedding extractor model
- `--pooling`: Pooling strategy
- `--clustering`: Clustering method

**Example:**
```bash
python scripts/generate_embedding_partitions.py \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --extractor clip-base-patch16 \
  --pooling gem \
  --pooling_p 3.0 \
  --clustering kmeans \
  --n_clusters 20 \
  --n_clients 10 \
  --heterogeneity 0.7 \
  --batch_size 64 \
  --output my_splits.json
```

### 2. Preset Configurations

Use one of four built-in presets and optionally override parameters.

**Available presets:**
- `default`: DINOv2-base + CLS + HDBSCAN (balanced quality/speed)
- `fast`: DINOv2-small + CLS + KMeans (fastest)
- `high_quality`: DINOv2-large + GeM + HDBSCAN (best quality)
- `spectral`: DINOv2-base + Spectral pooling + Spectral clustering (research)

**Example:**
```bash
# Use preset with default settings
python scripts/generate_embedding_partitions.py \
  --preset high_quality \
  --dataset cifar10 \
  --base_path /data/cifar10 \
  --n_clients 50 \
  --output splits.json

# Use preset but override some settings
python scripts/generate_embedding_partitions.py \
  --preset default \
  --batch_size 128 \
  --n_clients 20 \
  --heterogeneity 0.3 \
  --dataset food101 \
  --base_path /data/food101 \
  --output splits.json
```

### 3. JSON Configuration Files

Create reusable configuration files and load them.

**Example config file** (`my_config.json`):
```json
{
  "model": {
    "model_type": "siglip",
    "variant": "base-patch16-384",
    "batch_size": 32
  },
  "pooling": {
    "strategy": "mean"
  },
  "clustering": {
    "method": "hdbscan",
    "params": {"min_cluster_size": 20}
  },
  "n_clients": 15
}
```

**Usage:**
```bash
# Use config file
python scripts/generate_embedding_partitions.py \
  --config my_config.json \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --output splits.json

# Use config file and override parameters
python scripts/generate_embedding_partitions.py \
  --config my_config.json \
  --n_clients 20 \
  --batch_size 64 \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --output splits.json
```

**Save final configuration:**
```bash
python scripts/generate_embedding_partitions.py \
  --config my_config.json \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --n_clients 20 \
  --save_config my_config_final.json \
  --output splits.json
```

---

## Component Selection

### Embedding Extractors (`--extractor`)

**Pre-trained Vision Models:**

| Model | Variants | Example | Output Dim | Speed |
|-------|----------|---------|------------|-------|
| DINOv2 | small, base, large, giant | `dinov2-base` | 768 | Fast-Medium |
| CLIP | base-patch32, base-patch16, large-patch14 | `clip-base-patch16` | 768 | Fast |
| SigLIP | base-patch16-224, base-patch16-384, large-patch16-256 | `siglip-base-patch16-384` | 768 | Fast |
| MAE | base, large, huge | `mae-base` | 768 | Fast |
| InceptionV3 | (single) | `inception_v3` | 2048 | Fast |

**Untrained/Baseline Models:**

| Model | Format | Example | Use Case |
|-------|--------|---------|----------|
| Raw Pixel | `raw_pixel_resizer-<size>` | `raw_pixel_resizer-64` | Baseline |
| Random Projection | `sparse_random_projection_resizer-<size>-<dim>` | `sparse_random_projection_resizer-64-1000` | Baseline |
| Untrained Encoder | `UAE-default` | `UAE-default` | Baseline |
| Wavelet | `wavelet_scattering-<size>-<J>` | `wavelet_scattering-224-3` | Research |

**Examples:**
```bash
# DINOv2 base (recommended default)
--extractor dinov2-base

# CLIP for vision-language features
--extractor clip-large-patch14

# SigLIP with higher resolution
--extractor siglip-base-patch16-384

# InceptionV3 (compatible with FID metrics)
--extractor inception_v3
```

### Pooling Strategies (`--pooling`)

| Strategy | Description | Parameters | Use Case |
|----------|-------------|------------|----------|
| `cls` | CLS token (first token) | None | Default for transformers |
| `mean` | Average all tokens | None | Simple and effective |
| `max` | Max over tokens | None | Emphasize salient features |
| `gem` | Generalized mean | `--pooling_p` (default: 3.0) | Robust aggregation |
| `lastk` | Last K layers | `--pooling_k`, `--pooling_agg` | Multi-layer features |
| `spectral` | Spectral sharpening | `--spectral_k` (default: 50) | Graph-based |

**Examples:**
```bash
# CLS token (standard)
--pooling cls

# GeM pooling with p=4.0
--pooling gem --pooling_p 4.0

# Mean pooling (simple averaging)
--pooling mean

# Last 4 layers with weighted aggregation
--pooling lastk --pooling_k 4 --pooling_agg weighted

# Spectral sharpening with 100 eigenvectors
--pooling spectral --spectral_k 100
```

### Clustering Methods (`--clustering`)

| Method | Description | Parameters | Automatic K | Use Case |
|--------|-------------|------------|-------------|----------|
| `hdbscan` | Density-based hierarchical | `--min_cluster_size`, `--min_samples` | Yes | Automatic, flexible |
| `kmeans` | Centroid-based | `--n_clusters` (required) | No | Fast, fixed K |
| `knn` | K-NN graph components | `--knn_neighbors` | Yes | Graph-based |
| `spectral` | Laplacian eigenvectors | `--n_clusters`, `--spectral_affinity` | No | Complex distributions |

**Examples:**
```bash
# HDBSCAN with automatic cluster detection
--clustering hdbscan --min_cluster_size 20

# K-means with 30 clusters
--clustering kmeans --n_clusters 30

# K-NN graph with 20 neighbors
--clustering knn --knn_neighbors 20

# Spectral clustering with RBF affinity
--clustering spectral --n_clusters 15 --spectral_affinity rbf
```

---

## Common Use Cases

### 1. Quick Experimentation (Fast Preset)

```bash
python scripts/generate_embedding_partitions.py \
  --preset fast \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --n_clients 10 \
  --output quick_test.json
```

**Configuration:**
- Extractor: DINOv2-small (22M params)
- Pooling: CLS token
- Clustering: K-means
- Speed: ~2-3 minutes for Tiny ImageNet

### 2. Production Quality (High Quality Preset)

```bash
python scripts/generate_embedding_partitions.py \
  --preset high_quality \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --n_clients 50 \
  --heterogeneity 0.8 \
  --output production_splits.json \
  --save_metadata metadata.json \
  --save_embeddings embeddings.npz
```

**Configuration:**
- Extractor: DINOv2-large (300M params)
- Pooling: GeM (p=3.0)
- Clustering: HDBSCAN (automatic)
- Speed: ~10-15 minutes for Tiny ImageNet

### 3. Vision-Language Features (CLIP)

```bash
python scripts/generate_embedding_partitions.py \
  --extractor clip-large-patch14 \
  --pooling cls \
  --clustering hdbscan \
  --min_cluster_size 25 \
  --dataset food101 \
  --base_path /data/food101 \
  --n_clients 20 \
  --output clip_splits.json
```

### 4. Semantic Similarity (SigLIP + Mean)

```bash
python scripts/generate_embedding_partitions.py \
  --extractor siglip-base-patch16-384 \
  --pooling mean \
  --clustering spectral \
  --n_clusters 25 \
  --spectral_affinity rbf \
  --dataset cifar100 \
  --base_path /data/cifar100 \
  --n_clients 10 \
  --output siglip_splits.json
```

### 5. Balanced Client Sizes

```bash
python scripts/generate_embedding_partitions.py \
  --preset default \
  --assignment_strategy size_balanced \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --n_clients 30 \
  --output balanced_splits.json
```

### 6. High Heterogeneity

```bash
python scripts/generate_embedding_partitions.py \
  --extractor dinov2-base \
  --pooling gem \
  --pooling_p 4.0 \
  --clustering hdbscan \
  --min_cluster_size 10 \
  --n_clients 50 \
  --heterogeneity 0.9 \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --output high_hetero_splits.json
```

---

## Advanced Options

### Cache Management

```bash
# Use custom cache directory
--cache_dir /scratch/embedding_cache

# Disable caching (not recommended)
--no_cache
```

### Assignment Strategies

```bash
# Round-robin (default)
--assignment_strategy round_robin

# Random assignment
--assignment_strategy random

# Balance client sizes
--assignment_strategy size_balanced
```

### Noise Handling (HDBSCAN only)

```bash
# Distribute noise samples across clients (default)
--handle_noise distribute

# Assign all noise to one client
--handle_noise separate_client

# Ignore noise samples (not recommended)
--handle_noise ignore
```

### Validation Set Options

```bash
# Global validation (default) - shared across all clients
--valset_type global

# Local validation - split proportionally to clients
--valset_type local

# No validation set
--valset_type none
```

### Device Selection

```bash
# Auto-detect (default)
# (no --device argument)

# Force CPU
--device cpu

# Use specific GPU
--device cuda:0

# Use CUDA (any available GPU)
--device cuda
```

### Save Additional Outputs

```bash
python scripts/generate_embedding_partitions.py \
  --preset high_quality \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --n_clients 10 \
  --output splits.json \
  --save_config final_config.json \
  --save_metadata cluster_metadata.json \
  --save_embeddings embeddings.npz
```

---

## Complete Examples

### Example 1: Research Paper Quality

```bash
python scripts/generate_embedding_partitions.py \
  --extractor dinov2-large \
  --pooling gem \
  --pooling_p 3.5 \
  --clustering hdbscan \
  --min_cluster_size 15 \
  --n_clients 100 \
  --heterogeneity 0.7 \
  --dataset tiny_imagenet \
  --base_path /data/tiny-imagenet-200 \
  --batch_size 16 \
  --assignment_strategy size_balanced \
  --valset_type global \
  --output paper_splits.json \
  --save_config paper_config.json \
  --save_metadata paper_metadata.json \
  --save_embeddings paper_embeddings.npz \
  --seed 42
```

### Example 2: Baseline Comparison

```bash
# Untrained baseline
python scripts/generate_embedding_partitions.py \
  --extractor raw_pixel_resizer-64 \
  --pooling mean \
  --clustering kmeans \
  --n_clusters 20 \
  --n_clients 10 \
  --dataset cifar10 \
  --base_path /data/cifar10 \
  --output baseline_splits.json

# Pre-trained comparison
python scripts/generate_embedding_partitions.py \
  --extractor dinov2-base \
  --pooling cls \
  --clustering kmeans \
  --n_clusters 20 \
  --n_clients 10 \
  --dataset cifar10 \
  --base_path /data/cifar10 \
  --output pretrained_splits.json
```

### Example 3: Multi-Config Experiment

```bash
# Create configs directory with multiple experiments
mkdir -p experiments/configs

# Generate config files with different settings
for method in hdbscan kmeans spectral; do
  for model in dinov2-base clip-base-patch16 siglip-base-patch16-224; do
    python scripts/generate_embedding_partitions.py \
      --extractor $model \
      --pooling cls \
      --clustering $method \
      --n_clients 10 \
      --dataset tiny_imagenet \
      --base_path /data/tiny-imagenet-200 \
      --output experiments/${model}_${method}_splits.json \
      --save_config experiments/configs/${model}_${method}_config.json
  done
done
```

---

## Troubleshooting

### Common Issues

**Issue: Out of memory during embedding extraction**
```bash
# Solution: Reduce batch size
--batch_size 8
```

**Issue: HDBSCAN produces too many/few clusters**
```bash
# Too many clusters: Increase min_cluster_size
--min_cluster_size 30

# Too few clusters: Decrease min_cluster_size
--min_cluster_size 10
```

**Issue: K-means requires n_clusters**
```bash
# Solution: Specify number of clusters
--clustering kmeans --n_clusters 20
```

**Issue: Model download fails**
```bash
# Solution: Set proxy
--proxy http://proxy.example.com:3128
```

**Issue: Unbalanced client sizes**
```bash
# Solution: Use size-balanced assignment
--assignment_strategy size_balanced
```

---

## Getting Help

View full help documentation:
```bash
python scripts/generate_embedding_partitions.py --help
```

List all available extractors:
```bash
python -c "from embedding.embedding_extractors import ModelRegistry; print(ModelRegistry.list_models())"
```

---

## See Also

- **Configuration files:** `configs/README.md`
- **Component documentation:** `docs/README_embeddings.md`
- **Example configs:** `configs/example_*.json`
