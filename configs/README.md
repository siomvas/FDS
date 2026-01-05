# Configuration Files for Embedding-Based Partitioning

This directory contains JSON configuration files for the embedding-based data partitioning pipeline.

## Usage

Use these config files with the `generate_embedding_partitions.py` script:

```bash
python scripts/generate_embedding_partitions.py \
  --config configs/example_clip_gem_kmeans.json \
  --dataset tiny_imagenet \
  --base_path /path/to/data \
  --output splits.json
```

## Available Configurations

### `embedding_partition_template.json`
Basic template configuration with common defaults:
- **Extractor:** DINOv2 base
- **Pooling:** CLS token
- **Clustering:** HDBSCAN (automatic cluster detection)
- **Use case:** General-purpose starting point

### `example_clip_gem_kmeans.json`
CLIP with GeM pooling and K-means clustering:
- **Extractor:** CLIP base-patch16
- **Pooling:** GeM (p=3.0)
- **Clustering:** K-means (20 clusters)
- **Use case:** Fixed number of clusters, vision-language features

### `example_high_quality.json`
High-quality configuration for research/production:
- **Extractor:** DINOv2 large (300M params)
- **Pooling:** GeM (p=3.0)
- **Clustering:** HDBSCAN (automatic, min_size=10)
- **Use case:** Best quality partitions, sufficient compute

### `example_siglip_spectral.json`
Spectral clustering with SigLIP features:
- **Extractor:** SigLIP base-patch16-384
- **Pooling:** Spectral sharpening (k=50)
- **Clustering:** Spectral clustering (15 clusters)
- **Use case:** Complex data distributions, research experiments

## Configuration Schema

```json
{
  "model": {
    "model_type": "dinov2|clip|siglip|mae|inception_v3|...",
    "variant": "base|small|large|...",
    "device": null,  // Auto-detect or specify "cuda", "cpu", "cuda:0"
    "batch_size": 32,
    "num_workers": 4,
    "proxy": null  // HTTP proxy for model downloads
  },
  "pooling": {
    "strategy": "cls|mean|max|gem|lastk|spectral",
    "params": {
      // Strategy-specific parameters
      // For GeM: {"p": 3.0}
      // For LastK: {"k": 4, "aggregation": "mean"}
      // For Spectral: {"k": 50}
    }
  },
  "clustering": {
    "method": "hdbscan|kmeans|knn|spectral",
    "params": {
      // Method-specific parameters
      // For HDBSCAN: {"min_cluster_size": 15, "min_samples": 15}
      // For KMeans: {"n_clusters": 10}
      // For KNN: {"n_neighbors": 15}
      // For Spectral: {"n_clusters": 10, "affinity": "rbf"}
    }
  },
  "n_clients": 10,
  "cache_dir": "./.embedding_cache",
  "cache_enabled": true,
  "assignment_strategy": "round_robin|random|size_balanced",
  "handle_noise": "distribute|ignore|separate_client"
}
```

## Component Options

### Embedding Extractors

**Pre-trained Vision Models:**
- `dinov2` variants: `small`, `base`, `large`, `giant`
- `clip` variants: `base-patch32`, `base-patch16`, `large-patch14`
- `siglip` variants: `base-patch16-224`, `base-patch16-384`, `large-patch16-256`
- `mae` variants: `base`, `large`, `huge`
- `inception_v3` (single variant)

**Untrained/Baseline Models:**
- `raw_pixel_resizer` (specify size: `64`, `128`)
- `sparse_random_projection_resizer` (format: `size-dim`, e.g., `64-1000`)
- `UAE` (Untrained AutoEncoder)
- `wavelet_scattering` (format: `size-J`, e.g., `224-3`)

### Pooling Strategies

- **`cls`**: Extract CLS token (first token) - standard for transformers
- **`mean`**: Average all token embeddings - simple and effective
- **`max`**: Max pooling over tokens - emphasizes salient features
- **`gem`**: Generalized mean pooling - robust aggregation (param: `p`)
- **`lastk`**: Aggregate last K layers - multi-layer features (params: `k`, `aggregation`)
- **`spectral`**: Spectral sharpening via Laplacian - graph-based (param: `k`)

### Clustering Methods

- **`hdbscan`**: Hierarchical density-based - automatic cluster detection
  - Params: `min_cluster_size`, `min_samples`
  - Can produce noise points (label -1)

- **`kmeans`**: Centroid-based - fast and simple
  - Params: `n_clusters` (required)
  - Fixed number of clusters

- **`knn`**: K-nearest neighbors graph - connected components
  - Params: `n_neighbors`, `metric`
  - Automatic cluster count

- **`spectral`**: Graph Laplacian eigenvectors + k-means
  - Params: `n_clusters` (required), `affinity`, `gamma`, `n_neighbors`
  - Good for complex distributions

## Creating Custom Configurations

1. Copy `embedding_partition_template.json`
2. Modify the components you want to change
3. Save with a descriptive name
4. Test with your dataset

Example workflow:
```bash
# Copy template
cp configs/embedding_partition_template.json configs/my_config.json

# Edit the file
# vim configs/my_config.json

# Test configuration
python scripts/generate_embedding_partitions.py \
  --config configs/my_config.json \
  --dataset tiny_imagenet \
  --base_path /path/to/data \
  --n_clients 20 \
  --output test_splits.json

# Save the actual configuration used
python scripts/generate_embedding_partitions.py \
  --config configs/my_config.json \
  --dataset tiny_imagenet \
  --base_path /path/to/data \
  --n_clients 20 \
  --output splits.json \
  --save_config configs/my_config_final.json
```

## CLI Overrides

You can override any config file parameter via CLI:

```bash
python scripts/generate_embedding_partitions.py \
  --config configs/embedding_partition_template.json \
  --extractor dinov2-large \
  --batch_size 64 \
  --n_clients 20 \
  --dataset tiny_imagenet \
  --base_path /path/to/data
```

CLI arguments take precedence over config file values.

## Preset Configurations

Instead of JSON files, you can also use built-in presets:

```bash
python scripts/generate_embedding_partitions.py \
  --preset high_quality \
  --dataset tiny_imagenet \
  --base_path /path/to/data \
  --n_clients 20
```

Available presets: `default`, `fast`, `high_quality`, `spectral`

Presets can also be customized with CLI arguments.
