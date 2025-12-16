# Embedding-Based Data Partitioning Pipeline

This module provides a sophisticated data partitioning system for federated learning based on semantic clustering of foundation model embeddings.

## Overview

The embedding pipeline creates data partitions by:
1. **Extracting embeddings** from images using frontier foundation models (DINOv2, CLIP, SigLIP)
2. **Applying pooling strategies** to aggregate token-level embeddings
3. **Clustering** embeddings into semantically meaningful groups
4. **Assigning clusters** to clients for heterogeneous data distributions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EmbeddingSplitter                        │
│                 (inherits BaseDatasetSplitter)              │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │Embedding │    │ Pooling  │    │Clustering│
   │Extractor │───▶│ Strategy │───▶│  Method  │
   └──────────┘    └──────────┘    └──────────┘
          │               │               │
          ▼               ▼               ▼
   Foundation       Token-to-       Semantic
   Models (FM)      Sample         Grouping
                   Aggregation
```

## Components

### 1. Embedding Extractors (`embedding_extractors.py`)

Foundation model wrappers for extracting visual embeddings.

**Available Models:**

- **DINOv2** (Meta AI)
  - Variants: `small` (384d), `base` (768d), `large` (1024d), `giant` (1536d)
  - Self-supervised vision transformer
  - Excellent for semantic similarity

- **CLIP** (OpenAI)
  - Variants: `base-patch32`, `base-patch16`, `large-patch14`
  - Vision-language model
  - Good for text-aligned features

- **SigLIP** (Google)
  - Variants: `base-patch16-224/384`, `large-patch16-256`
  - Improved CLIP with better performance
  - Efficient sigmoid loss

**Usage:**
```python
from embedding_pipeline import DINOv2Extractor

extractor = DINOv2Extractor(variant="base", batch_size=32)
embeddings = extractor.extract_embeddings(
    images=["path/to/img1.jpg", "path/to/img2.jpg"],
    return_all_tokens=False  # False = CLS token only
)
# Shape: (n_images, embed_dim)
```

### 2. Pooling Strategies (`pooling_strategies.py`)

Methods for aggregating token-level embeddings into sample-level representations.

**Available Strategies:**

- **CLS Pooling** - Extract first token (CLS token)
  ```python
  CLSPooling()  # Simple and effective
  ```

- **Mean Pooling** - Average all tokens
  ```python
  MeanPooling()  # Robust to outliers
  ```

- **Max Pooling** - Element-wise maximum
  ```python
  MaxPooling()  # Captures salient features
  ```

- **GeM Pooling** - Generalized mean pooling
  ```python
  GeMLPooling(p=3.0)  # p=1: mean, p→∞: max
  ```

- **Last-K Layer Pooling** - Aggregate multiple layers
  ```python
  LastKLayerPooling(k=4, aggregation="weighted")
  # aggregation: "mean", "concat", or "weighted"
  ```

- **Spectral Sharpening** - Project onto Laplacian eigenvectors
  ```python
  SpectralSharpeningPooling(k=50, sigma=None)
  # Uses graph signal processing to emphasize structure
  ```

### 3. Clustering Methods (`clustering_methods.py`)

Algorithms for grouping embeddings into clusters.

**Available Methods:**

- **HDBSCAN** - Hierarchical density-based clustering
  ```python
  HDBSCANMethod(
      min_cluster_size=15,
      metric="euclidean",
      cluster_selection_method="eom"
  )
  # Pros: Automatic cluster count, handles noise
  # Cons: Can be slow for large datasets
  ```

- **K-Means** - Centroid-based clustering
  ```python
  KMeansMethod(n_clusters=10)
  # Pros: Fast, deterministic
  # Cons: Requires specifying K
  ```

- **K-NN Graph** - Connected components in k-NN graph
  ```python
  KNNGraphMethod(n_neighbors=15, metric="cosine")
  # Pros: Captures local structure
  # Cons: Sensitive to k parameter
  ```

- **Spectral Clustering** - Graph Laplacian eigenvectors + k-means
  ```python
  SpectralMethod(
      n_clusters=10,
      affinity="rbf",
      gamma=1.0
  )
  # Pros: Handles non-convex clusters
  # Cons: Requires specifying K
  ```

### 4. Cache Manager (`cache_manager.py`)

Efficient caching of computed embeddings to avoid redundant extraction.

**Features:**
- Content-based hashing for cache keys
- Automatic cache invalidation
- Metadata tracking
- Compressed storage (npz format)

**Usage:**
```python
from embedding_pipeline import CacheManager

cache = CacheManager(cache_dir="./.embedding_cache", enabled=True)

# Cache is automatically used by EmbeddingSplitter
# Or use manually:
embeddings = cache.get(
    model_name="dinov2/base",
    image_paths=image_list,
    pooling_strategy="cls"
)

if embeddings is None:
    # Extract and cache
    embeddings = extract_fresh()
    cache.put(embeddings, model_name="dinov2/base", ...)
```

### 5. Configuration System (`config.py`)

Dataclass-based configuration for the entire pipeline.

**Usage:**
```python
from embedding_pipeline import (
    EmbeddingConfig,
    ModelConfig,
    PoolingConfig,
    ClusteringConfig,
)

# Create custom configuration
config = EmbeddingConfig(
    model=ModelConfig(
        model_type="dinov2",
        variant="base",
        batch_size=32,
    ),
    pooling=PoolingConfig(
        strategy="gem",
        params={"p": 3.0}
    ),
    clustering=ClusteringConfig(
        method="hdbscan",
        params={"min_cluster_size": 15}
    ),
    n_clients=10,
    valset_type="global",
)

# Save/load configuration
config.to_json("my_config.json")
loaded = EmbeddingConfig.from_json("my_config.json")

# Or use presets
from embedding_pipeline import get_preset_config
config = get_preset_config("default", n_clients=10)
# Presets: "default", "fast", "high_quality", "spectral"
```

### 6. Embedding Splitter (`embedding_splitter.py`)

Main class that orchestrates the entire pipeline.

**Usage:**
```python
from embedding_pipeline import EmbeddingSplitter, get_preset_config
from dataset import BaseDataset

# Load dataset
train_dataset = BaseDataset(
    root="./data/cifar10/train",
    label_map_path="./data/cifar10/label_map.json"
)

val_dataset = BaseDataset(
    root="./data/cifar10/val",
    label_map_path="./data/cifar10/label_map.json"
)

# Create configuration
config = get_preset_config("default", n_clients=10)

# Create splitter
splitter = EmbeddingSplitter(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    n_clients=10,
    heterogeneity=0.5,  # 0.0 = IID, 1.0 = highly non-IID
    valset_type="global",  # "local", "global", or "none"
    config=config,
    seed=42
)

# Generate splits
splits = splitter.get_client_splits()

# Save results
splitter.store_client_splits("embedding_split.json")
splitter.save_cluster_metadata("embedding_metadata.json")
splitter.save_embeddings("embeddings.npz")

# Validate splits
splitter.validate_splits()

# Get summary
summary = splitter.get_split_summary()
```

## Preset Configurations

We provide four preset configurations for common use cases:

| Preset | Model | Pooling | Clustering | Use Case |
|--------|-------|---------|------------|----------|
| `default` | DINOv2-base | CLS | HDBSCAN | Balanced quality/speed |
| `fast` | DINOv2-small | CLS | K-Means | Quick experiments |
| `high_quality` | DINOv2-large | GeM | HDBSCAN | Best semantic clustering |
| `spectral` | DINOv2-base | Spectral | Spectral | Graph-based partitioning |

## Output Format

The splitter produces `DatasetSplits` compatible with `BaseDatasetSplitter`:

```json
{
  "clients": {
    "client_0": {
      "train": [
        {"image": "path/to/img1.jpg", "label": "cat"},
        {"image": "path/to/img2.jpg", "label": "dog"}
      ],
      "val": [...]
    },
    "client_1": {...}
  },
  "global_val": [...]
}
```

## Cluster Assignment Strategies

After clustering, clusters are assigned to clients using one of these strategies:

- **`round_robin`** (default) - Distribute clusters sequentially
- **`random`** - Random cluster assignment
- **`size_balanced`** - Balance client dataset sizes

Noise handling (for HDBSCAN):
- **`distribute`** - Spread noise samples across clients
- **`separate_client`** - Assign all noise to one client
- **`ignore`** - Drop noise samples (not recommended)

## Heterogeneity Control

The `heterogeneity` parameter (0.0-1.0) controls data heterogeneity:

- **0.0** → Larger clusters (more IID-like)
- **0.5** → Moderate cluster sizes
- **1.0** → Smaller clusters (highly non-IID)

For HDBSCAN, this adjusts `min_cluster_size`:
```python
min_size = max(5, base_size * (1.0 - heterogeneity))
```

## Advanced Features

### Multi-Layer Extraction

Extract embeddings from multiple layers:

```python
extractor = DINOv2Extractor(variant="base")
layer_embeddings = extractor.get_layer_embeddings(
    images=image_paths,
    layers=[-4, -3, -2, -1]  # Last 4 layers
)
# Returns: {-4: array, -3: array, -2: array, -1: array}

# Then use LastKLayerPooling
pooling = LastKLayerPooling(k=4, aggregation="weighted")
pooled = pooling.pool(layer_embeddings)
```

### Spectral Sharpening Details

The spectral sharpening strategy:
1. Computes pairwise RBF kernel similarities
2. Builds graph Laplacian matrix
3. Extracts top-K eigenvectors
4. Projects embeddings onto eigenvector space

This emphasizes dominant structures in the data manifold.

## Performance Considerations

### GPU Usage
All models automatically use GPU if available:
```python
config = EmbeddingConfig(
    model=ModelConfig(
        model_type="dinov2",
        device="cuda",  # or "cpu", or None for auto
        batch_size=64   # Increase for GPU
    ),
    ...
)
```

### Caching
Embeddings are cached by default. To clear cache:
```python
cache = CacheManager()
cache.clear()  # Clear all
cache.clear(model_name="dinov2/base")  # Clear specific model
```

### Memory Usage

| Model | Variant | Memory (approx) |
|-------|---------|-----------------|
| DINOv2 | small | ~350 MB |
| DINOv2 | base | ~700 MB |
| DINOv2 | large | ~1.5 GB |
| DINOv2 | giant | ~2.5 GB |
| CLIP | base | ~600 MB |
| SigLIP | base | ~600 MB |

## Examples

See `example_usage.py` for complete examples:
- Basic usage with presets
- Custom configuration
- Spectral sharpening
- Comparing different configurations
- Loading/saving configurations

## Integration with Existing Code

The `EmbeddingSplitter` is a drop-in replacement for other splitters:

```python
# Before: RandomSplitter
from data_partitioning import RandomSplitter
splitter = RandomSplitter(train_ds, val_ds, n_clients, ...)

# After: EmbeddingSplitter
from embedding_pipeline import EmbeddingSplitter, get_preset_config
config = get_preset_config("default", n_clients)
splitter = EmbeddingSplitter(train_ds, val_ds, n_clients, ..., config=config)

# Same interface!
splits = splitter.get_client_splits()
splitter.store_client_splits("splits.json")
```

## References

- **DINOv2**: [Oquab et al., 2023](https://arxiv.org/abs/2304.07193)
- **CLIP**: [Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- **SigLIP**: [Zhai et al., 2023](https://arxiv.org/abs/2303.15343)
- **HDBSCAN**: [McInnes et al., 2017](https://joss.theoj.org/papers/10.21105/joss.00205)
- **Spectral Clustering**: [von Luxburg, 2007](https://arxiv.org/abs/0711.0189)

## Troubleshooting

### Model Download Issues
If behind a proxy:
```python
config = EmbeddingConfig(
    model=ModelConfig(
        model_type="dinov2",
        proxy="http://your-proxy:3128"
    ),
    ...
)
```

### Out of Memory
Reduce batch size:
```python
config.model.batch_size = 8  # Default is 32
```

### Too Many/Few Clusters
Adjust clustering parameters:
```python
# HDBSCAN: Increase for fewer clusters
config.clustering.params["min_cluster_size"] = 30

# K-Means: Set directly
config.clustering.params["n_clusters"] = 20
```
