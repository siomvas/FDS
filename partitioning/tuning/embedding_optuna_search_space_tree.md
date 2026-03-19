# Embedding Optuna Search Space Tree

This file describes the search flow currently implemented in
`partitioning/tuning/embedding_optuna.py`, based on the behavior of:

- `embedding/embedding_extractors.py`
- `partitioning/embedding_splitter.py`
- `partitioning/embedding_utils/pooling_strategies.py`
- `partitioning/embedding_utils/clustering_methods.py`

## Tree

```text
embedding_optuna trial
└── extractor
    ├── Transformer-backed families
    │   ├── dinov2-*
    │   ├── clip-*
    │   ├── siglip-*
    │   └── mae-*
    │       └── token extraction
    │           ├── cls
    │           │   └── pooling = cls
    │           │       └── output shape: [n_samples, embed_dim]
    │           │
    │           ├── all_last
    │           │   └── pooling
    │           │       ├── mean
    │           │       ├── gem
    │           │       │   └── p = 2.0
    │
    └── Non-transformer / already-sample-level families
        ├── inception_v3
        ├── raw_pixel_resizer-*
        ├── sparse_random_projection_resizer-*
        ├── UAE
        └── wavelet_scattering-*
            └── token extraction = cls
                └── pooling = cls
                    └── output shape: [n_samples, embed_dim]

After embeddings are produced:
└── preprocessing
    ├── normalize
    │   ├── false
    │   └── true
    │       └── L2 normalization
    └── dim reduction
        ├── none
        └── umap
            ├── dr_dims ∈ {32, 128} by default
            ├── umap_neighbors = 15
            ├── umap_min_dist = 0.1
            └── umap_metric = euclidean

Then clustering:
└── study_mode
    ├── fixed
    │   └── clustering
    │       └── kmeans
    │           ├── n_clusters = target_n_clients
    │           └── random_state = seed
    │
    └── discovery
        └── clustering
            └── hdbscan
                ├── min_cluster_size ∈ {1x, 2x, 4x} * min_client_size
                │   └── default with min_client_size=20: {20, 40, 80}
                ├── min_samples ∈ {1, 5, 10} by default
                ├── hdbscan_metric
                │   ├── normalize = false -> {euclidean, cosine}
                │   └── normalize = true -> euclidean
                └── hdbscan_method ∈ {eom, leaf}

Then staged evaluation:
└── stage loop
    ├── build splitter
    ├── generate client splits
    ├── reject trial if:
    │   ├── fewer than 2 clients
    │   ├── any client size < min_client_size
    │   ├── fixed mode client count != target_n_clients
    │   └── discovery mode client count outside [min_discovered_clients, max_discovered_clients]
    ├── evaluate stage MMD
    ├── report to Optuna
    └── prune on intermediate stages if pruner says so

Completed trial:
└── final full-stage evaluation
    ├── balance metrics
    ├── MMD
    ├── FID
    ├── centroid distance
    ├── feature variance
    └── artifact writeout
```

## Current Defaults

```text
Fixed mode defaults
├── extractors
│   ├── dinov2-base
│   ├── clip-base-patch16
│   ├── siglip-base-patch16-224
│   ├── mae-base
│   ├── inception_v3
│   ├── raw_pixel_resizer-64
│   ├── sparse_random_projection_resizer
│   ├── UAE
│   └── wavelet_scattering
├── token choices
│   ├── cls
│   ├── all_last
│   ├── k_last_cls
│   └── k_last_all
└── pooling choices
    ├── mean
    ├── max
    └── gem (p = 2.0)
```

```text
Discovery mode defaults
├── extractors
│   ├── dinov2-base
│   ├── clip-base-patch16
│   └── siglip-base-patch16-224
├── token choices
│   ├── cls
│   └── all_last
└── pooling choices
    ├── mean
    └── gem (p = 2.0)
```

The default fixed-mode study remains broad. The narrowed extractor/token/pooling
selection is only the default for discovery/HDBSCAN runs.

This fixed-mode default set covers the distinct registered model families used by
the study:

```text
dinov2
clip
siglip
mae
inception_v3
raw_pixel_resizer
sparse_random_projection_resizer
UAE
wavelet_scattering
```

## Important Constraints

```text
1. Pooling is only meaningfully searched for transformer-backed extractors.
2. Non-transformer extractors already return [n_samples, embed_dim], so token/pooling
   branches are forced off.
3. token_extraction = cls always forces pooling = cls.
4. k_last_* token branches are active in the default fixed-mode study, but excluded
   from the default discovery/HDBSCAN study.
5. Spectral pooling is deprecated and is not part of the study search space.
6. Spectral clustering is not part of the study search space.
```

## Not Currently Active As Search Axes

These exist in CLI/config surface area, but are not real Optuna branches in the
current refined study:

```text
lastk_pooling_choices
└── retained in CLI surface, but not sampled by embedding_optuna

layer_aggregation_choices
└── used by fixed-mode k_last_* branches as a fixed placeholder metadata choice
    (first allowed value; default: mean)

lastk_choices
└── sampled by fixed-mode k_last_* branches
```

If you want, the next step can be to make this file and the code converge even
further by either:

1. Removing inactive knobs from the study CLI, or
2. Refactoring the extractor/splitter path so true last-k layer aggregation becomes
   a real searchable branch again.
