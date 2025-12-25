"""Distribution heterogeneity evaluation metrics for federated learning.

This package provides comprehensive metrics for quantifying data heterogeneity
across federated learning clients. It includes both static distribution metrics
(computed on data splits before training) and dynamic gradient-based metrics
(computed during FL training).

Metric Categories
-----------------

**Label Distribution Metrics** (Static)
    Measure heterogeneity in class label distributions across clients:

    - **Jensen-Shannon Divergence (JSD)**: Symmetric measure of distribution
      divergence (pairwise across clients)
    - **Total Variation (TV)**: L1 distance between label distributions
    - **JSD Client-to-Global**: Divergence from each client to global distribution

**Feature Distribution Metrics** (Static)
    Measure heterogeneity in semantic feature distributions using embeddings
    from foundation models (InceptionV3, DINOv2, CLIP, SigLIP, MAE):

    - **Fréchet Inception Distance (FID)**: Standard metric for distribution
      similarity, uses InceptionV3 features
    - **Maximum Mean Discrepancy (MMD)**: Non-parametric two-sample test using
      RBF kernel
    - **Energy Distance**: Metric based on pairwise Euclidean distances
    - **Sliced Wasserstein Distance**: Efficient approximation via random
      projections
    - **Kolmogorov-Smirnov (KS) Test**: Multivariate two-sample test with
      optional PCA preprocessing
    - **Vendi Score**: Diversity measure based on matrix determinants
    - **Intrinsic Dimensionality**: Estimate manifold dimension (TWO-NN or MLE)
    - **k-NN Separability**: Fraction of neighbors from same client
    - **Centroid Distance**: Euclidean distance between feature centroids
    - **Feature Variance**: Variance of features within each client

**Gradient-Based Metrics** (Dynamic)
    Measure heterogeneity in client gradient updates during FL training:

    - **Cosine Diversity**: 1 - mean pairwise cosine similarity of gradients
    - **Gradient Variance**: Variance of gradients across clients
    - **Gradient Conflict**: Fraction of client pairs with opposing gradients
    - **L2 Diversity**: Mean pairwise L2 distance between gradients
    - **Spectral Diversity**: Condition number of gradient matrix

Key Features
------------

- **Flexible Feature Extraction**: All feature metrics support multiple
  foundation models via `ModelRegistry` (InceptionV3, DINOv2, CLIP, SigLIP, MAE)
- **Embedding Caching**: Persistent caching of extracted features for efficiency
- **Pairwise Computation**: Most metrics compute all pairwise comparisons and
  return mean or full list
- **Normalization Strategies**: Gradient metrics support multiple normalization
  methods (L2, layer-wise, standardization)
- **GPU Acceleration**: FAISS support for intrinsic dimensionality estimation

Usage Examples
--------------

Label distribution metrics:

    >>> from evaluation.metrics import JS_divergence, TV_label_distributions
    >>>
    >>> # Client datasets with labels
    >>> client_data = [
    ...     [{"image": "img1.jpg", "label": 0}, {"image": "img2.jpg", "label": 1}],
    ...     [{"image": "img3.jpg", "label": 0}, {"image": "img4.jpg", "label": 0}],
    ...     [{"image": "img5.jpg", "label": 1}, {"image": "img6.jpg", "label": 1}],
    ... ]
    >>>
    >>> # Compute label heterogeneity
    >>> jsd = JS_divergence(client_data, num_classes=2)
    >>> tv = TV_label_distributions(client_data, num_classes=2)
    >>> print(f"JSD: {jsd:.4f}, TV: {tv:.4f}")

Feature distribution metrics:

    >>> from evaluation.metrics import FID_distance, MMD, vendi_score
    >>>
    >>> # FID using InceptionV3 features (standard)
    >>> fid = FID_distance(
    ...     client_data,
    ...     use_cache=True,
    ...     cache_dir=".cache"
    ... )
    >>>
    >>> # MMD using DINOv2 features
    >>> mmd = MMD(
    ...     client_data,
    ...     model_name="dinov2-base",
    ...     use_cache=True
    ... )
    >>>
    >>> # Vendi score (diversity within each client)
    >>> vendi = vendi_score(
    ...     client_data,
    ...     model_name="clip-vit-b-32",
    ...     return_all=True  # Per-client scores
    ... )

Gradient diversity metrics:

    >>> from evaluation.gradient_metrics import GradientDiversityMetric
    >>>
    >>> # Initialize metric
    >>> metric = GradientDiversityMetric(
    ...     metric_type="cosine",
    ...     normalization="layer_wise",
    ...     aggregation="weighted_mean"
    ... )
    >>>
    >>> # Compute at each FL round
    >>> client_gradients = [
    ...     {"layer1.weight": torch.randn(10, 5), "layer2.weight": torch.randn(5, 3)},
    ...     {"layer1.weight": torch.randn(10, 5), "layer2.weight": torch.randn(5, 3)},
    ...     {"layer1.weight": torch.randn(10, 5), "layer2.weight": torch.randn(5, 3)},
    ... ]
    >>> diversity = metric.compute(client_gradients)
    >>> print(f"Gradient diversity: {diversity:.4f}")

Advanced usage with PCA and caching:

    >>> from evaluation.metrics import KS_multivariate_test, intrinsic_dimensionality
    >>>
    >>> # KS test with PCA preprocessing
    >>> ks_result = KS_multivariate_test(
    ...     client_data,
    ...     model_name="siglip-base-patch16-384",
    ...     pca_variance=0.95,  # Keep 95% variance
    ...     alpha=0.05,
    ...     use_cache=True
    ... )
    >>> print(f"Reject null: {ks_result['reject_null']}")
    >>>
    >>> # Intrinsic dimensionality with GPU acceleration
    >>> id_estimates = intrinsic_dimensionality(
    ...     client_data,
    ...     model_name="mae-large",
    ...     method="two_nn",  # or "mle"
    ...     use_faiss=True,  # GPU acceleration
    ...     return_all=True
    ... )

Notes
-----

- Feature metrics require images to exist at the specified paths
- Caching significantly speeds up repeated evaluations (embeddings persist)
- Gradient metrics require PyTorch gradient dictionaries from FL training
- Some metrics (FID) are model-specific (InceptionV3); others support any model
- Intrinsic dimensionality estimation benefits from FAISS GPU acceleration

See Also
--------

- embedding.embedding_extractors: Foundation models for feature extraction
- embedding.cache_manager: Persistent embedding caching system
- partitioning: Data splitting strategies for federated learning
"""
