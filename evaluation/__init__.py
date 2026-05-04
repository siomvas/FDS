"""Distribution heterogeneity metrics for federated datasets.

The evaluation package quantifies how different client partitions are before or
after split generation. The public package focuses on static label and feature
distribution metrics used by the reviewer reproduction workflow.

Metric categories
-----------------

**Label distribution metrics**
    Jensen-Shannon divergence, total variation, and client-to-global label
    divergence.

**Feature distribution metrics**
    FID, MMD, energy distance, sliced Wasserstein distance, multivariate KS
    testing, Vendi score, intrinsic dimensionality, k-NN separability, centroid
    distance, and feature variance.

Feature metrics use embeddings from the ``embedding`` package and can share a
persistent ``CacheManager`` across repeated evaluations.
"""
