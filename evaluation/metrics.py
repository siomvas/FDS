"""Evaluation metrics for measuring data heterogeneity in federated learning.

This module provides comprehensive metrics to quantify heterogeneity between
client data distributions in federated learning scenarios. Metrics are divided
into label-based (statistical) and feature-based (embedding) categories.

Label-Based Metrics
-------------------
JS_divergence
    Jensen-Shannon Divergence between label distributions.
TV_label_distributions
    Total Variation distance between label distributions.
JSD_client_to_global
    Per-client JSD relative to global distribution.

Feature-Based Metrics
---------------------
FID_distance
    Frechet Inception Distance (uses InceptionV3 features).
MMD
    Maximum Mean Discrepancy with RBF kernel.
energy_distance_features
    Energy distance in feature space.
sliced_wasserstein_distance
    Sliced Wasserstein Distance using random projections.
KS_multivariate_test
    Kolmogorov-Smirnov test with optional PCA.
vendi_score
    Vendi Score for intra-client diversity.
intrinsic_dimensionality
    Estimate intrinsic dimension using TWO-NN or MLE.
knn_separability
    k-NN isolation metric for client separability.
centroid_distance
    Euclidean distance between client centroids.
feature_variance
    Feature variance within each client.

Usage
-----
All metrics accept a list of datasets (one per client) and return either
a single aggregated value or a list of pairwise/per-client values:

    >>> from evaluation.metrics import FID_distance, JS_divergence
    >>> client_data = [
    ...     [{"image": "img1.jpg", "label": 0}, ...],  # client 0
    ...     [{"image": "img2.jpg", "label": 1}, ...],  # client 1
    ... ]
    >>> fid = FID_distance(client_data, use_cache=True, dataset_name="tiny_imagenet")
    >>> jsd = JS_divergence(client_data)

Caching
-------
Feature-based metrics support caching via the ``use_cache``, ``dataset_name``,
and ``base_path`` parameters. Always provide these for efficient repeated
computation.

See Also
--------
embedding.cache_manager : Persistent feature caching used by feature metrics.
embedding.embedding_extractors : Registered feature extractors.
"""

import logging
import numpy as np
import torch
import scipy
from scipy import stats
from scipy.spatial.distance import cdist
from PIL import Image
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from embedding.cache_manager import CacheManager

from data_loading.dataset import ImageDataset

_GLOBAL_CACHE: Optional[CacheManager] = None
log = logging.getLogger(__name__)
ID_ESTIMATION_ERRORS = (RuntimeError, ValueError, TypeError, FloatingPointError, OSError)


def _get_model_registry():
    from embedding.embedding_extractors import ModelRegistry

    return ModelRegistry


def _get_pca_class():
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "PCA-backed KS testing requires the optional dependency 'scikit-learn'."
        ) from exc
    return PCA


def _score_vendi(features: np.ndarray, q: float = 1, normalize: bool = True) -> float:
    try:
        from vendi_score import vendi
    except ImportError as exc:
        raise ImportError(
            "Vendi score requires the optional dependency 'vendi-score'."
        ) from exc
    return float(vendi.score_dual(features, q=q, normalize=normalize))


def _default_cache_dir() -> str:
    return os.environ.get("FDS_CACHE_DIR", str(Path.cwd() / "embedding_cache"))


def get_global_cache(cache_dir: Optional[Union[str, os.PathLike]] = None) -> CacheManager:
    """Get or create the global CacheManager instance.

    Uses a singleton pattern to reuse the same cache manager across
    multiple metric computations.

    Parameters
    ----------
    cache_dir : str or path-like, optional
        Directory for cache storage. If omitted, uses ``FDS_CACHE_DIR`` or
        ``./embedding_cache``. If different from current cache,
        creates a new instance.

    Returns
    -------
    CacheManager
        The global cache manager instance.
    """
    cache_dir = str(cache_dir or _default_cache_dir())
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is not None and str(_GLOBAL_CACHE.cache_dir) != str(cache_dir):
        _GLOBAL_CACHE = None
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = CacheManager(cache_dir=cache_dir, enabled=True)
    return _GLOBAL_CACHE


def _extract_features(
    dataset_list: List[List[Dict[str, Any]]],
    model_name: str = "inception_v3",
    device: Optional[str] = None,
    batch_size: int = 200,
    use_cache: bool = True,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    dataset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    proxy: Optional[str] = None,
    **extractor_kwargs: Any,
) -> List[np.ndarray]:
    """Extract features from a list of datasets using any model from ModelRegistry.

    This is the shared feature extraction function used by all feature-based
    metrics. It handles model creation, caching, and batched inference.

    Parameters
    ----------
    dataset_list : List[List[Dict[str, Any]]]
        List of datasets, one per client. Each dataset is a list of dicts
        with "image" (path) and "label" keys.
    model_name : str, optional
        Model identifier for embedding extraction. Supports:

        - "inception_v3": InceptionV3 (2048-dim, for FID)
        - "dinov2-small/base/large/giant": DINOv2 (384-1536-dim)
        - "clip-base-patch16/base-patch32/large-patch14": CLIP (768-1024-dim)
        - "siglip-base-patch16-224/384": SigLIP (768-dim)
        - "mae-base/large/huge": MAE (768-1280-dim)

        Default: "inception_v3".
    device : str, optional
        Device for computation ("cuda", "cpu"). If None, auto-detects.
    batch_size : int, optional
        Batch size for feature extraction. Default: 200.
    use_cache : bool, optional
        Whether to cache extracted features. Default: True.
    cache_dir : str or path-like, optional
        Directory for cache storage. If omitted, uses ``FDS_CACHE_DIR`` or
        ``./embedding_cache``.
    dataset_name : str, optional
        Dataset identifier for cache lookup. Required for effective caching.
    base_path : str, optional
        Base path for relative path computation in cache keys.
    proxy : str, optional
        HTTP proxy URL for model downloads (e.g., "http://proxy:3128").
    **extractor_kwargs
        Additional arguments passed to the extractor (e.g., return_which_tokens).

    Returns
    -------
    List[np.ndarray]
        List of feature arrays, one per dataset. Each array has shape
        (n_samples, embed_dim).

    Raises
    ------
    ValueError
        If the model_name is not supported by ModelRegistry.

    Notes
    -----
    Always provide ``dataset_name`` and ``base_path`` for efficient caching.
    Without these, embeddings are recomputed on every call.
    """
    # Get or create cache manager
    cache_manager = get_global_cache(cache_dir) if use_cache else None

    # Create extractor using ModelRegistry
    try:
        extractor = _get_model_registry().create(
            model_name,
            device=device,
            batch_size=batch_size,
            proxy=proxy,
            cache_manager=cache_manager,
        )
    except ValueError as e:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Use ModelRegistry.create() with supported models like:\n"
            f"  - 'inception_v3' (InceptionV3)\n"
            f"  - 'dino-small/base/large/giant' (DINOv2)\n"
            f"  - 'clip-base-patch16/base-patch32/large-patch14' (CLIP)\n"
            f"  - 'siglip-base-patch16-224/384' (SigLIP)\n"
            f"  - 'mae-base/large/huge' (MAE)\n"
            f"Original error: {e}"
        )

    # Extract features for each dataset
    features_list = []
    for dataset in dataset_list:
        image_paths = [str(item["image"]) for item in dataset]

        # Use extractor's extract_embeddings with caching support
        # logger.debug(f"This is called with use_cache={use_cache}, dataset_name={dataset_name}, base_path={base_path} and extractor_kwargs={extractor_kwargs}")
        features = extractor.extract_embeddings(
            images=image_paths,
            use_cache=use_cache,
            dataset_name=dataset_name,
            base_path=base_path,
            **extractor_kwargs,
        )
        features_list.append(features)

    return features_list


def _compute_pairwise(items, metric_fn, return_all=False):
    """Compute pairwise metrics across all pairs of items.

    For n items, computes n*(n-1)/2 pairwise comparisons.

    Parameters
    ----------
    items : list
        List of items to compare pairwise.
    metric_fn : callable
        Function that takes two items and returns a metric value.
    return_all : bool, optional
        If True, return list of all pairwise values. If False, return mean.
        Default: False.

    Returns
    -------
    float or list
        Mean of pairwise metrics, or list of all pairwise values.

    Raises
    ------
    ValueError
        If fewer than 2 items are provided.
    """
    n = len(items)
    if n < 2:
        raise ValueError("At least two items required for pairwise comparison.")

    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(metric_fn(items[i], items[j]))

    return scores if return_all else np.mean(scores)


# --- Label Metrics --- #


def JS_divergence(dataset_list, num_classes=None, return_all=False):
    """Compute Jensen-Shannon Divergence between label distributions.

    Measures symmetric divergence between probability distributions of class
    labels across clients. JSD is a symmetric and finite version of KL divergence,
    bounded in [0, log(2)] for binary comparisons.

    Args:
        dataset_list: List of datasets, each containing dicts with 'label' key
        num_classes: Number of classes (auto-detected if None)
        return_all: If True, return list of all pairwise JSD values;
            if False, return mean

    Returns:
        Mean JSD across all client pairs, or list of pairwise JSD values

    Examples:
        >>> client_data = [
        ...     [{"image": "a.jpg", "label": 0}, {"image": "b.jpg", "label": 1}],
        ...     [{"image": "c.jpg", "label": 0}, {"image": "d.jpg", "label": 0}],
        ... ]
        >>> jsd = JS_divergence(client_data, num_classes=2)
        >>> f"JSD: {jsd:.4f}"

    Notes:
        - Higher JSD indicates more heterogeneous label distributions
        - JSD = 0 when distributions are identical
        - JSD ≈ 0.69 (log 2) when distributions are completely disjoint
    """
    labels = []
    for d in dataset_list:
        if len(d) == 0:
            raise ValueError("Datasets must contain at least one sample each.")
        labels.append(np.array([item["label"] for item in d]))
    if num_classes is None:
        num_classes = max(l.max() for l in labels) + 1
    dists = []
    for l in labels:
        c = np.bincount(l, minlength=num_classes) + 1e-10
        dists.append(c / c.sum())

    def compute_jsd(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

    return _compute_pairwise(dists, compute_jsd, return_all)


def TV_label_distributions(dataset_list, num_classes=None, return_all=False):
    """Compute Total Variation distance between label distributions.

    Measures L1 distance between probability distributions of class labels.
    Total Variation is the maximum difference in probabilities assigned to any
    event, bounded in [0, 1].

    Args:
        dataset_list: List of datasets, each containing dicts with 'label' key
        num_classes: Number of classes (auto-detected if None)
        return_all: If True, return list of all pairwise TV values;
            if False, return mean

    Returns:
        Mean TV distance across all client pairs, or list of pairwise TV distances

    Examples:
        >>> client_data = [
        ...     [{"image": "a.jpg", "label": 0}, {"image": "b.jpg", "label": 1}],
        ...     [{"image": "c.jpg", "label": 1}, {"image": "d.jpg", "label": 1}],
        ... ]
        >>> tv = TV_label_distributions(client_data, num_classes=2)
        >>> f"Total Variation: {tv:.4f}"

    Notes:
        - TV = 0 when distributions are identical
        - TV = 1 when distributions are completely disjoint (no overlap)
        - TV = 0.5 * ||p - q||_1 (half of L1 distance)
    """
    labels = [np.array([item["label"] for item in d]) for d in dataset_list]
    if num_classes is None:
        num_classes = max(l.max() for l in labels) + 1
    dists = []
    for l in labels:
        c = np.bincount(l, minlength=num_classes) + 1e-10
        dists.append(c / c.sum())
    return _compute_pairwise(dists, lambda p, q: 0.5 * np.sum(np.abs(p - q)), return_all)


def JSD_client_to_global(dataset_list, num_classes=None, return_all=False):
    """Compute JSD from each client distribution to global distribution.

    Measures how much each client's label distribution diverges from the
    combined global distribution (all clients' data pooled together).
    Useful for identifying clients with unusual label distributions.

    Args:
        dataset_list: List of datasets, each containing dicts with 'label' key
        num_classes: Number of classes (auto-detected if None)
        return_all: If True, return list of per-client JSD values;
            if False, return mean

    Returns:
        Mean JSD to global distribution, or list of per-client JSD values

    Examples:
        >>> client_data = [
        ...     [{"image": "a.jpg", "label": 0}] * 90 + [{"image": "b.jpg", "label": 1}] * 10,
        ...     [{"image": "c.jpg", "label": 0}] * 50 + [{"image": "d.jpg", "label": 1}] * 50,
        ...     [{"image": "e.jpg", "label": 0}] * 10 + [{"image": "f.jpg", "label": 1}] * 90,
        ... ]
        >>> jsd_list = JSD_client_to_global(client_data, num_classes=2, return_all=True)
        >>> # jsd_list[0] is high (client 0 differs from balanced global)
        >>> # jsd_list[1] is low (client 1 matches balanced global)

    Notes:
        - Global distribution is computed by pooling all clients' data
        - Useful for one-to-many comparisons (each client vs. global)
        - Complementary to pairwise JSD which measures client-to-client heterogeneity
    """
    labels = [np.array([item["label"] for item in d]) for d in dataset_list]
    if num_classes is None:
        num_classes = max(l.max() for l in labels) + 1
    all_labels = np.concatenate(labels)
    global_counts = np.bincount(all_labels, minlength=num_classes) + 1e-10
    global_dist = global_counts / global_counts.sum()
    jsd_list = []
    for l in labels:
        c = np.bincount(l, minlength=num_classes)
        client_dist = (c + 1e-10) / c.sum()
        m = 0.5 * (client_dist + global_dist)
        jsd = 0.5 * (stats.entropy(client_dist, m) + stats.entropy(global_dist, m))
        jsd_list.append(jsd)
    return jsd_list if return_all else float(np.mean(jsd_list))


# --- Feature Metrics --- #


def FID_distance(dataset_list, return_all=False, **kwargs):
    """Compute Frechet Inception Distance (FID) between datasets.

    FID measures the distance between two multivariate Gaussians fitted to
    InceptionV3 features. Lower FID indicates more similar distributions.

    Parameters
    ----------
    dataset_list : List[List[Dict]]
        List of datasets, each containing dicts with 'image' key.
    return_all : bool, optional
        If True, return list of pairwise FID values; if False, return mean.
        Default: False.
    **kwargs
        Passed to _extract_features. Note: model_name is forced to 'inception_v3'.

        Common kwargs:

        - use_cache : bool - Enable caching (default: True)
        - dataset_name : str - Dataset identifier for cache
        - base_path : str - Base path for cache
        - batch_size : int - Batch size for extraction

    Returns
    -------
    float or list
        Mean FID across all client pairs, or list of pairwise FID values.

    Raises
    ------
    ValueError
        If fewer than 2 datasets are provided.

    Notes
    -----
    - FID is only meaningful with InceptionV3 features (hardcoded)
    - Lower FID = more similar distributions
    - FID = 0 if and only if distributions are identical
    - Sensitive to sample size (may underestimate with small samples)

    References
    ----------
    Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge
    to a Local Nash Equilibrium" (2017)
    """
    if len(dataset_list) < 2:
        raise ValueError("At least two datasets are required.")

    # FID is defined for InceptionV3 features only
    kwargs["model_name"] = "inception_v3"
    features = _extract_features(dataset_list, **kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_list = [torch.as_tensor(f, device=device, dtype=torch.float64) for f in features]

    def compute_stats(act):
        return torch.mean(act, dim=0), torch.cov(act.T)

    def compute_fid(stats1, stats2):
        mu1, sigma1 = stats1
        mu2, sigma2 = stats2

        mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
        sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

        diff = mu1 - mu2
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    def compute_fid_gpu(stat1, stat2):
        mu1, sigma1 = stat1
        mu2, sigma2 = stat2
        diff = mu1 - mu2
        ssd = diff.dot(diff)
        covprod = sigma1 @ sigma2
        
        # Compute eigenvalues of the product
        # The trace of the matrix square root = sum of sqrt of eigenvalues
        # Note: Σ1Σ2 is not necessarily symmetric, but its eigenvalues are real/non-negative
        eigvals = torch.linalg.eigvals(covprod)
        eigvals = eigvals.real
        eigvals = torch.clamp(eigvals, min=0.0)
        tr_covmean = torch.sum(torch.sqrt(eigvals))
        fid = ssd + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
        
        return max(0.0, fid.item())

    stats_list = [compute_stats(f) for f in features_list]
    return _compute_pairwise(stats_list, compute_fid_gpu, return_all)


def MMD(dataset_list, sigma=None, return_all=False, **kwargs):
    """Compute Maximum Mean Discrepancy between feature distributions.

    Non-parametric two-sample test based on embedding distributions into a
    reproducing kernel Hilbert space (RKHS) using RBF kernel. Measures distance
    between distribution means in the RKHS.

    Args:
        dataset_list: List of datasets with image paths
        sigma: RBF kernel bandwidth (auto-computed from median heuristic if None)
        return_all: If True, return list of pairwise MMD values; if False, return mean
        **kwargs: Passed to _extract_features (model_name, use_cache, etc.)

    Returns:
        Mean MMD across all client pairs, or list of pairwise MMD values

    Examples:
        >>> client_data = [
        ...     [{"image": "img1.jpg", "label": 0}, {"image": "img2.jpg", "label": 1}],
        ...     [{"image": "img3.jpg", "label": 0}, {"image": "img4.jpg", "label": 0}],
        ... ]
        >>> mmd = MMD(client_data, model_name="dinov2-base", use_cache=True)
        >>> f"MMD: {mmd:.4f}"

    Notes:
        - MMD ≈ 0 when distributions are identical
        - Higher MMD indicates more heterogeneous feature distributions
        - Uses median heuristic for sigma if not specified
        - Supports any model from ModelRegistry (InceptionV3, DINOv2, CLIP, etc.)

    References:
        Gretton et al., "A Kernel Two-Sample Test" (2012)
    """
    features = _extract_features(dataset_list, **kwargs)
    return _compute_pairwise(features, lambda f1, f2: MMD_rbf_kernel(f1, f2, sigma), return_all)


def energy_distance_features(dataset_list, return_all=False, **kwargs):
    """Compute Energy Distance between feature distributions.

    Energy distance is a statistical distance based on Euclidean distances
    between sample points. It characterizes equality of distributions and is
    zero if and only if distributions are identical.

    Args:
        dataset_list: List of datasets with image paths
        return_all: If True, return list of pairwise energy distances;
            if False, return mean
        **kwargs: Passed to _extract_features (model_name, use_cache, etc.)

    Returns:
        Mean energy distance across all client pairs, or list of pairwise values

    Examples:
        >>> client_data = [
        ...     [{"image": "img1.jpg", "label": 0}, {"image": "img2.jpg", "label": 1}],
        ...     [{"image": "img3.jpg", "label": 1}, {"image": "img4.jpg", "label": 1}],
        ... ]
        >>> ed = energy_distance_features(client_data, model_name="clip-vit-b-32")
        >>> f"Energy Distance: {ed:.4f}"

    Notes:
        - Energy distance = 0 if and only if distributions are identical
        - Metric distance (satisfies triangle inequality)
        - Generalizes to arbitrary dimensions unlike Wasserstein
        - Computationally efficient (O(n²) pairwise distances)

    References:
        Székely & Rizzo, "Energy statistics: A class of statistics based on
        distances" (2013)
    """
    features = _extract_features(dataset_list, **kwargs)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def energy_distance(X, Y):
        dXX = cdist(X, X, "euclidean")
        dYY = cdist(Y, Y, "euclidean")
        dXY = cdist(X, Y, "euclidean")
        return 2 * np.mean(dXY) - np.mean(dXX) - np.mean(dYY)

    def energy_distance_gpu(X, Y):
        # Convert to GPU
        X = torch.as_tensor(X, device=device, dtype=torch.float32)
        Y = torch.as_tensor(Y, device=device, dtype=torch.float32)
        
        # Helper for mean euclidean distance
        def mean_dist(A, B):
            # p=2 is Euclidean
            return torch.cdist(A, B, p=2).mean()

        dXY = mean_dist(X, Y)
        dXX = mean_dist(X, X)
        dYY = mean_dist(Y, Y)
        
        return (2 * dXY - dXX - dYY).item()

    return _compute_pairwise(features, energy_distance_gpu if torch.cuda.is_available() else energy_distance, return_all)


def sliced_wasserstein_distance(dataset_list, return_all=False, **kwargs):
    """Compute Sliced Wasserstein Distance between feature distributions.

    Efficiently approximates Wasserstein distance by averaging 1D Wasserstein
    distances along random projections. Reduces high-dimensional optimal
    transport problem to many 1D problems with closed-form solutions.

    Args:
        dataset_list: List of datasets with image paths
        return_all: If True, return list of pairwise distances; if False, return mean
        **kwargs: Passed to _extract_features (model_name, use_cache, etc.)

    Returns:
        Mean sliced Wasserstein distance, or list of pairwise distances

    Examples:
        >>> client_data = [
        ...     [{"image": "img1.jpg", "label": 0}, {"image": "img2.jpg", "label": 1}],
        ...     [{"image": "img3.jpg", "label": 0}, {"image": "img4.jpg", "label": 0}],
        ... ]
        >>> swd = sliced_wasserstein_distance(
        ...     client_data,
        ...     model_name="siglip-base-patch16-384"
        ... )
        >>> f"Sliced Wasserstein: {swd:.4f}"

    Notes:
        - Uses 500 random projections for approximation
        - Much faster than exact Wasserstein (especially in high dimensions)
        - Converges to true Wasserstein as num_projections → ∞
        - Metric distance (satisfies triangle inequality)
        - SWD = 0 if and only if distributions are identical

    References:
        Bonneel et al., "Sliced and Radon Wasserstein Barycenters of Measures" (2015)
    """
    features = _extract_features(dataset_list, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = features[0].shape[1]

    seed = kwargs.get("random_seed", 42)
    if seed is not None:
        torch.manual_seed(seed)
    
    num_projections = kwargs.get("num_projections", 500)
    projections = torch.randn((dim, num_projections), device=device, dtype=torch.float32)
    projections = torch.nn.functional.normalize(projections, p=2, dim=0)

    def compute_swd_gpu(X_np, Y_np):
        X = torch.as_tensor(X_np, device=device, dtype=torch.float32)
        Y = torch.as_tensor(Y_np, device=device, dtype=torch.float32)
        
        X_proj = X @ projections
        Y_proj = Y @ projections
        
        # Sorting along the sample dimension (dim=0) gives us the empirical quantile function (Inverse CDF)
        X_sorted, _ = torch.sort(X_proj, dim=0)
        Y_sorted, _ = torch.sort(Y_proj, dim=0)
        
        # Handle unequal sample sizes (N1 != N2)
        if X_sorted.shape[0] != Y_sorted.shape[0]:
            # We must interpolate the smaller one to match the other.
            target_size = max(X_sorted.shape[0], Y_sorted.shape[0])
            
            def _interpolate_to_target(tensor, target_len):
                t_reshaped = tensor.t().unsqueeze(0) 
                t_interp = torch.nn.functional.interpolate(t_reshaped, size=target_len, mode='linear', align_corners=True)
                return t_interp.squeeze(0).t()

            X_sorted = _interpolate_to_target(X_sorted, target_size)
            Y_sorted = _interpolate_to_target(Y_sorted, target_size)

        swd = torch.abs(X_sorted - Y_sorted).mean()       
        return swd.item()

    return _compute_pairwise(features, compute_swd_gpu, return_all)


def KS_multivariate_test(
    dataset_list,
    alpha: float = 0.05,
    pca_components: Optional[int] = None,
    pca_variance: Optional[float] = None,
    return_all: bool = False,
    **kwargs,
):
    """Multivariate Kolmogorov-Smirnov test with optional PCA preprocessing.

    Tests whether client feature distributions are statistically different
    using per-dimension KS tests with Bonferroni correction for multiple
    comparisons.

    Parameters
    ----------
    dataset_list : List[List[Dict]]
        List of datasets, each containing dicts with 'image' key.
    alpha : float, optional
        Significance level for the hypothesis test. Bonferroni correction
        is applied (alpha / n_dimensions). Default: 0.05.
    pca_components : int, optional
        If provided, reduce to this many PCA components before testing.
    pca_variance : float, optional
        If provided, reduce to components explaining this variance
        (e.g., 0.95 for 95%). Overrides pca_components.
    return_all : bool, optional
        If True, return list of pairwise test results. Default: False.
    **kwargs
        Passed to _extract_features (model_name, use_cache, etc.).

    Returns
    -------
    dict or list
        If return_all=False, returns aggregate result:

        - "reject_null": True if any pair rejects H0 (distributions differ)
        - "min_pvalue": Minimum p-value across all pairs and dimensions
        - "significant_dims": Mean number of dimensions rejecting H0

        If return_all=True, returns list of per-pair dicts.

    Notes
    -----
    - Uses Bonferroni correction for multiple testing
    - High-dimensional tests benefit from PCA dimensionality reduction
    - Null hypothesis: distributions are identical
    """
    features = _extract_features(dataset_list, **kwargs)

    if pca_components is not None or pca_variance is not None:
        all_features = np.vstack(features)

        if pca_variance is not None:
            pca_temp = _get_pca_class()()
            pca_temp.fit(all_features)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = int(np.argmax(cumsum_var >= pca_variance) + 1)
            log.info("PCA: Using %s components to explain %.1f%% variance", n_components, pca_variance * 100.0)
        else:
            n_components = pca_components

        pca = _get_pca_class()(n_components=n_components)
        pca.fit(all_features)
        features = [pca.transform(f) for f in features]
        log.info("PCA: Reduced from %s to %s dimensions", all_features.shape[1], n_components)

    def compute_ks(f1, f2):
        n = f1.shape[1]
        pvalues = [stats.ks_2samp(f1[:, i], f2[:, i])[1] for i in range(n)]
        min_p = min(pvalues)

        return {
            "reject_null": min_p < (alpha / n),
            "min_pvalue": float(min_p),
            "significant_dims": sum(p < (alpha / n) for p in pvalues),
        }

    results = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            results.append(compute_ks(features[i], features[j]))

    if return_all:
        return results

    return {
        "reject_null": any(r["reject_null"] for r in results),
        "min_pvalue": min(r["min_pvalue"] for r in results),
        "significant_dims": np.mean([r["significant_dims"] for r in results]),
    }


def vendi_score(dataset_list, q=1, normalize=True, return_all=False, **kwargs):
    """Compute Vendi Score measuring diversity within each client's dataset.

    Vendi Score quantifies diversity using the effective number of elements in a
    dataset, generalizing from species diversity metrics. Higher scores indicate
    more diverse feature representations within a client.

    Args:
        dataset_list: List of datasets with image paths
        q: Diversity order parameter (q=1 is default, related to Shannon entropy)
        normalize: Whether to normalize the similarity matrix
        return_all: If True, return list of per-client scores; if False, return mean
        **kwargs: Passed to _extract_features (model_name, use_cache, etc.)

    Returns:
        Mean Vendi score across clients, or list of per-client scores

    Examples:
        >>> client_data = [
        ...     [{"image": "img1.jpg", "label": 0}, {"image": "img2.jpg", "label": 1}],
        ...     [{"image": "img3.jpg", "label": 0}, {"image": "img4.jpg", "label": 0}],
        ... ]
        >>> vendi = vendi_score(
        ...     client_data,
        ...     model_name="dinov2-large",
        ...     return_all=True
        ... )
        >>> # vendi[0] > vendi[1] (client 0 more diverse)

    Notes:
        - Higher Vendi score = more diverse client dataset
        - Score of 1.0 means all samples are identical
        - Score of N means all N samples are maximally diverse
        - Useful for identifying clients with low intra-client diversity
        - Complements inter-client heterogeneity metrics (FID, MMD, etc.)

    References:
        Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric for
        Machine Learning" (2023)
    """
    features = _extract_features(dataset_list, **kwargs)
    scores = [_score_vendi(f, q=q, normalize=normalize) for f in features]
    return scores if return_all else float(np.mean(scores))


def intrinsic_dimensionality(
    dataset_list: List[List[Dict[str, Any]]],
    method: str = "mle",
    k: int = 10,
    return_all: bool = False,
    **kwargs,
) -> Union[List[float], float]:
    """Estimate intrinsic dimensionality of feature distributions.

    Intrinsic dimensionality measures the true "complexity" of data,
    which is often much lower than the ambient dimension. Different
    clients may have different intrinsic dimensions if their data
    varies in complexity.

    Parameters
    ----------
    dataset_list : List[List[Dict]]
        List of datasets, each containing dicts with 'image' key.
    method : str, optional
        Estimation method:

        - "two_nn": TWO-NN estimator (fast, robust)
        - "mle": Maximum Likelihood Estimator (more accurate, uses k neighbors)

        Default: "mle".
    k : int, optional
        Number of neighbors for MLE method. Default: 10.
    use_faiss : bool, optional
        Whether to use FAISS for fast nearest neighbor search.
        Falls back to sklearn if FAISS not available. Default: True.
    return_all : bool, optional
        If True, return list of per-dataset estimates; if False, return mean.
        Default: False.
    **kwargs
        Passed to _extract_features (model_name, use_cache, etc.).

    Returns
    -------
    float or List[float]
        Mean intrinsic dimensionality across datasets, or list of
        per-dataset estimates.

    Notes
    -----
    - TWO-NN uses only 2 nearest neighbors (very efficient)
    - MLE is more accurate but sensitive to k
    - Lower ID may indicate more structured/homogeneous client data
    - NaN returned for datasets where estimation fails

    References
    ----------
    - TWO-NN: Facco et al., "Estimating the intrinsic dimension of datasets
      by a minimal neighborhood information" (2017)
    - MLE: Levina & Bickel, "Maximum Likelihood Estimation of Intrinsic
      Dimension" (2004)
    """

    features = _extract_features(dataset_list, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def two_nn_torch(X_np: np.ndarray) -> float:
        N = X_np.shape[0]
        if N < 3: return 0.0

        X = torch.as_tensor(X_np, device=device, dtype=torch.float32)
        dists = torch.cdist(X, X, p=2)

        top_dists, _ = dists.topk(3, dim=1, largest=False)
        r1 = top_dists[:, 1]
        r2 = top_dists[:, 2]

        valid_mask = r1 > 1e-10
        r1 = r1[valid_mask]
        r2 = r2[valid_mask]
        mu = r2 / r1

        id_val = r1.shape[0] / torch.sum(torch.log(mu))

        return id_val.item()

    def mle_torch(X_np: np.ndarray, k_neighbors: int) -> float:

        N = X_np.shape[0]
        if N < k_neighbors + 1: return 0.0
        
        X = torch.as_tensor(X_np, device=device, dtype=torch.float32)
        
        # Compute distances
        dists = torch.cdist(X, X, p=2)
        top_dists, _ = dists.topk(k_neighbors + 1, dim=1, largest=False)

        r_k = top_dists[:, k_neighbors]      # The radius of the k-th neighbor
        r_i = top_dists[:, 1:k_neighbors+1]  # Radii of all k neighbors

        valid = r_k > 1e-10
        r_k = r_k[valid]
        r_i = r_i[valid]
        
        if r_k.shape[0] == 0: return 0.0
        
        # Formula: m_i = [ 1/(k-1) * sum(log(r_k / r_j)) ] ^ -1
        log_ratios = torch.log(r_k.unsqueeze(1) / (r_i + 1e-10))
        
        # Mean of logs -> Invert -> Mean across samples
        m_i = 1.0 / (torch.mean(log_ratios, dim=1) + 1e-10)
        
        return torch.mean(m_i).item()

    # Select estimator
    if method == "two_nn":
        estimator = two_nn_torch
    elif method == "mle":
        estimator = lambda X: mle_torch(X, k)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'two_nn' or 'mle'.")

    # Estimate for each dataset
    id_estimates = []
    for i, feat in enumerate(features):
        try:
            val = estimator(feat)
            id_estimates.append(val)
        except ID_ESTIMATION_ERRORS as exc:
            log.warning("Failed to estimate ID for dataset %s: %s", i, exc)
            id_estimates.append(np.nan)

    return id_estimates if return_all else float(np.nanmean(id_estimates))


def knn_separability(dataset_list, k=5, return_all=False, **kwargs):
    """Compute k-nearest neighbor separability across clients.

    Measures how well client data clusters in feature space. For each sample,
    computes the fraction of its k nearest neighbors that belong to the same
    client. Higher values indicate more separable (heterogeneous) client data.

    Args:
        dataset_list: List of datasets with image paths
        k: Number of nearest neighbors to consider
        return_all: If True, return list of per-sample isolation scores;
            if False, return mean
        **kwargs: Passed to _extract_features (model_name, use_cache, etc.)

    Returns:
        Mean k-NN isolation score, or list of per-sample scores

    Examples:
        >>> client_data = [
        ...     [{"image": "dog1.jpg", "label": 0}, {"image": "dog2.jpg", "label": 0}],
        ...     [{"image": "cat1.jpg", "label": 1}, {"image": "cat2.jpg", "label": 1}],
        ... ]
        >>> separability = knn_separability(
        ...     client_data,
        ...     k=5,
        ...     model_name="clip-vit-l-14"
        ... )
        >>> # High separability → clients cluster separately in feature space

    Notes:
        - Score in [0, 1] where 1.0 means all neighbors from same client
        - Higher scores indicate more heterogeneous feature distributions
        - Sensitive to k: small k emphasizes local structure, large k global
        - Complements distance-based metrics (MMD, Wasserstein) with topology
        - Useful for visualizing client clustering via t-SNE/UMAP

    References:
        Related to Hopkins statistic for clustering tendency
    """
    features = _extract_features(dataset_list, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = []
    all_features = []
    for idx, f in enumerate(features):
        t = torch.as_tensor(f, device=device, dtype=torch.float32)
        all_features.append(t)
        labels.append(torch.full((t.size(0),), idx, device=device, dtype=torch.long))        
    X = torch.cat(all_features, dim=0)
    y = torch.cat(labels, dim=0)

    del features, labels, all_features
    torch.cuda.empty_cache()
    
    try:
        dists = torch.cdist(X, X, p=2)
        dists.fill_diagonal_(float('inf'))  # Exclude self in nearest neighbors
        knn_indices = dists.topk(k, dim=1, largest=False).indices  # (N, k)
        knn_labels = y[knn_indices]  # (N, k)
        matches = (knn_labels == y.unsqueeze(1)).float()  # (N, k)
        iso_scores = matches.mean(dim=1)  # (N,)

        return iso_scores.cpu().tolist() if return_all else float(iso_scores.mean().item())

    except RuntimeError as e:
        if 'out of memory' in str(e):
            log.warning(
                "CUDA out of memory during k-NN computation. Falling back to batched implementation."
            )
            torch.cuda.empty_cache()
            
            batch_size = 5000
            N = X.shape[0]
            all_scores = []
            for i in range(0, N, batch_size):

                X_batch = X[i : i + batch_size]
                dists_batch = torch.cdist(X_batch, X, p=2)
                # where global_col == i + local_row
                global_indices = torch.arange(i, i + X_batch.shape[0], device=device).view(-1, 1)
                dists_batch.scatter_(1, global_indices, float('inf'))
                knn_indices = dists_batch.topk(k, dim=1, largest=False).indices
                neighbor_labels = y[knn_indices]
                query_labels = y[global_indices]
                
                matches = (neighbor_labels == query_labels).float()
                batch_scores = matches.mean(dim=1)
                all_scores.append(batch_scores)
                del dists_batch, knn_indices, matches

            final_scores = torch.cat(all_scores)

            if return_all:
                return final_scores.cpu().tolist()
            else:
                return float(final_scores.mean().item())


def centroid_distance(dataset_list, return_all=False, **kwargs):
    """Compute Euclidean distance between client feature centroids.

    Measures the distance between mean feature vectors (centroids) of each
    client's data. Simple and interpretable metric for feature distribution
    heterogeneity, though less sensitive than distribution-based metrics.

    Args:
        dataset_list: List of datasets with image paths
        return_all: If True, return list of pairwise distances; if False, return mean
        **kwargs: Passed to _extract_features (model_name, use_cache, etc.)

    Returns:
        Mean centroid distance across all client pairs, or list of pairwise distances

    Examples:
        >>> client_data = [
        ...     [{"image": "img1.jpg", "label": 0}, {"image": "img2.jpg", "label": 1}],
        ...     [{"image": "img3.jpg", "label": 0}, {"image": "img4.jpg", "label": 0}],
        ... ]
        >>> cd = centroid_distance(client_data, model_name="mae-base")
        >>> f"Centroid Distance: {cd:.4f}"

    Notes:
        - Higher distance indicates clients have different mean features
        - Only captures first-order statistics (ignores variance/shape)
        - Fast to compute (single mean per client)
        - Less sensitive than MMD/Wasserstein which capture full distributions
        - Useful for quick heterogeneity assessment
    """
    features = _extract_features(dataset_list, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centroids = [torch.mean(torch.as_tensor(f, device=device), dim=0) for f in features]
    return _compute_pairwise(centroids, lambda c1, c2: torch.linalg.norm(c1 - c2).item(), return_all)


def feature_variance(dataset_list, return_all=False, **kwargs):
    """Compute feature variance within each client's dataset.

    Measures the spread of feature vectors within each client. Higher variance
    indicates more diverse intra-client data. Complements inter-client
    heterogeneity metrics.

    Args:
        dataset_list: List of datasets with image paths
        return_all: If True, return list of per-client variances;
            if False, return mean
        **kwargs: Passed to _extract_features (model_name, use_cache, etc.)

    Returns:
        Mean feature variance across clients, or list of per-client variances

    Examples:
        >>> client_data = [
        ...     [{"image": "img1.jpg", "label": 0}, {"image": "img2.jpg", "label": 1}],
        ...     [{"image": "img3.jpg", "label": 0}, {"image": "img4.jpg", "label": 0}],
        ... ]
        >>> variances = feature_variance(
        ...     client_data,
        ...     model_name="dinov2-base",
        ...     return_all=True
        ... )
        >>> # variances[0] > variances[1] (client 0 more diverse)

    Notes:
        - Higher variance = more diverse client dataset
        - Measures intra-client diversity (within-client heterogeneity)
        - Complements metrics like Vendi score
        - Useful for identifying clients with homogeneous data
        - Total variance = mean(variances) measures overall diversity
    """
    features = _extract_features(dataset_list, **kwargs)
    variances = [float(np.sum((f - np.mean(f, axis=0)) ** 2) / (f.shape[0] - 1)) for f in features]
    return variances if return_all else np.mean(variances)


# --- Utils ---


def MMD_rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: Optional[float] = None,
    sigma_samples: int = 1000,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> float:
    """Calculate squared Maximum Mean Discrepancy using RBF kernel.

    This is the core MMD computation used by the :func:`MMD` metric.
    Uses unbiased estimator with proper diagonal handling.

    Parameters
    ----------
    X : np.ndarray
        First set of samples, shape (m, d).
    Y : np.ndarray
        Second set of samples, shape (n, d).
    sigma : float, optional
        RBF kernel bandwidth. If None, uses median heuristic
        (median of non-zero pairwise distances).
    sigma_samples : int, optional
        Number of samples to use for median heuristic if sigma is None. Default: 1000.
    device : torch.device, optional
        Device to perform computation on (CPU or GPU). If None,
        defaults to GPU if available, else CPU.
    seed : int, optional
        Random seed for reproducibility when computing median heuristic.

    Returns
    -------
    float
        Squared MMD value (non-negative).

    Notes
    -----
    Uses unbiased estimator: excludes self-comparisons from kernel matrices.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # default to GPU if available

    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    m, d = X.shape
    n, d2 = Y.shape

    if m < 2 or n < 2:
        raise ValueError("Each sample must have at least 2 points for unbiased MMD.")

    if d != d2:
        raise ValueError("Feature dimensions must match.")

    # --- 1. Convert to torch tensors on specified device ---
    X = torch.from_numpy(X).to(device).float()
    Y = torch.from_numpy(Y).to(device).float()

    # --- 2. Compute sigma using median heuristic if not provided ---
    if sigma is None:
        if seed is not None:
            torch.manual_seed(seed)
        if max(m, n) > sigma_samples:
            indices_x = torch.randperm(m, device=device)[:sigma_samples]
            indices_y = torch.randperm(n, device=device)[:sigma_samples]
            combined = torch.cat([X[indices_x], Y[indices_y]], dim=0)
        else:
            combined = torch.cat([X, Y], dim=0)
        
        sq_dists = torch.cdist(combined, combined, p=2.0).pow(2).view(-1)
        # dists = torch.pdist(torch.cat([x, y], dim=0)[:,None])
        sq_dists = sq_dists[sq_dists > 0]  # exclude zero distances

        denominator = sq_dists.median() if sq_dists.numel() > 0 else torch.tensor(1.0, device=device)

        del combined, sq_dists
        torch.cuda.empty_cache()
    
    else:
        denominator = torch.tensor(2 * sigma**2, device=device)

    def _rbf_kernel_sum(A, B, exclude_diag) -> float:
        sq_dists = torch.cdist(A, B, p=2.0).pow(2)
        K = torch.exp(-sq_dists / denominator)
        
        if exclude_diag:
            K.fill_diagonal_(0.0)
        return K.sum().item()

    # batched computation of sums for GPUs with less memory
    def _batched_rbf_kernel_sum(A, B, exclude_diag, batch_size=5000) -> float:
            total_sum = 0.0
            n_A = A.shape[0]
            n_B = B.shape[0]
            
            for i in range(0, n_A, batch_size):
                end_i = min(i + batch_size, n_A)
                A_chunk = A[i:end_i]
                for j in range(0, n_B, batch_size):
                    end_j = min(j + batch_size, n_B)
                    B_chunk = B[j:end_j]

                    dists_blk = torch.cdist(A_chunk, B_chunk, p=2.0).pow(2)
                    K_blk = torch.exp(-dists_blk / denominator)

                    if exclude_diag:
                        # Generate global indices for the current block
                        rows = torch.arange(i, end_i, device=device).view(-1, 1)
                        cols = torch.arange(j, end_j, device=device).view(1, -1)
                        mask = (rows == cols)
                        K_blk.masked_fill_(mask, 0.0)
                    
                    total_sum += K_blk.sum().item()
                    
            return total_sum

    # If OOM occurs, try batched computation
    try:
        sum_XY = _rbf_kernel_sum(X, Y, exclude_diag=False)
        sum_XX = _rbf_kernel_sum(X, X, exclude_diag=True)
        sum_YY = _rbf_kernel_sum(Y, Y, exclude_diag=True)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            sum_XY = _batched_rbf_kernel_sum(X, Y, exclude_diag=False)
            sum_XX = _batched_rbf_kernel_sum(X, X, exclude_diag=True)
            sum_YY = _batched_rbf_kernel_sum(Y, Y, exclude_diag=True)
        else:
            raise e

    mmd_squared = (sum_XX / (m * (m - 1))) + \
                  (sum_YY / (n * (n - 1))) - \
                  (2 * sum_XY / (m * n))

    return max(0.0, mmd_squared)
