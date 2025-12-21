import numpy as np
from torchvision import models, transforms
import torch
import scipy
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.decomposition import PCA
from PIL import Image
import os
from typing import List, Dict, Optional, Any, Union
from vendi_score import vendi

from embedding import CacheManager, ModelRegistry
from data_loading.dataset import ImageDataset

_GLOBAL_CACHE: Optional[CacheManager] = None

"""
List of available metrics

- Label Metrics:
    - Jensen-Shannon Divergence (JSD) -> across many?
    - Total Variation (TV)
    - JSD one-to-many

- Image/Feature/Embedding Metrics:
    - FID (hardcoded to InceptionV3 features)
    - MMD w/ RBF kernel
    - Energy Distance
    - Sliced Wasserstein Distance
    - Kolmogorov-Smirnov Multivariate Test (PCA optional)
    - Vendi Score
    - Intrinsic Dimensionality using i) MLE ii) TWO-NN
    - k-NN Separability
    - Centroid Distance
    - Feature Variance
"""


def get_global_cache(cache_dir: str = "./.embedding_cache") -> CacheManager:
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
    cache_dir: str = "./.embedding_cache",
    dataset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    proxy: Optional[str] = None,
    **extractor_kwargs: Any,
) -> List[np.ndarray]:
    """
    Extract features from a list of datasets using any model from ModelRegistry.

    Args:
        dataset_list: List of datasets, each is [{"image": path, "label": label}, ...]
        model_name: Model identifier (e.g., "inception_v3", "dino-base", "clip-base-patch16")
        device: Device for computation ("cuda", "cpu", or None for auto-detect)
        batch_size: Batch size for feature extraction
        use_cache: Whether to use caching
        cache_dir: Directory for cache storage
        dataset_name: Dataset identifier for cache lookup
        base_path: Base path for relative path computation in cache
        proxy: HTTP proxy for model downloads
        **extractor_kwargs: Additional arguments passed to extractor (e.g., return_which_tokens)

    Returns:
        List of feature arrays, one per dataset
    """
    # Get or create cache manager
    cache_manager = get_global_cache(cache_dir) if use_cache else None

    # Create extractor using ModelRegistry
    try:
        extractor = ModelRegistry.create(
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
        features = extractor.extract_embeddings(
            images=image_paths,
            use_cache=use_cache,
            dataset_name=dataset_name,
            base_path=base_path,
            **extractor_kwargs,
        )
        features_list.append(features)

    return features_list

"""
Concept: Compare datasets D1 and D2 by computing pairwise metrics.
Compare -> 
are D1 and D2 the same distribution? <- KS-test
How "far" are D1 and D2? <- FID, MMD, SWD, Energy Distance
Other aspects of diversity <- Vendi Score, Intrinsic Dimensionality

So then if we have datasets = [D1, D2, D3, ..., Dn], we can compute pairwise metrics
and if return_all = False, return the mean across all pairs.
This way we can compare with a single number how heterogeneous two entire partitionings are.
"""


def _compute_pairwise(items, metric_fn, return_all=False):
    """Compute pairwise metrics. Returns mean or list of all pairs."""
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
        >>> print(f"JSD: {jsd:.4f}")

    Notes:
        - Higher JSD indicates more heterogeneous label distributions
        - JSD = 0 when distributions are identical
        - JSD ≈ 0.69 (log 2) when distributions are completely disjoint
    """
    labels = [np.array([item["label"] for item in d]) for d in dataset_list]
    if num_classes is None: num_classes = max(l.max() for l in labels) + 1
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
        >>> print(f"Total Variation: {tv:.4f}")

    Notes:
        - TV = 0 when distributions are identical
        - TV = 1 when distributions are completely disjoint (no overlap)
        - TV = 0.5 * ||p - q||_1 (half of L1 distance)
    """
    labels = [np.array([item["label"] for item in d]) for d in dataset_list]
    if num_classes is None: num_classes = max(l.max() for l in labels) + 1
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
    if num_classes is None: num_classes = max(l.max() for l in labels) + 1
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
    """
    Compute Fréchet Inception Distance (FID) between datasets.

    Note: FID is specifically designed for InceptionV3 features.
    This function forces model_name='inception_v3'.

    Args:
        dataset_list: List of datasets
        return_all: Return all pairwise FID scores or mean
        **kwargs: Passed to _extract_features (overridden: model_name='inception_v3')

    Returns:
        Mean FID or list of pairwise FID scores
    """
    if len(dataset_list) < 2:
        raise ValueError("At least two datasets are required.")

    # FID is defined for InceptionV3 features only
    kwargs['model_name'] = 'inception_v3'
    features = _extract_features(dataset_list, **kwargs)

    def compute_stats(act):
        return np.mean(act, axis=0), np.cov(act, rowvar=False)

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

    stats_list = [compute_stats(f) for f in features]
    return _compute_pairwise(stats_list, compute_fid, return_all)


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
        >>> print(f"MMD: {mmd:.4f}")

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
        >>> print(f"Energy Distance: {ed:.4f}")

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
    def energy_distance(X, Y):
        dXX = cdist(X, X, 'euclidean')
        dYY = cdist(Y, Y, 'euclidean')
        dXY = cdist(X, Y, 'euclidean')
        return 2 * np.mean(dXY) - np.mean(dXX) - np.mean(dYY)
    return _compute_pairwise(features, energy_distance, return_all)


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
        >>> print(f"Sliced Wasserstein: {swd:.4f}")

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
    # def wd_1d(f1, f2):
    #     return np.mean([stats.wasserstein_distance(f1[:, i], f2[:, i]) for i in range(f1.shape[1])])
    # switch to proper sliced wasserstein distance
    def compute_swd(f1, f2, num_projections=500):
        n_dims = f1.shape[1]
        projections = np.random.randn(num_projections, n_dims)
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)
        swd = []
        for p in projections:
            proj_f1 = f1.dot(p)
            proj_f2 = f2.dot(p)
            swd.append(stats.wasserstein_distance(proj_f1, proj_f2))
        return np.mean(swd)

    return _compute_pairwise(features, compute_swd, return_all)


def KS_multivariate_test(
    dataset_list,
    alpha: float = 0.05,
    pca_components: Optional[int] = None,
    pca_variance: Optional[float] = None,
    return_all: bool = False,
    **kwargs
):
    """
    Multivariate Kolmogorov-Smirnov test with optional PCA preprocessing.

    Args:
        dataset_list: List of datasets
        alpha: Significance level for Bonferroni correction
        pca_components: If provided, reduce to this many PCA components before test
        pca_variance: If provided, reduce to components explaining this variance (e.g., 0.95)
        return_all: Return all pairwise results or aggregate
        **kwargs: Passed to _extract_features (model_name, etc.)

    Returns:
        Dictionary with test results or list of pairwise results
    """
    features = _extract_features(dataset_list, **kwargs)

    if pca_components is not None or pca_variance is not None:
        all_features = np.vstack(features)

        if pca_variance is not None:
            pca_temp = PCA()
            pca_temp.fit(all_features)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = int(np.argmax(cumsum_var >= pca_variance) + 1)
            print(f"PCA: Using {n_components} components to explain {pca_variance:.1%} variance")
        else:
            n_components = pca_components

        pca = PCA(n_components=n_components)
        pca.fit(all_features)
        features = [pca.transform(f) for f in features]
        print(f"PCA: Reduced from {all_features.shape[1]} to {n_components} dimensions")

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
    scores = [float(vendi.score_dual(f, q=q, normalize=normalize)) for f in features]
    return scores if return_all else float(np.mean(scores))


def intrinsic_dimensionality(
    dataset_list: List[List[Dict[str, Any]]],
    method: str = "two_nn",
    k: int = 10,
    use_faiss: bool = True,
    return_all: bool = False,
    **kwargs
) -> Union[List[float], float]:
    """
    Estimate intrinsic dimensionality of feature distributions.

    Args:
        dataset_list: List of datasets
        method: Estimation method ("two_nn" or "mle")
        k: Number of neighbors for MLE method
        use_faiss: Use FAISS for GPU acceleration (if available)
        return_all: Return list of per-dataset estimates or mean
        **kwargs: Passed to _extract_features

    Returns:
        Intrinsic dimensionality estimate(s)

    References:
        - TWO-NN: Facco et al., "Estimating the intrinsic dimension of datasets
          by a minimal neighborhood information" (2017)
        - MLE: Levina & Bickel, "Maximum Likelihood Estimation of Intrinsic
          Dimension" (2004)
    """
    features = _extract_features(dataset_list, **kwargs)

    def two_nn_faiss(X: np.ndarray) -> float:
        """TWO-NN estimator using FAISS on GPU."""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS required for GPU acceleration. Install with: pip install faiss-gpu")

        X = np.ascontiguousarray(X.astype('float32'))
        n, d = X.shape

        # Build index on GPU if available
        try:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(d)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(X)

            # Search for 3 nearest neighbors (including self)
            distances, _ = gpu_index.search(X, 3)
        except RuntimeError:
            # Fallback to CPU if GPU not available
            print("GPU not available, using CPU FAISS")
            index = faiss.IndexFlatL2(d)
            index.add(X)
            distances, _ = index.search(X, 3)

        r1 = np.sqrt(distances[:, 1])
        r2 = np.sqrt(distances[:, 2])
        valid = r1 > 1e-10
        mu = r2[valid] / r1[valid]

        return len(mu) / np.sum(np.log(mu))

    def two_nn_sklearn(X: np.ndarray) -> float:
        from sklearn.neighbors import NearestNeighbors

        n, d = X.shape

        # Find 3 nearest neighbors (including self)
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)

        # TWO-NN formula
        r1 = distances[:, 1]  # Distance to 1st neighbor (excluding self)
        r2 = distances[:, 2]  # Distance to 2nd neighbor

        valid = r1 > 1e-10
        mu = r2[valid] / r1[valid]

        return len(mu) / np.sum(np.log(mu))

    def mle_faiss(X: np.ndarray, k: int) -> float:
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS required for GPU acceleration. Install with: pip install faiss-gpu")

        X = np.ascontiguousarray(X.astype('float32'))
        n, d = X.shape

        # Build index on GPU if available
        try:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(d)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(X)
            distances, _ = gpu_index.search(X, k + 1)
        except RuntimeError:
            print("GPU not available, using CPU FAISS")
            index = faiss.IndexFlatL2(d)
            index.add(X)
            distances, _ = index.search(X, k + 1)

        # MLE formula: m_hat = [1/k * Σ log(r_k / r_i)]^-1
        # distances[:, 0] is self, so skip it
        r_k = np.sqrt(distances[:, k])  # Distance to k-th neighbor
        r_i = np.sqrt(distances[:, 1:k + 1])  # Distances to neighbors 1...k

        # Compute for each point
        m_estimates = []
        for i in range(n):
            if r_k[i] > 1e-10:
                log_ratios = np.log(r_k[i] / (r_i[i] + 1e-10))
                m_i = 1.0 / (np.mean(log_ratios) + 1e-10)
                m_estimates.append(m_i)

        return float(np.mean(m_estimates))

    def mle_sklearn(X: np.ndarray, k: int) -> float:
        from sklearn.neighbors import NearestNeighbors

        n, d = X.shape

        # Find k+1 nearest neighbors (including self)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)

        # MLE formula
        r_k = distances[:, k]
        r_i = distances[:, 1:k + 1]  # Exclude self (index 0)

        m_estimates = []
        for i in range(n):
            if r_k[i] > 1e-10:
                log_ratios = np.log(r_k[i] / (r_i[i] + 1e-10))
                m_i = 1.0 / (np.mean(log_ratios) + 1e-10)
                m_estimates.append(m_i)

        return float(np.mean(m_estimates))

    # Select estimator
    if method == "two_nn":
        estimator = two_nn_faiss if use_faiss else two_nn_sklearn
    elif method == "mle":
        estimator = lambda X: (mle_faiss(X, k) if use_faiss else mle_sklearn(X, k))
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'two_nn' or 'mle'.")

    # Estimate for each dataset
    id_estimates = []
    for i, feat in enumerate(features):
        try:
            id_est = estimator(feat)
            id_estimates.append(id_est)
            print(f"Dataset {i}: Intrinsic dimension ≈ {id_est:.2f}")
        except Exception as e:
            print(f"Warning: Failed to estimate ID for dataset {i}: {e}")
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
    labels = []
    all_features = []
    for idx, f in enumerate(features):
        all_features.append(f)
        labels.append(np.full(f.shape[0], idx))
    X = np.vstack(all_features)
    y = np.concatenate(labels)
    dists = cdist(X, X, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    knn_indices = np.argsort(dists, axis=1)[:, :k]
    iso_list = [np.mean(y[n_idx] == y[i]) for i, n_idx in enumerate(knn_indices)]

    return iso_list if return_all else float(np.mean(iso_list))

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
        >>> print(f"Centroid Distance: {cd:.4f}")

    Notes:
        - Higher distance indicates clients have different mean features
        - Only captures first-order statistics (ignores variance/shape)
        - Fast to compute (single mean per client)
        - Less sensitive than MMD/Wasserstein which capture full distributions
        - Useful for quick heterogeneity assessment
    """
    features = _extract_features(dataset_list, **kwargs)
    centroids = [np.mean(f, axis=0) for f in features]
    return _compute_pairwise(centroids, lambda c1, c2: np.linalg.norm(c1 - c2), return_all)

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
    variances = [float(np.sum((f - np.mean(f, axis=0))**2) / (f.shape[0] - 1)) for f in features]
    return variances if return_all else np.mean(variances)

# --- Utils ---


def MMD_rbf_kernel(X, Y, sigma=None):
    """Calculate MMD^2 using RBF kernel."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    m, n = X.shape[0], Y.shape[0]

    if m < 2 or n < 2:
        raise ValueError("Each sample must have at least 2 points for unbiased MMD.")

    if sigma is None:
        combined = np.vstack([X, Y])
        dists = cdist(combined, combined, "euclidean")
        non_zero_dists = dists[dists > 0]
        sigma = np.median(non_zero_dists) if len(non_zero_dists) > 0 else 1.0

    def rbf(A, B):
        return np.exp(-cdist(A, B, "sqeuclidean") / (2 * sigma**2))

    K_XX, K_YY, K_XY = rbf(X, X), rbf(Y, Y), rbf(X, Y)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd_squared = K_XX.sum() / (m * (m - 1)) + K_YY.sum() / (n * (n - 1)) - 2 * K_XY.sum() / (m * n)

    return max(mmd_squared, 0.0)
