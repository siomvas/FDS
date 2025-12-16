import numpy as np
from torchvision import models, transforms
import torch
import scipy
from scipy import stats
from scipy.spatial.distance import cdist
from PIL import Image
import os
from typing import List, Dict, Optional, Any
from vendi_score import vendi

from embedding.embedding_cache import DatasetEmbeddingCache
from data_loading.dataset import ImageDataset

_GLOBAL_CACHE: Optional[DatasetEmbeddingCache] = None

"""
List of available metrics

- Label Metrics:
    - Jensen-Shannon Divergence (JSD)
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


def get_global_cache(cache_dir: str = "./.embedding_cache") -> DatasetEmbeddingCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is not None and str(_GLOBAL_CACHE.cache_dir) != str(cache_dir):
        _GLOBAL_CACHE = None
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = DatasetEmbeddingCache(cache_dir=cache_dir)
    return _GLOBAL_CACHE


def _extract_features(
    dataset_list: List[List[Dict[str, Any]]],
    model_name: str = "inception_v3",
    device=None,
    batch_size: int = 200,
    use_cache: bool = True,
    cache_dir: str = "./.embedding_cache",
    dataset_name: Optional[str] = None,
    base_path: Optional[str] = None,
    proxy: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Extract features from a list of datasets. Returns list of numpy arrays.

    If dataset_name is provided and cache exists, loads from cache.
    Otherwise computes embeddings on-the-fly.
    """
    if use_cache and dataset_name and base_path:
        cache = get_global_cache(cache_dir)
        if cache.exists(dataset_name):
            try:
                features_list = []
                for dataset in dataset_list:
                    image_paths = [str(item["image"]) for item in dataset]  # cast to str for Path compatibility
                    features = cache.get(dataset_name, image_paths, base_path)
                    features_list.append(features)
                return features_list
            except Exception as e:
                print(f"Warning: Could not load from cache: {e}. Recomputing embeddings.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))

    if model_name == "inception_v3":
        if proxy:
            os.environ["https_proxy"] = proxy
        model = models.inception_v3(weights="IMAGENET1K_V1", transform_input=False)
        model.fc = torch.nn.Identity()
        os.environ.pop("https_proxy", None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features_list = []
    for dataset in dataset_list:
        img_dataset = ImageDataset(dataset, transform=transform)
        dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, shuffle=False)
        features = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                feats = model(images)
                features.append(feats.cpu().numpy())
        features_list.append(np.concatenate(features, axis=0))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return features_list


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
    labels = [np.array([item["label"] for item in d]) for d in dataset_list]
    if num_classes is None: num_classes = max(l.max() for l in labels) + 1
    dists = []
    for l in labels:
        c = np.bincount(l, minlength=num_classes) + 1e-10
        dists.append(c / c.sum())
    return _compute_pairwise(dists, lambda p, q: 0.5 * np.sum(np.abs(p - q)), return_all)


def JSD_client_to_global(dataset_list, num_classes=None, return_all=False):
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
    if len(dataset_list) < 2:
        raise ValueError("At least two datasets are required.")

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
    features = _extract_features(dataset_list, **kwargs)
    return _compute_pairwise(features, lambda f1, f2: MMD_rbf_kernel(f1, f2, sigma), return_all)

def energy_distance_features(dataset_list, return_all=False, **kwargs):
    features = _extract_features(dataset_list, **kwargs)
    def energy_distance(X, Y):
        dXX = cdist(X, X, 'euclidean')
        dYY = cdist(Y, Y, 'euclidean')
        dXY = cdist(X, Y, 'euclidean')
        return 2 * np.mean(dXY) - np.mean(dXX) - np.mean(dYY)
    return _compute_pairwise(features, energy_distance, return_all)


def sliced_wasserstein_distance(dataset_list, return_all=False, **kwargs):
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


def KS_multivariate_test(dataset_list, alpha=0.05, return_all=False, **kwargs):
    features = _extract_features(dataset_list, **kwargs)

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
    features = _extract_features(dataset_list, **kwargs)
    scores = [float(vendi.score_dual(f, q=q, normalize=normalize)) for f in features]
    return scores if return_all else float(np.mean(scores))


def intrinsic_dimensionality(
        dataset_list: List[List[Dict[str, Any]]],
        target_size: int = 64,
        method: str = "mle",
        k: int = 10,
        use_faiss: bool = True,
        faiss_approx: bool = True,
        return_all: bool = False,
        **kwargs
) -> List[float]:
    features = _extract_features(dataset_list, **kwargs)
    
    def two_nn_faiss(X):
        """GPU-accelerated TWO-NN using FAISS."""
        import faiss
        n, d = X.shape
        if faiss_approx and n > 10000:
            n_clusters = min(int(np.sqrt(n)), 256)
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, n_clusters)
            index.train(X)
            index.add(X)
            index.nprobe = min(16, n_clusters)
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(X)
        distances, _ = gpu_index.search(X, 3)
        r1 = np.sqrt(distances[:, 1])
        r2 = np.sqrt(distances[:, 2])
        valid = r1 > 1e-10
        mu = r2[valid] / r1[valid]
        
        return len(mu) / np.sum(np.log(mu))
    
    def two_nn(X):
    
    def mle_faiss(X):
        pass

    def mle(X):
        pass

def knn_separability(dataset_list, k=5, return_all=False, **kwargs):
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
    features = _extract_features(dataset_list, **kwargs)
    centroids = [np.mean(f, axis=0) for f in features]
    return _compute_pairwise(centroids, lambda c1, c2: np.linalg.norm(c1 - c2), return_all)

def feature_variance(dataset_list, return_all=False, **kwargs):
    features = _extract_features(dataset_list, **kwargs)
    variances = [float(np.sum(f - np.mean(f, axis=0))**2) / (f.shape[0] - 1) for f in features]
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
