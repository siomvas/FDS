import numpy as np
from torchvision import models, transforms
import torch
import scipy
from scipy import stats
from scipy.spatial.distance import cdist
from PIL import Image
import os

from evaluation.inception_cache import InceptionV3EmbeddingCache

_INCEPTION_CACHE = None


def get_inception_cache(cache_dir="./.inception_cache"):
    global _INCEPTION_CACHE
    if _INCEPTION_CACHE is None:
        _INCEPTION_CACHE = InceptionV3EmbeddingCache(cache_dir=cache_dir)
    return _INCEPTION_CACHE


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, file_dataset, transforms=None):
        self.file_dataset = file_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.file_dataset)

    def __getitem__(self, i):
        img_path, label = self.file_dataset[i]['image'], self.file_dataset[i]['label']
        img = Image.open(img_path).convert("RGB")
        # label = torch.tensor(label) label is currently str in tiny imagenet, need to convert with tiny_imagenet_wnid_mapping.json
        # but for feature extraction we don't need label as tensor
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label


def _extract_features(dataset_list, model_name='InceptionV3', device=None,
                      batch_size=50, use_cache=True, cache_dir="./.inception_cache"):
    """Extract features from a list of datasets. Returns list of numpy arrays."""
    if use_cache and model_name == 'InceptionV3':
        cache = get_inception_cache(cache_dir)
        features_list = []
        for dataset in dataset_list:
            image_paths = [item["image"] for item in dataset]
            embeddings_dict = cache.get_embeddings(image_paths, batch_size)
            features = np.array([embeddings_dict[p] for p in image_paths])
            features_list.append(features)
        return features_list

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.hub.set_dir('~/archive/.cache/torch/hub')

    if model_name == 'InceptionV3':
        os.environ['https_proxy'] = 'http://proxy.example.org:3128'
        model = models.inception_v3(pretrained=True, transform_input=False)
        os.environ.pop('https_proxy', None)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    features_list = []
    for dataset in dataset_list:
        img_dataset = ImagePathDataset(dataset, transforms=transform)
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


def FID_distance(dataset_list, model_name='InceptionV3', device=None,
                 return_all=False, batch_size=50, use_cache=True,
                 cache_dir="./.inception_cache"):
    """Calculate average pairwise Frechet Inception Distance of N datasets."""
    if len(dataset_list) < 2:
        raise ValueError("At least two datasets are required.")

    features_list = _extract_features(
        dataset_list, model_name, device, batch_size, use_cache, cache_dir
    )

    def compute_statistics(activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

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

    stats_list = [compute_statistics(f) for f in features_list]
    return _compute_pairwise(stats_list, compute_fid, return_all)


def KL_divergence(dataset_list, num_classes=None, return_all=False):
    """Calculate average pairwise KL Divergence between label distributions."""
    if any(len(d) == 0 for d in dataset_list):
        raise ValueError("Datasets must not be empty.")

    labels_list = [np.array([item['label'] for item in d]) for d in dataset_list]

    if num_classes is None:
        num_classes = max(l.max() for l in labels_list) + 1

    epsilon = 1e-10
    distributions = []
    for labels in labels_list:
        counts = np.bincount(labels, minlength=num_classes) + epsilon
        distributions.append(counts / counts.sum())

    def compute_kl(p, q):
        return np.sum(p * np.log(p / q))

    return _compute_pairwise(distributions, compute_kl, return_all)


def KS_label_test(dataset_list, return_all=False):
    """Perform pairwise Kolmogorov-Smirnov tests on label distributions."""
    labels_list = [np.array([item['label'] for item in d]) for d in dataset_list]

    def compute_ks(l1, l2):
        statistic, pvalue = stats.ks_2samp(l1, l2)
        return {'statistic': statistic, 'pvalue': pvalue}

    n = len(labels_list)
    if n < 2:
        raise ValueError("At least two datasets required.")

    results = []
    for i in range(n):
        for j in range(i + 1, n):
            results.append(compute_ks(labels_list[i], labels_list[j]))

    if return_all:
        return results

    return {
        'statistic': np.mean([r['statistic'] for r in results]),
        'pvalue': np.mean([r['pvalue'] for r in results])
    }


def MMD_rbf_kernel(X, Y, sigma=None):
    """Calculate MMD^2 using RBF kernel."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    m, n = X.shape[0], Y.shape[0]

    if m < 2 or n < 2:
        raise ValueError("Each sample must have at least 2 points for unbiased MMD.")

    if sigma is None:
        combined = np.vstack([X, Y])
        dists = cdist(combined, combined, 'euclidean')
        non_zero_dists = dists[dists > 0]
        sigma = np.median(non_zero_dists) if len(non_zero_dists) > 0 else 1.0

    def rbf_kernel(A, B):
        return np.exp(-cdist(A, B, 'sqeuclidean') / (2 * sigma**2))

    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)

    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd_squared = (K_XX.sum() / (m * (m - 1)) +
                   K_YY.sum() / (n * (n - 1)) -
                   2 * K_XY.sum() / (m * n))

    return max(mmd_squared, 0.0)


def MMD_label_distributions(dataset_list, num_classes=None, return_all=False):
    """Calculate average pairwise MMD between label distributions."""
    labels_list = [np.array([item['label'] for item in d]) for d in dataset_list]

    if num_classes is None:
        num_classes = max(l.max() for l in labels_list) + 1

    means = [np.eye(num_classes)[labels].mean(axis=0) for labels in labels_list]

    def compute_mmd(m1, m2):
        return np.sqrt(np.sum((m1 - m2) ** 2))

    return _compute_pairwise(means, compute_mmd, return_all)


def MMD_features(dataset_list, model_name='InceptionV3', device=None,
                 batch_size=50, sigma=None, return_all=False,
                 use_cache=True, cache_dir="./.inception_cache"):
    """Calculate average pairwise MMD between feature distributions."""
    features_list = _extract_features(
        dataset_list, model_name, device, batch_size, use_cache, cache_dir
    )

    def compute_mmd(f1, f2):
        return MMD_rbf_kernel(f1, f2, sigma=sigma)

    return _compute_pairwise(features_list, compute_mmd, return_all)


def mahalanobis_distance_labels(dataset_list, num_classes=None, return_all=False):
    """Calculate average pairwise Mahalanobis distance between label distributions."""
    labels_list = [np.array([item['label'] for item in d]) for d in dataset_list]

    if num_classes is None:
        num_classes = max(l.max() for l in labels_list) + 1

    distributions = []
    for labels in labels_list:
        counts = np.bincount(labels, minlength=num_classes)
        distributions.append(counts / counts.sum())

    def compute_mahalanobis(p, q):
        avg_dist = (p + q) / 2
        cov = np.diag(avg_dist) - np.outer(avg_dist, avg_dist)
        cov += np.eye(cov.shape[0]) * 1e-6

        diff = p - q
        try:
            inv_cov = np.linalg.inv(cov)
            return np.sqrt(diff @ inv_cov @ diff.T)
        except np.linalg.LinAlgError:
            return np.linalg.norm(diff)

    return _compute_pairwise(distributions, compute_mahalanobis, return_all)


def wasserstein_distance_features(dataset_list, model_name='InceptionV3', device=None,
                                  batch_size=50, return_all=False,
                                  use_cache=True, cache_dir="./.inception_cache"):
    """Calculate average pairwise Wasserstein distance between feature distributions."""
    features_list = _extract_features(
        dataset_list, model_name, device, batch_size, use_cache, cache_dir
    )

    def compute_wasserstein(f1, f2):
        n_features = f1.shape[1]
        dists = [stats.wasserstein_distance(f1[:, i], f2[:, i]) for i in range(n_features)]
        return np.mean(dists)

    return _compute_pairwise(features_list, compute_wasserstein, return_all)


def cramer_von_mises_features(dataset_list, model_name='InceptionV3', device=None,
                              batch_size=50, return_all=False,
                              use_cache=True, cache_dir="./.inception_cache"):
    """Calculate average pairwise Cramér-von Mises statistic between feature distributions."""
    features_list = _extract_features(
        dataset_list, model_name, device, batch_size, use_cache, cache_dir
    )

    def compute_cvm(f1, f2):
        n_features = f1.shape[1]
        cvm_stats = [stats.cramervonmises_2samp(f1[:, i], f2[:, i]).statistic
                     for i in range(n_features)]
        return np.mean(cvm_stats)

    return _compute_pairwise(features_list, compute_cvm, return_all)


def KS_multivariate_test(dataset_list, model_name='InceptionV3', device=None,
                         batch_size=50, alpha=0.05, return_all=False,
                         use_cache=True, cache_dir="./.inception_cache"):
    """Perform pairwise multivariate KS tests with Bonferroni correction."""
    features_list = _extract_features(
        dataset_list, model_name, device, batch_size, use_cache, cache_dir
    )

    def compute_ks(f1, f2):
        n_features = f1.shape[1]
        pvalues = np.array([stats.ks_2samp(f1[:, i], f2[:, i])[1] for i in range(n_features)])
        min_pvalue = pvalues.min()
        corrected_alpha = alpha / n_features

        return {
            'reject_null': min_pvalue < corrected_alpha,
            'min_pvalue': float(min_pvalue),
            'corrected_alpha': corrected_alpha,
            'n_tests': n_features,
            'significant_dims': int(np.sum(pvalues < corrected_alpha))
        }

    n = len(features_list)
    if n < 2:
        raise ValueError("At least two datasets required.")

    results = []
    for i in range(n):
        for j in range(i + 1, n):
            results.append(compute_ks(features_list[i], features_list[j]))

    if return_all:
        return results

    return {
        'reject_null': any(r['reject_null'] for r in results),
        'min_pvalue': min(r['min_pvalue'] for r in results),
        'corrected_alpha': results[0]['corrected_alpha'],
        'n_tests': results[0]['n_tests'],
        'significant_dims': np.mean([r['significant_dims'] for r in results])
    }


def vendi_score_from_kernel(kernel_matrix, q=1):
    from vendi_score import vendi
    return float(vendi.score_K(kernel_matrix, q=q))


def vendi_score_features(dataset_list, model_name='InceptionV3', device=None,
                         batch_size=200, q=1, normalize=True, return_all=False,
                         use_cache=True, cache_dir="./.inception_cache"):
    """Calculate Vendi score for each dataset and return average (or all scores)."""
    from vendi_score import vendi

    features_list = _extract_features(
        dataset_list, model_name, device, batch_size, use_cache, cache_dir
    )

    scores = [float(vendi.score_dual(f, q=q, normalize=normalize)) for f in features_list]

    return scores if return_all else float(np.mean(scores))