import numpy as np
from torchvision import models, transforms
# import alibi_detect as ad
import torch
import scipy
from scipy import stats
from scipy.spatial.distance import cdist
from PIL import Image
from collections import Counter
import os

# We want:
# Category 1: Operations on label distributions
# - KL Divergence
# - Kolmogorov-Smirnov two-sample univariate test
# - MMD (Maximum Mean Discrepancy) 
# - Mahalanobis distance

# Category 2: Operations on feature distributions (features extracted from pre-trained models, e.g., ResNet)
# - Wasserstein distance
# - Cramér-von Mises criterion
# - Frechet Inception Distance (FID)
# - MMD (Maximum Mean Discrepancy)
# - KS Multivariate test (K univariate tests combined)

# Category 3: Operations on embeddings
# - Similar to Category 2, but specifically for embeddings from models like DINO, UNILM, etc.

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, file_dataset, transforms=None):
        self.file_dataset = file_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.file_dataset)

    def __getitem__(self, i):
        img_path, label = self.file_dataset[i]['image'], self.file_dataset[i]['label']
        img = Image.open(img_path).convert("RGB")
        label = torch.tensor(label)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

def FID_distance(dataset_list, model_name='InceptionV3', device=None, return_all=False, batch_size=50):

    """
    Calculate the average pairwise Frechet Inception Distance (FID) of N datasets.
    
    Args:
        dataset_list: List of datasets (each dataset is a list of image file paths)
        model_name: Name of the model to use for feature extraction
        device: Device to run model on ('cuda' or 'cpu')
        batch_size: Batch size for feature extraction
        
    Returns:
        FID score as float
    """
    
    if len(dataset_list) < 2:
        raise ValueError("At least two datasets are required to compute FID.")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up model directory
    torch.hub.set_dir('~/archive/.cache/torch/hub')

    if model_name == 'InceptionV3':
        os.environ['https_proxy'] = 'http://proxy.example.org:3128'
        model = models.inception_v3(pretrained=True, transform_input=False)
        os.environ.pop('https_proxy', None)
    else:
        raise ValueError("Unsupported model for FID calculation")

    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def get_activations(dataset):
        activations = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (image, label) in dataloader:
                image = image.to(device)
                preds = model(image)
                activations.append(preds.cpu().numpy())
        return np.concatenate(activations, axis=0)
    
    def compute_statistics(activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def compute_fid_from_stats(stats1, stats2):

        mu1, sigma1 = stats1
        mu2, sigma2 = stats2
    
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (mu1.shape == mu2.shape), "Training and test mean vectors have different lengths"
        assert (sigma1.shape == sigma2.shape), "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            print("fid calculation produces singular product; adding a small offset to the diagonal of covariances")
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return fid
    
    n_datasets = len(dataset_list)
    stats_list = []
    for i, dataset in enumerate(dataset_list):
        img_dataset = ImagePathDataset(dataset, transforms=transform)
        activations = get_activations(img_dataset)
        stats = compute_statistics(activations)
        stats_list.append(stats)
        del activations  # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    fid_scores = []
    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            fid = compute_fid_from_stats(stats_list[i], stats_list[j])
            fid_scores.append(fid)
    
    # structure of fidscores is [fid_0_1, fid_0_2, ..., fid_1_2, ...]
    
    return np.mean(fid_scores) if not return_all else fid_scores
    


def KL_divergence(dataset1, dataset2, num_classes=None):
    """
    Calculate KL Divergence between label distributions of two datasets.

    Args:
        dataset1: List of dicts with 'label' field
        dataset2: List of dicts with 'label' field
        num_classes: Total number of classes (if None, inferred from data)

    Returns:
        KL divergence D_KL(P||Q) where P is dataset1, Q is dataset2
    """
    # raise ValueError if datasets are empty
    if len(dataset1) == 0 or len(dataset2) == 0:
        raise ValueError("Datasets must not be empty.")

    labels1 = np.array([item['label'] for item in dataset1])
    labels2 = np.array([item['label'] for item in dataset2])

    if num_classes is None:
        num_classes = max(labels1.max(), labels2.max()) + 1

    # Compute label distributions with smoothing to avoid log(0)
    epsilon = 1e-10
    counts1 = np.bincount(labels1, minlength=num_classes) + epsilon
    counts2 = np.bincount(labels2, minlength=num_classes) + epsilon

    p = counts1 / counts1.sum()
    q = counts2 / counts2.sum()

    kl_div = np.sum(p * np.log(p / q))
    return kl_div


def KS_label_test(dataset1, dataset2):
    """
    Perform Kolmogorov-Smirnov test on label distributions.

    Args:
        dataset1: List of dicts with 'label' field
        dataset2: List of dicts with 'label' field

    Returns:
        dict with 'statistic' and 'pvalue'
    """
    labels1 = np.array([item['label'] for item in dataset1])
    labels2 = np.array([item['label'] for item in dataset2])

    # KS test for discrete distributions
    statistic, pvalue = stats.ks_2samp(labels1, labels2)

    return {'statistic': statistic, 'pvalue': pvalue}


def MMD_rbf_kernel(X, Y, sigma=None):
    """
    Calculate Maximum Mean Discrepancy (MMD) using RBF kernel.

    Args:
        X: First sample (n_samples_X, n_features)
        Y: Second sample (n_samples_Y, n_features)
        sigma: RBF kernel bandwidth (if None, uses median heuristic)

    Returns:
        MMD^2 statistic
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    m, n = X.shape[0], Y.shape[0]

    # Compute median heuristic for sigma if not provided
    if sigma is None:
        # Use median of pairwise distances
        combined = np.vstack([X, Y])
        dists = cdist(combined, combined, 'euclidean')
        non_zero_dists = dists[dists > 0]
        if len(non_zero_dists) == 0:
            sigma = 1.0
        else:
            sigma = np.median(non_zero_dists)

    def rbf_kernel(A, B, sigma):
        """Compute RBF kernel matrix between A and B"""
        dists = cdist(A, B, 'sqeuclidean')
        return np.exp(-dists / (2 * sigma**2))

    # Compute kernel matrices
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)

    # MMD^2 unbiased estimate
    # Remove diagonal from K_XX and K_YY
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd_squared = (K_XX.sum() / (m * (m - 1)) +
                   K_YY.sum() / (n * (n - 1)) -
                   2 * K_XY.sum() / (m * n))

    return max(mmd_squared, 0.0)


def MMD_label_distributions(dataset1, dataset2, num_classes=None):
    """
    Calculate MMD between label distributions (treating distributions as vectors).

    Args:
        dataset1: List of dicts with 'label' field
        dataset2: List of dicts with 'label' field
        num_classes: Total number of classes

    Returns:
        MMD between label distributions
    """
    labels1 = np.array([item['label'] for item in dataset1])
    labels2 = np.array([item['label'] for item in dataset2])

    if num_classes is None:
        num_classes = max(labels1.max(), labels2.max()) + 1
    
    onehot1 = np.eye(num_classes)[labels1]
    onehot2 = np.eye(num_classes)[labels2]

    mean1 = onehot1.mean(axis=0).reshape(1, -1)
    mean2 = onehot2.mean(axis=0).reshape(1, -1)
    mmd_squared = np.sum((mean1 - mean2) ** 2)
    return np.sqrt(mmd_squared)

    # Convert to distribution vectors
    # counts1 = np.bincount(labels1, minlength=num_classes)
    # counts2 = np.bincount(labels2, minlength=num_classes)

    # p = counts1 / counts1.sum()
    # q = counts2 / counts2.sum()

    # # Reshape to 2D for MMD
    # p = p.reshape(1, -1)
    # q = q.reshape(1, -1)

    # # For single distributions, MMD simplifies to distance between means
    # return np.linalg.norm(p - q)


def MMD_features(dataset1, dataset2, model_name='InceptionV3', device=None, batch_size=50, sigma=None):
    """
    Calculate MMD between feature distributions of two datasets.

    Args:
        dataset1: List of dicts with 'image' and 'label' fields
        dataset2: List of dicts with 'image' and 'label' fields
        model_name: Name of the model for feature extraction
        device: Device to run model on
        batch_size: Batch size for feature extraction
        sigma: RBF kernel bandwidth (if None, uses median heuristic)

    Returns:
        MMD^2 statistic between feature distributions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up model directory
    torch.hub.set_dir('~/archive/.cache/torch/hub')

    # Load model
    if model_name == 'InceptionV3':
        os.environ['https_proxy'] = 'http://proxy.example.org:3128'
        model = models.inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()
        os.environ.pop('https_proxy', None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def extract_features(dataset):
        img_dataset = ImagePathDataset(dataset, transforms=transform)
        dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, shuffle=False)
        features = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                feats = model(images)
                features.append(feats.cpu().numpy())
        return np.concatenate(features, axis=0)

    features1 = extract_features(dataset1)
    features2 = extract_features(dataset2)

    mmd_squared = MMD_rbf_kernel(features1, features2, sigma=sigma)

    # Cleanup
    del features1, features2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mmd_squared


def mahalanobis_distance_labels(dataset1, dataset2, num_classes=None):
    """
    Calculate Mahalanobis distance between label distributions.

    Args:
        dataset1: List of dicts with 'label' field
        dataset2: List of dicts with 'label' field
        num_classes: Total number of classes

    Returns:
        Mahalanobis distance
    """
    labels1 = np.array([item['label'] for item in dataset1])
    labels2 = np.array([item['label'] for item in dataset2])

    if num_classes is None:
        num_classes = max(labels1.max(), labels2.max()) + 1

    # Convert to distribution vectors
    counts1 = np.bincount(labels1, minlength=num_classes)
    counts2 = np.bincount(labels2, minlength=num_classes)

    p = counts1 / counts1.sum()
    q = counts2 / counts2.sum()

    # Estimate covariance from both distributions
    # combined = np.vstack([p, q])
    # cov = np.cov(combined.T)
    avg_dist = (p + q) / 2
    cov = np.diag(avg_dist) - np.outer(avg_dist, avg_dist)
    # Add small regularization to avoid singularity
    cov += np.eye(cov.shape[0]) * 1e-6

    # Compute Mahalanobis distance
    diff = p - q
    try:
        inv_cov = np.linalg.inv(cov)
        mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
    except np.linalg.LinAlgError:
        mahal_dist = np.linalg.norm(diff)  # Fallback to Euclidean if singular

    return mahal_dist

def wasserstein_distance_features(dataset1, dataset2, model_name='InceptionV3', device=None, batch_size=50):
    """
    Calculate Wasserstein distance (Earth Mover's Distance) between feature distributions.

    Args:
        dataset1: List of dicts with 'image' and 'label' fields
        dataset2: List of dicts with 'image' and 'label' fields
        model_name: Name of the model for feature extraction
        device: Device to run model on
        batch_size: Batch size for feature extraction

    Returns:
        Wasserstein distance (averaged across feature dimensions)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up model directory
    torch.hub.set_dir('~/archive/.cache/torch/hub')

    # Load model
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

    def extract_features(dataset):
        img_dataset = ImagePathDataset(dataset, transforms=transform)
        dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, shuffle=False)
        features = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                feats = model(images)
                features.append(feats.cpu().numpy())
        return np.concatenate(features, axis=0)

    features1 = extract_features(dataset1)
    features2 = extract_features(dataset2)

    # Compute Wasserstein distance for each feature dimension and average
    n_features = features1.shape[1]
    wasserstein_dists = []

    for i in range(n_features):
        wd = stats.wasserstein_distance(features1[:, i], features2[:, i])
        wasserstein_dists.append(wd)

    avg_wasserstein = np.mean(wasserstein_dists)

    # Cleanup
    del features1, features2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_wasserstein


def cramer_von_mises_features(dataset1, dataset2, model_name='InceptionV3', device=None, batch_size=50):
    """
    Calculate Cramér-von Mises criterion between feature distributions.

    Args:
        dataset1: List of dicts with 'image' and 'label' fields
        dataset2: List of dicts with 'image' and 'label' fields
        model_name: Name of the model for feature extraction
        device: Device to run model on
        batch_size: Batch size for feature extraction

    Returns:
        Cramér-von Mises statistic (averaged across feature dimensions)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up model directory
    torch.hub.set_dir('~/archive/.cache/torch/hub')

    # Load model
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

    def extract_features(dataset):
        img_dataset = ImagePathDataset(dataset, transforms=transform)
        dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, shuffle=False)
        features = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                feats = model(images)
                features.append(feats.cpu().numpy())
        return np.concatenate(features, axis=0)

    features1 = extract_features(dataset1)
    features2 = extract_features(dataset2)

    # Compute Cramér-von Mises test for each feature dimension and average
    n_features = features1.shape[1]
    cvm_statistics = []

    for i in range(n_features):
        result = stats.cramervonmises_2samp(features1[:, i], features2[:, i])
        cvm_statistics.append(result.statistic)

    avg_cvm = np.mean(cvm_statistics)

    # Cleanup
    del features1, features2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_cvm


def KS_multivariate_test(dataset1, dataset2, model_name='InceptionV3', device=None, batch_size=50, alpha=0.05):
    """
    Perform multivariate KS test by combining multiple univariate KS tests
    with Bonferroni correction.

    Args:
        dataset1: List of dicts with 'image' and 'label' fields
        dataset2: List of dicts with 'image' and 'label' fields
        model_name: Name of the model for feature extraction
        device: Device to run model on
        batch_size: Batch size for feature extraction
        alpha: Significance level

    Returns:
        dict with 'reject_null' (bool), 'min_pvalue', 'corrected_alpha', and 'n_tests'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up model directory
    torch.hub.set_dir('~/archive/.cache/torch/hub')

    # Load model
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

    def extract_features(dataset):
        img_dataset = ImagePathDataset(dataset, transforms=transform)
        dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, shuffle=False)
        features = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                feats = model(images)
                features.append(feats.cpu().numpy())
        return np.concatenate(features, axis=0)

    features1 = extract_features(dataset1)
    features2 = extract_features(dataset2)

    # Perform KS test for each feature dimension
    n_features = features1.shape[1]
    pvalues = []

    for i in range(n_features):
        statistic, pvalue = stats.ks_2samp(features1[:, i], features2[:, i])
        pvalues.append(pvalue)

    pvalues = np.array(pvalues)
    min_pvalue = pvalues.min()

    # Bonferroni correction: reject null hypothesis if minimum p-value < alpha/K
    corrected_alpha = alpha / n_features
    reject_null = min_pvalue < corrected_alpha

    # Cleanup
    del features1, features2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'reject_null': reject_null,
        'min_pvalue': float(min_pvalue),
        'corrected_alpha': corrected_alpha,
        'n_tests': n_features,
        'significant_dims': int(np.sum(pvalues < corrected_alpha))
    }
