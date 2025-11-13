"""
Validation script to test evaluation metrics.

This script compares Random (IID) splits vs high-heterogeneity Dirichlet splits
to validate that our metrics correctly detect distribution shifts.

Expected behavior:
- Random splits should show LOW heterogeneity (similar distributions)
- Dirichlet splits (low alpha) should show HIGH heterogeneity (different distributions)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torchvision import datasets, transforms
from data_partitioning.random_split.random_split import RandomSplitter
from data_partitioning.dirichlet_split.dirichlet_splitter import DirichletSplitter
from evaluation import metrics
import argparse
import json
from datetime import datetime


def prepare_dataset_for_metrics(dataset, indices, include_images=False):
    """
    Convert dataset + indices to format expected by metrics.

    Args:
        dataset: PyTorch dataset with images and labels
        indices: List of indices for this client
        include_images: If True, save image paths (needed for feature-based metrics)

    Returns:
        List of dicts with 'label' and optionally 'image_path' fields
    """
    data_list = []

    for idx in indices:
        item = {'label': int(dataset[idx][1])}

        if include_images:
            # For validation, we can use dummy paths or actual images
            # For now, we'll add image data directly for feature extraction
            item['image'] = dataset[idx][0]

        data_list.append(item)

    return data_list


def create_image_path_dataset(dataset, indices, temp_dir='/tmp/validation_images'):
    """
    Create temporary image files and return dataset format for feature metrics.

    Args:
        dataset: PyTorch dataset
        indices: List of indices
        temp_dir: Directory to save temporary images

    Returns:
        List of dicts with 'image_path' and 'label' fields
    """
    os.makedirs(temp_dir, exist_ok=True)
    data_list = []

    for i, idx in enumerate(indices[:100]):  # Limit to 100 images for speed
        img, label = dataset[idx]

        # Save image temporarily
        img_path = os.path.join(temp_dir, f'img_{i}.png')
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        img.save(img_path)

        data_list.append({
            'image_path': img_path,
            'label': int(label)
        })

    return data_list


def run_label_distribution_metrics(dataset1, dataset2, num_classes):
    """Run all label distribution metrics."""
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION METRICS")
    print("="*60)

    results = {}

    # KL Divergence
    try:
        kl_div = metrics.KL_divergence(dataset1, dataset2, num_classes=num_classes)
        results['KL_divergence'] = float(kl_div)
        print(f"✓ KL Divergence: {kl_div:.6f}")
    except Exception as e:
        print(f"✗ KL Divergence failed: {e}")
        results['KL_divergence'] = None

    # KS Label Test
    try:
        ks_result = metrics.KS_label_test(dataset1, dataset2)
        results['KS_label_test'] = ks_result
        print(f"✓ KS Test: statistic={ks_result['statistic']:.6f}, p-value={ks_result['pvalue']:.6f}")
    except Exception as e:
        print(f"✗ KS Label Test failed: {e}")
        results['KS_label_test'] = None

    # MMD Label Distributions
    try:
        mmd_labels = metrics.MMD_label_distributions(dataset1, dataset2, num_classes=num_classes)
        results['MMD_label_distributions'] = float(mmd_labels)
        print(f"✓ MMD (Labels): {mmd_labels:.6f}")
    except Exception as e:
        print(f"✗ MMD Label Distributions failed: {e}")
        results['MMD_label_distributions'] = None

    # Mahalanobis Distance
    try:
        mahal_dist = metrics.mahalanobis_distance_labels(dataset1, dataset2, num_classes=num_classes)
        results['mahalanobis_distance'] = float(mahal_dist)
        print(f"✓ Mahalanobis Distance: {mahal_dist:.6f}")
    except Exception as e:
        print(f"✗ Mahalanobis Distance failed: {e}")
        results['mahalanobis_distance'] = None

    return results


def run_feature_distribution_metrics(dataset1, dataset2, device, batch_size=32):
    """Run all feature distribution metrics."""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION METRICS")
    print("="*60)
    print("(Using subset of 100 images per client for speed)")

    results = {}

    # FID Distance
    try:
        print("Computing FID distance...")
        fid_score = metrics.FID_distance(
            [dataset1, dataset2],
            device=device,
            batch_size=batch_size
        )
        results['FID_distance'] = float(fid_score)
        print(f"✓ FID Distance: {fid_score:.6f}")
    except Exception as e:
        print(f"✗ FID Distance failed: {e}")
        results['FID_distance'] = None

    # MMD Features
    try:
        print("Computing MMD on features...")
        mmd_feats = metrics.MMD_features(
            dataset1, dataset2,
            device=device,
            batch_size=batch_size
        )
        results['MMD_features'] = float(mmd_feats)
        print(f"✓ MMD (Features): {mmd_feats:.6f}")
    except Exception as e:
        print(f"✗ MMD Features failed: {e}")
        results['MMD_features'] = None

    # Wasserstein Distance
    try:
        print("Computing Wasserstein distance...")
        wasserstein = metrics.wasserstein_distance_features(
            dataset1, dataset2,
            device=device,
            batch_size=batch_size
        )
        results['wasserstein_distance'] = float(wasserstein)
        print(f"✓ Wasserstein Distance: {wasserstein:.6f}")
    except Exception as e:
        print(f"✗ Wasserstein Distance failed: {e}")
        results['wasserstein_distance'] = None

    # Cramér-von Mises
    try:
        print("Computing Cramér-von Mises criterion...")
        cvm = metrics.cramer_von_mises_features(
            dataset1, dataset2,
            device=device,
            batch_size=batch_size
        )
        results['cramer_von_mises'] = float(cvm)
        print(f"✓ Cramér-von Mises: {cvm:.6f}")
    except Exception as e:
        print(f"✗ Cramér-von Mises failed: {e}")
        results['cramer_von_mises'] = None

    # KS Multivariate Test
    try:
        print("Computing KS Multivariate test...")
        ks_multi = metrics.KS_multivariate_test(
            dataset1, dataset2,
            device=device,
            batch_size=batch_size,
            alpha=0.05
        )
        results['KS_multivariate_test'] = ks_multi
        print(f"✓ KS Multivariate: reject_null={ks_multi['reject_null']}, "
              f"min_pvalue={ks_multi['min_pvalue']:.6e}, "
              f"significant_dims={ks_multi['significant_dims']}/{ks_multi['n_tests']}")
    except Exception as e:
        print(f"✗ KS Multivariate Test failed: {e}")
        results['KS_multivariate_test'] = None

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate evaluation metrics')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100', 'FashionMNIST'],
                        help='Dataset to use for validation')
    parser.add_argument('--num_clients', type=int, default=2,
                        help='Number of clients to split into')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1,
                        help='Dirichlet alpha parameter (lower = more heterogeneous)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for dataset')
    parser.add_argument('--skip_feature_metrics', action='store_true',
                        help='Skip feature-based metrics (faster)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*60)
    print("METRIC VALIDATION SCRIPT")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Dirichlet alpha: {args.dirichlet_alpha}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("="*60)

    # Load dataset
    if args.dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        dataset = datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        num_classes = 100
    elif args.dataset == 'FashionMNIST':
        dataset = datasets.FashionMNIST(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Grayscale(3),  # Convert to 3 channels for InceptionV3
                transforms.ToTensor()
            ])
        )
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"\n✓ Loaded {args.dataset}: {len(dataset)} samples, {num_classes} classes")

    # Create Random split (IID)
    print("\n" + "="*60)
    print("CREATING RANDOM (IID) SPLIT")
    print("="*60)
    random_splitter = RandomSplitter(dataset, num_clients=args.num_clients, seed=args.seed)
    random_splitter.split()
    random_indices = random_splitter.get_client_data_indices()
    print(f"✓ Random split created: {[len(indices) for indices in random_indices.values()]} samples per client")

    # Create Dirichlet split (non-IID)
    print("\n" + "="*60)
    print(f"CREATING DIRICHLET SPLIT (alpha={args.dirichlet_alpha})")
    print("="*60)
    dirichlet_splitter = DirichletSplitter(
        dataset,
        num_clients=args.num_clients,
        heterogeneity=args.dirichlet_alpha,
        seed=args.seed
    )
    dirichlet_splitter.split()
    dirichlet_indices = dirichlet_splitter.get_client_data_indices()
    print(f"✓ Dirichlet split created: {[len(indices) for indices in dirichlet_indices.values()]} samples per client")

    # Show label distributions
    print("\n" + "="*60)
    print("LABEL DISTRIBUTIONS")
    print("="*60)

    for split_name, indices_dict in [('Random', random_indices), ('Dirichlet', dirichlet_indices)]:
        print(f"\n{split_name} Split:")
        for client_id, indices in indices_dict.items():
            labels = [dataset[i][1] for i in indices]
            label_counts = np.bincount(labels, minlength=num_classes)
            label_probs = label_counts / label_counts.sum()
            print(f"  Client {client_id}: {label_counts.tolist()}")
            print(f"    Probabilities: {[f'{p:.3f}' for p in label_probs]}")

    # Prepare datasets for metrics (compare client 0 vs client 1)
    print("\n" + "="*60)
    print("COMPARING CLIENT 0 VS CLIENT 1")
    print("="*60)

    random_client0 = prepare_dataset_for_metrics(dataset, random_indices[0], include_images=False)
    random_client1 = prepare_dataset_for_metrics(dataset, random_indices[1], include_images=False)

    dirichlet_client0 = prepare_dataset_for_metrics(dataset, dirichlet_indices[0], include_images=False)
    dirichlet_client1 = prepare_dataset_for_metrics(dataset, dirichlet_indices[1], include_images=False)

    # Run label distribution metrics
    print("\n" + "#"*60)
    print("# RANDOM SPLIT - LABEL METRICS")
    print("#"*60)
    random_label_results = run_label_distribution_metrics(
        random_client0, random_client1, num_classes
    )

    print("\n" + "#"*60)
    print("# DIRICHLET SPLIT - LABEL METRICS")
    print("#"*60)
    dirichlet_label_results = run_label_distribution_metrics(
        dirichlet_client0, dirichlet_client1, num_classes
    )

    # Run feature distribution metrics (if not skipped)
    random_feature_results = {}
    dirichlet_feature_results = {}

    if not args.skip_feature_metrics:
        # Prepare image path datasets (use subset for speed)
        random_client0_imgs = create_image_path_dataset(dataset, random_indices[0])
        random_client1_imgs = create_image_path_dataset(dataset, random_indices[1])

        dirichlet_client0_imgs = create_image_path_dataset(dataset, dirichlet_indices[0])
        dirichlet_client1_imgs = create_image_path_dataset(dataset, dirichlet_indices[1])

        print("\n" + "#"*60)
        print("# RANDOM SPLIT - FEATURE METRICS")
        print("#"*60)
        random_feature_results = run_feature_distribution_metrics(
            random_client0_imgs, random_client1_imgs, device
        )

        print("\n" + "#"*60)
        print("# DIRICHLET SPLIT - FEATURE METRICS")
        print("#"*60)
        dirichlet_feature_results = run_feature_distribution_metrics(
            dirichlet_client0_imgs, dirichlet_client1_imgs, device
        )

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: RANDOM vs DIRICHLET COMPARISON")
    print("="*60)
    print("\nExpected: Dirichlet metrics >> Random metrics (more heterogeneous)")
    print("\nLabel Distribution Metrics:")
    print(f"{'Metric':<30} {'Random':<15} {'Dirichlet':<15} {'Ratio (D/R)':<15}")
    print("-"*75)

    for metric_name in random_label_results.keys():
        random_val = random_label_results[metric_name]
        dirichlet_val = dirichlet_label_results[metric_name]

        if random_val is not None and dirichlet_val is not None:
            if isinstance(random_val, dict):
                # Handle KS test result
                random_val = random_val['statistic']
                dirichlet_val = dirichlet_val['statistic']

            ratio = dirichlet_val / random_val if random_val > 1e-10 else float('inf')
            print(f"{metric_name:<30} {random_val:<15.6f} {dirichlet_val:<15.6f} {ratio:<15.2f}")

    if not args.skip_feature_metrics:
        print("\nFeature Distribution Metrics:")
        print(f"{'Metric':<30} {'Random':<15} {'Dirichlet':<15} {'Ratio (D/R)':<15}")
        print("-"*75)

        for metric_name in random_feature_results.keys():
            random_val = random_feature_results[metric_name]
            dirichlet_val = dirichlet_feature_results[metric_name]

            if random_val is not None and dirichlet_val is not None:
                if isinstance(random_val, dict):
                    continue  # Skip KS multivariate for this table

                ratio = dirichlet_val / random_val if random_val > 1e-10 else float('inf')
                print(f"{metric_name:<30} {random_val:<15.6f} {dirichlet_val:<15.6f} {ratio:<15.2f}")

    # Save results
    results = {
        'metadata': {
            'dataset': args.dataset,
            'num_clients': args.num_clients,
            'dirichlet_alpha': args.dirichlet_alpha,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat()
        },
        'random_split': {
            'label_metrics': random_label_results,
            'feature_metrics': random_feature_results
        },
        'dirichlet_split': {
            'label_metrics': dirichlet_label_results,
            'feature_metrics': dirichlet_feature_results
        }
    }

    output_file = f'metric_validation_{args.dataset}_alpha{args.dirichlet_alpha}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
