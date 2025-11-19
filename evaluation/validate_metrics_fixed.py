"""
Validation script to test evaluation metrics (FIXED VERSION).

This script compares Random (IID) splits vs high-heterogeneity Dirichlet splits
to validate that our metrics correctly detect distribution shifts.

Compatible with the actual BaseDatasetSplitter interface.
"""

import sys
import os

import numpy as np
import torch
from data_partitioning.dataset import BaseDataset
from data_partitioning.random_splitter import RandomSplitter
from data_partitioning.dirichlet_splitter import DirichletSplitter
from evaluation import metrics
import argparse
import json
from datetime import datetime
from collections import Counter


def convert_splits_to_metric_format(client_split_data):
    """
    Convert splitter output format to metric format.

    Splitter format: [{"image": "/path", "label": "class_name"}, ...]
    Metric format: [{"image_path": "/path", "label": int}, ...]
    """
    # Build label to int mapping
    all_labels = [item["label"] for item in client_split_data]
    unique_labels = sorted(set(all_labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    # Convert format
    metric_data = []
    for item in client_split_data:
        metric_data.append({
            "image_path": item["image"],
            "label": label_to_int[item["label"]]
        })

    return metric_data, label_to_int


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


def run_feature_distribution_metrics(dataset1, dataset2, device, batch_size=32, max_samples=100):
    """Run all feature distribution metrics on a subset."""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION METRICS")
    print("="*60)
    print(f"(Using subset of {max_samples} images per client for speed)")

    # Subsample for speed
    dataset1_subset = dataset1[:max_samples] if len(dataset1) > max_samples else dataset1
    dataset2_subset = dataset2[:max_samples] if len(dataset2) > max_samples else dataset2

    results = {}

    # FID Distance
    try:
        print("Computing FID distance...")
        fid_score = metrics.FID_distance(
            [dataset1_subset, dataset2_subset],
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
            dataset1_subset, dataset2_subset,
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
            dataset1_subset, dataset2_subset,
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
            dataset1_subset, dataset2_subset,
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
            dataset1_subset, dataset2_subset,
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
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['tiny_imagenet', 'food101', 'ade20k'],
                        help='Dataset to use for validation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--num_clients', type=int, default=2,
                        help='Number of clients to split into')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1,
                        help='Dirichlet alpha parameter (lower = more heterogeneous)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip_feature_metrics', action='store_true',
                        help='Skip feature-based metrics (faster)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_samples_features', type=int, default=100,
                        help='Max samples per client for feature metrics')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*60)
    print("METRIC VALIDATION SCRIPT (FIXED)")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Dirichlet alpha: {args.dirichlet_alpha}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("="*60)

    # Load datasets using BaseDataset
    try:
        train_dataset = BaseDataset(args.dataset, args.data_path, "train")
        val_dataset = BaseDataset(args.dataset, args.data_path, "val" if args.dataset != "food101" else "test")
        print(f"\n✓ Loaded {args.dataset}: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

        # Get number of unique classes
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        num_classes = len(set(train_labels))
        print(f"✓ Number of classes: {num_classes}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("\nMake sure the dataset is available at the specified path.")
        print(f"Expected structure for {args.dataset} at {args.data_path}")
        return

    # Create Random split (IID)
    print("\n" + "="*60)
    print("CREATING RANDOM (IID) SPLIT")
    print("="*60)
    try:
        random_splitter = RandomSplitter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=args.num_clients,
            heterogeneity=0.0,
            valset_type="none",  # Skip val for simplicity
            seed=args.seed
        )
        random_splits = random_splitter.get_client_splits()
        print(f"✓ Random split created:")
        for client_id in sorted(random_splits["clients"].keys()):
            n_samples = len(random_splits["clients"][client_id]["train"])
            print(f"  {client_id}: {n_samples} samples")
    except Exception as e:
        print(f"✗ Failed to create random split: {e}")
        return

    # Create Dirichlet split (non-IID)
    print("\n" + "="*60)
    print(f"CREATING DIRICHLET SPLIT (alpha={args.dirichlet_alpha})")
    print("="*60)
    try:
        dirichlet_splitter = DirichletSplitter(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=args.num_clients,
            heterogeneity=args.dirichlet_alpha,
            valset_type="none",
            seed=args.seed
        )
        dirichlet_splits = dirichlet_splitter.get_client_splits()
        print(f"✓ Dirichlet split created:")
        for client_id in sorted(dirichlet_splits["clients"].keys()):
            n_samples = len(dirichlet_splits["clients"][client_id]["train"])
            print(f"  {client_id}: {n_samples} samples")
    except Exception as e:
        print(f"✗ Failed to create Dirichlet split: {e}")
        return

    # Show label distributions
    print("\n" + "="*60)
    print("LABEL DISTRIBUTIONS")
    print("="*60)

    for split_name, splits in [('Random', random_splits), ('Dirichlet', dirichlet_splits)]:
        print(f"\n{split_name} Split:")
        for client_id in sorted(splits["clients"].keys()):
            client_data = splits["clients"][client_id]["train"]
            labels = [item["label"] for item in client_data]
            label_counts = Counter(labels)
            print(f"  {client_id}: {dict(label_counts.most_common(5))}...")

    # Convert to metric format
    print("\n" + "="*60)
    print("CONVERTING TO METRIC FORMAT")
    print("="*60)

    client_ids = sorted(random_splits["clients"].keys())
    random_c0_raw = random_splits["clients"][client_ids[0]]["train"]
    random_c1_raw = random_splits["clients"][client_ids[1]]["train"]
    dirichlet_c0_raw = dirichlet_splits["clients"][client_ids[0]]["train"]
    dirichlet_c1_raw = dirichlet_splits["clients"][client_ids[1]]["train"]

    random_c0, label_map_r0 = convert_splits_to_metric_format(random_c0_raw)
    random_c1, label_map_r1 = convert_splits_to_metric_format(random_c1_raw)
    dirichlet_c0, label_map_d0 = convert_splits_to_metric_format(dirichlet_c0_raw)
    dirichlet_c1, label_map_d1 = convert_splits_to_metric_format(dirichlet_c1_raw)

    print(f"✓ Converted to metric format ({len(random_c0)} and {len(random_c1)} samples)")

    # Run label distribution metrics
    print("\n" + "#"*60)
    print("# RANDOM SPLIT - LABEL METRICS")
    print("#"*60)
    random_label_results = run_label_distribution_metrics(
        random_c0, random_c1, num_classes
    )

    print("\n" + "#"*60)
    print("# DIRICHLET SPLIT - LABEL METRICS")
    print("#"*60)
    dirichlet_label_results = run_label_distribution_metrics(
        dirichlet_c0, dirichlet_c1, num_classes
    )

    # Run feature distribution metrics (if not skipped)
    random_feature_results = {}
    dirichlet_feature_results = {}

    if not args.skip_feature_metrics:
        print("\n" + "#"*60)
        print("# RANDOM SPLIT - FEATURE METRICS")
        print("#"*60)
        random_feature_results = run_feature_distribution_metrics(
            random_c0, random_c1, device, max_samples=args.max_samples_features
        )

        print("\n" + "#"*60)
        print("# DIRICHLET SPLIT - FEATURE METRICS")
        print("#"*60)
        dirichlet_feature_results = run_feature_distribution_metrics(
            dirichlet_c0, dirichlet_c1, device, max_samples=args.max_samples_features
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
            status = "✅" if ratio > 1.2 else "⚠️"
            print(f"{status} {metric_name:<27} {random_val:<15.6f} {dirichlet_val:<15.6f} {ratio:<15.2f}")

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
                status = "✅" if ratio > 1.2 else "⚠️"
                print(f"{status} {metric_name:<27} {random_val:<15.6f} {dirichlet_val:<15.6f} {ratio:<15.2f}")

    # Save results
    results = {
        'metadata': {
            'dataset': args.dataset,
            'data_path': args.data_path,
            'num_clients': args.num_clients,
            'dirichlet_alpha': args.dirichlet_alpha,
            'seed': args.seed,
            'num_classes': num_classes,
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
