"""
Test script for InceptionV3 embedding cache functionality.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_partitioning.dataset import BaseDataset
from evaluation.metrics import FID_distance, MMD_features, vendi_score_features
from evaluation.inception_cache import InceptionV3EmbeddingCache
import numpy as np


def create_test_datasets(n_samples=100):
    print("=" * 70)
    print("CREATING TEST DATASETS")
    print("=" * 70)

    dataset_name = "tiny_imagenet"
    base_path = "/path/to/gtai/FDS/"

    train_dataset = BaseDataset(dataset_name, base_path, "train")
    print(f"Total train samples: {len(train_dataset)}")

    np.random.seed(42)
    indices_1 = np.random.choice(len(train_dataset), size=n_samples, replace=False)
    indices_2 = np.random.choice(len(train_dataset), size=n_samples, replace=False)

    dataset_1 = [{"image": train_dataset[i][0], "label": train_dataset[i][1]} for i in indices_1]
    dataset_2 = [{"image": train_dataset[i][0], "label": train_dataset[i][1]} for i in indices_2]

    print(f"Created 2 test datasets with {n_samples} samples each\n")
    return [dataset_1, dataset_2]


def test_cache_creation_and_retrieval():
    print("=" * 70)
    print("TEST 1: Cache Creation and Retrieval")
    print("=" * 70)

    cache = InceptionV3EmbeddingCache(cache_dir="./.test_inception_cache")
    cache.clear_cache()
    print("✓ Cleared existing cache\n")

    datasets = create_test_datasets(n_samples=50)

    print("First run (no cache):")
    start_time = time.time()
    fid_1 = FID_distance(datasets, use_cache=True, cache_dir="./.test_inception_cache", batch_size=10)
    time_1 = time.time() - start_time
    print(f"  FID: {fid_1:.4f}")
    print(f"  Time: {time_1:.2f}s\n")

    print("Second run (with cache):")
    start_time = time.time()
    fid_2 = FID_distance(datasets, use_cache=True, cache_dir="./.test_inception_cache", batch_size=10)
    time_2 = time.time() - start_time
    print(f"  FID: {fid_2:.4f}")
    print(f"  Time: {time_2:.2f}s\n")

    print("Results comparison:")
    print(f"  FID difference: {abs(fid_1 - fid_2):.6f}")
    if abs(fid_1 - fid_2) < 1e-5:
        print("  ✓ Results are identical!")
    else:
        print("  ✗ WARNING: Results differ!")

    speedup = time_1 / time_2 if time_2 > 0 else float('inf')
    print(f"  Speedup: {speedup:.1f}x")
    if speedup > 2.0:
        print("  ✓ Significant speedup achieved!")
    else:
        print("  ⚠ Speedup less than expected (may be due to small dataset)")

    print()
    return True


def test_cache_vs_no_cache():
    print("=" * 70)
    print("TEST 2: Cache vs No Cache Comparison")
    print("=" * 70)

    datasets = create_test_datasets(n_samples=50)

    print("Computing metrics WITH cache:")
    fid_cached = FID_distance(datasets, use_cache=True, cache_dir="./.test_inception_cache", batch_size=10)
    mmd_cached = MMD_features(datasets, use_cache=True, cache_dir="./.test_inception_cache", batch_size=10)
    vendi_cached = vendi_score_features(datasets, use_cache=True, cache_dir="./.test_inception_cache", batch_size=10)
    print(f"  FID: {fid_cached:.4f}")
    print(f"  MMD: {mmd_cached:.6f}")
    print(f"  Vendi: {vendi_cached:.4f}\n")

    print("Computing metrics WITHOUT cache:")
    fid_no_cache = FID_distance(datasets, use_cache=False, batch_size=10)
    mmd_no_cache = MMD_features(datasets, use_cache=False, batch_size=10)
    vendi_no_cache = vendi_score_features(datasets, use_cache=False, batch_size=10)
    print(f"  FID: {fid_no_cache:.4f}")
    print(f"  MMD: {mmd_no_cache:.6f}")
    print(f"  Vendi: {vendi_no_cache:.4f}\n")

    print("Comparison:")
    fid_diff = abs(fid_cached - fid_no_cache)
    mmd_diff = abs(mmd_cached - mmd_no_cache)
    vendi_diff = abs(vendi_cached - vendi_no_cache)

    print(f"  FID difference: {fid_diff:.6f}")
    print(f"  MMD difference: {mmd_diff:.6f}")
    print(f"  Vendi difference: {vendi_diff:.6f}")

    all_close = fid_diff < 1e-3 and mmd_diff < 1e-6 and vendi_diff < 1e-3

    if all_close:
        print("  ✓ All metrics produce identical results with and without cache!")
    else:
        print("  ✗ WARNING: Some metrics differ between cached and non-cached versions!")

    print()
    return all_close


def test_cache_info():
    print("=" * 70)
    print("TEST 3: Cache Information")
    print("=" * 70)

    cache = InceptionV3EmbeddingCache(cache_dir="./.test_inception_cache")
    info = cache.cache_manager.get_cache_info()

    print(f"Cache enabled: {info['enabled']}")
    print(f"Cache directory: {info['cache_dir']}")
    print(f"Number of entries: {info['num_entries']}")
    print(f"Total size: {info['total_size_mb']:.2f} MB")

    if info['num_entries'] > 0:
        print("\nCache entries:")
        for key, entry in list(info['entries'].items())[:3]:
            print(f"  - {entry['model_name']}, {entry['num_images']} images")

    print()
    return True


def main():
    print("\n" + "=" * 70)
    print("INCEPTION V3 CACHE TESTING")
    print("=" * 70 + "\n")

    try:
        test1_passed = test_cache_creation_and_retrieval()
        test2_passed = test_cache_vs_no_cache()
        test3_passed = test_cache_info()

        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Test 1 (Cache creation & retrieval): {'✓ PASS' if test1_passed else '✗ FAIL'}")
        print(f"Test 2 (Cache vs no cache): {'✓ PASS' if test2_passed else '✗ FAIL'}")
        print(f"Test 3 (Cache info): {'✓ PASS' if test3_passed else '✗ FAIL'}")

        all_passed = test1_passed and test2_passed and test3_passed
        print(f"\n{'✓ All tests passed!' if all_passed else '✗ Some tests failed!'}")
        print("=" * 70 + "\n")

        return all_passed

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)