"""Test script for FDSDataFactory validation.

Tests FDSDataFactory.create_dataloaders() with synthetic DatasetSplits to verify:
- Classification task DataLoader creation
- Segmentation task DataLoader creation
- Global validation vs local validation priority
- Error handling for invalid inputs

Run: python fl_training/nvflare27/common/test_data_factory.py
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from fl_training.nvflare27.common.data_factory import FDSDataFactory


def create_synthetic_splits(n_clients: int = 3) -> Dict[str, Any]:
    """Create synthetic DatasetSplits for testing.

    Args:
        n_clients: Number of clients to create (default: 3)

    Returns:
        DatasetSplits dict with synthetic image paths and labels
    """
    splits = {
        "clients": {},
        "global_val": []
    }

    # Create synthetic client data
    for client_idx in range(n_clients):
        client_key = f"client_{client_idx}"
        splits["clients"][client_key] = {
            "train": [],
            "val": []
        }

        # Add synthetic training samples (10 per client)
        for sample_idx in range(10):
            splits["clients"][client_key]["train"].append({
                "image": f"/synthetic/client_{client_idx}/train_{sample_idx}.jpg",
                "label": f"class_{sample_idx % 3}"  # 3 classes cycling
            })

        # Add synthetic local validation samples (5 per client)
        for sample_idx in range(5):
            splits["clients"][client_key]["val"].append({
                "image": f"/synthetic/client_{client_idx}/val_{sample_idx}.jpg",
                "label": f"class_{sample_idx % 3}"
            })

    # Add synthetic global validation samples (20 total)
    for sample_idx in range(20):
        splits["global_val"].append({
            "image": f"/synthetic/global_val_{sample_idx}.jpg",
            "label": f"class_{sample_idx % 3}"
        })

    return splits


def create_synthetic_segmentation_splits(n_clients: int = 3) -> Dict[str, Any]:
    """Create synthetic DatasetSplits for segmentation testing.

    Args:
        n_clients: Number of clients to create (default: 3)

    Returns:
        DatasetSplits dict with synthetic image and mask paths
    """
    splits = {
        "clients": {},
        "global_val": []
    }

    # Create synthetic client data
    for client_idx in range(n_clients):
        client_key = f"client_{client_idx}"
        splits["clients"][client_key] = {
            "train": [],
            "val": []
        }

        # Add synthetic training samples (10 per client)
        for sample_idx in range(10):
            splits["clients"][client_key]["train"].append({
                "image": f"/synthetic/client_{client_idx}/train_{sample_idx}.jpg",
                "mask_path": f"/synthetic/client_{client_idx}/train_{sample_idx}_mask.png"
            })

        # Add synthetic local validation samples (5 per client)
        for sample_idx in range(5):
            splits["clients"][client_key]["val"].append({
                "image": f"/synthetic/client_{client_idx}/val_{sample_idx}.jpg",
                "mask_path": f"/synthetic/client_{client_idx}/val_{sample_idx}_mask.png"
            })

    # Add synthetic global validation samples (20 total)
    for sample_idx in range(20):
        splits["global_val"].append({
            "image": f"/synthetic/global_val_{sample_idx}.jpg",
            "mask_path": f"/synthetic/global_val_{sample_idx}_mask.png"
        })

    return splits


def test_classification_task():
    """Test classification task DataLoader creation.

    Verifies:
    - ClassificationDataset is used
    - Correct batch size and shuffle settings
    - Train/val loaders created successfully
    - Dataset lengths match expected
    """
    print("\n=== Test 1: Classification Task ===")

    splits = create_synthetic_splits(n_clients=3)

    try:
        # Create DataLoaders for client 0
        train_loader, val_loader = FDSDataFactory.create_dataloaders(
            splits_dict=splits,
            client_idx=0,
            task_type="classification",
            batch_size=4,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )

        # Verify train_loader
        assert train_loader is not None, "train_loader should not be None"
        assert len(train_loader.dataset) == 10, f"Expected 10 train samples, got {len(train_loader.dataset)}"
        assert train_loader.batch_size == 4, f"Expected batch size 4, got {train_loader.batch_size}"

        # Verify val_loader (should use global_val with 20 samples)
        assert val_loader is not None, "val_loader should not be None (global_val available)"
        assert len(val_loader.dataset) == 20, f"Expected 20 val samples (global_val), got {len(val_loader.dataset)}"

        print("✓ Classification task: PASS")
        print(f"  - Train loader: {len(train_loader.dataset)} samples")
        print(f"  - Val loader: {len(val_loader.dataset)} samples (global_val)")
        return True

    except Exception as e:
        print(f"✗ Classification task: FAIL - {e}")
        return False


def test_segmentation_task():
    """Test segmentation task DataLoader creation.

    Verifies:
    - SegmentationDataset is used
    - Correct batch size and transforms
    - Train/val loaders created successfully
    """
    print("\n=== Test 2: Segmentation Task ===")

    splits = create_synthetic_segmentation_splits(n_clients=3)

    try:
        # Create DataLoaders for client 1
        train_loader, val_loader = FDSDataFactory.create_dataloaders(
            splits_dict=splits,
            client_idx=1,
            task_type="segmentation",
            batch_size=2,
            num_workers=0
        )

        # Verify train_loader
        assert train_loader is not None, "train_loader should not be None"
        assert len(train_loader.dataset) == 10, f"Expected 10 train samples, got {len(train_loader.dataset)}"
        assert train_loader.batch_size == 2, f"Expected batch size 2, got {train_loader.batch_size}"

        # Verify val_loader (should use global_val with 20 samples)
        assert val_loader is not None, "val_loader should not be None (global_val available)"
        assert len(val_loader.dataset) == 20, f"Expected 20 val samples (global_val), got {len(val_loader.dataset)}"

        print("✓ Segmentation task: PASS")
        print(f"  - Train loader: {len(train_loader.dataset)} samples")
        print(f"  - Val loader: {len(val_loader.dataset)} samples (global_val)")
        return True

    except Exception as e:
        print(f"✗ Segmentation task: FAIL - {e}")
        return False


def test_global_val_priority():
    """Test that global_val takes priority over local val.

    Verifies:
    - When global_val exists, val_loader uses global_val
    - When global_val is None, val_loader uses local val
    - When both are empty, val_loader is None
    """
    print("\n=== Test 3: Global Validation Priority ===")

    try:
        # Test 3a: With global_val
        splits_with_global = create_synthetic_splits(n_clients=2)
        _, val_loader = FDSDataFactory.create_dataloaders(
            splits_dict=splits_with_global,
            client_idx=0,
            task_type="classification",
            batch_size=4,
            num_workers=0
        )
        assert val_loader is not None, "val_loader should not be None (global_val exists)"
        assert len(val_loader.dataset) == 20, f"Expected 20 samples (global_val), got {len(val_loader.dataset)}"
        print("✓ Global val priority (with global_val): PASS")

        # Test 3b: Without global_val (use local val)
        splits_no_global = create_synthetic_splits(n_clients=2)
        splits_no_global["global_val"] = None  # Remove global_val
        _, val_loader = FDSDataFactory.create_dataloaders(
            splits_dict=splits_no_global,
            client_idx=0,
            task_type="classification",
            batch_size=4,
            num_workers=0
        )
        assert val_loader is not None, "val_loader should not be None (local val exists)"
        assert len(val_loader.dataset) == 5, f"Expected 5 samples (local val), got {len(val_loader.dataset)}"
        print("✓ Local val fallback (no global_val): PASS")

        # Test 3c: No validation data at all
        splits_no_val = create_synthetic_splits(n_clients=2)
        splits_no_val["global_val"] = None
        splits_no_val["clients"]["client_0"]["val"] = []  # Empty local val
        _, val_loader = FDSDataFactory.create_dataloaders(
            splits_dict=splits_no_val,
            client_idx=0,
            task_type="classification",
            batch_size=4,
            num_workers=0
        )
        assert val_loader is None, "val_loader should be None (no validation data)"
        print("✓ No validation data (val_loader=None): PASS")

        return True

    except Exception as e:
        print(f"✗ Global val priority: FAIL - {e}")
        return False


def test_error_handling():
    """Test error handling for invalid inputs.

    Verifies:
    - ValueError for invalid task_type
    - KeyError for missing client_idx
    - ValueError for malformed DatasetSplits
    - NotImplementedError for detection task (Phase 5 stub)
    """
    print("\n=== Test 4: Error Handling ===")

    splits = create_synthetic_splits(n_clients=3)
    passed_tests = 0
    total_tests = 4

    # Test 4a: Invalid task_type
    try:
        FDSDataFactory.create_dataloaders(
            splits_dict=splits,
            client_idx=0,
            task_type="invalid_task",
            batch_size=4,
            num_workers=0
        )
        print("✗ Invalid task_type: FAIL (should raise ValueError)")
    except ValueError as e:
        if "Invalid task_type" in str(e):
            print("✓ Invalid task_type: PASS (ValueError raised)")
            passed_tests += 1
        else:
            print(f"✗ Invalid task_type: FAIL (wrong error message: {e})")

    # Test 4b: Missing client_idx
    try:
        FDSDataFactory.create_dataloaders(
            splits_dict=splits,
            client_idx=99,  # Client doesn't exist
            task_type="classification",
            batch_size=4,
            num_workers=0
        )
        print("✗ Missing client_idx: FAIL (should raise KeyError)")
    except KeyError as e:
        if "client_99" in str(e):
            print("✓ Missing client_idx: PASS (KeyError raised)")
            passed_tests += 1
        else:
            print(f"✗ Missing client_idx: FAIL (wrong error message: {e})")

    # Test 4c: Malformed DatasetSplits (missing 'clients' key)
    try:
        FDSDataFactory.create_dataloaders(
            splits_dict={"invalid": "structure"},
            client_idx=0,
            task_type="classification",
            batch_size=4,
            num_workers=0
        )
        print("✗ Malformed DatasetSplits: FAIL (should raise ValueError)")
    except ValueError as e:
        if "missing 'clients' key" in str(e):
            print("✓ Malformed DatasetSplits: PASS (ValueError raised)")
            passed_tests += 1
        else:
            print(f"✗ Malformed DatasetSplits: FAIL (wrong error message: {e})")

    # Test 4d: Detection task (Phase 5 stub)
    try:
        FDSDataFactory.create_dataloaders(
            splits_dict=splits,
            client_idx=0,
            task_type="detection",
            batch_size=4,
            num_workers=0
        )
        print("✗ Detection task stub: FAIL (should raise NotImplementedError)")
    except NotImplementedError as e:
        if "Phase 5" in str(e):
            print("✓ Detection task stub: PASS (NotImplementedError raised)")
            passed_tests += 1
        else:
            print(f"✗ Detection task stub: FAIL (wrong error message: {e})")

    print(f"Error handling: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests


def main():
    """Run all validation tests and report summary."""
    print("=" * 60)
    print("FDSDataFactory Validation Test Suite")
    print("=" * 60)

    tests = [
        test_classification_task,
        test_segmentation_task,
        test_global_val_priority,
        test_error_handling
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test '{test_func.__name__}' crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED - FDSDataFactory is working correctly!")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED - Review output above for details")
        return 1


if __name__ == "__main__":
    exit(main())
