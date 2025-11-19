"""
Test script for data modules.

Tests that data modules can load splits, create datasets, and provide dataloaders.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_modules.data_modules import (
    get_datamodule,
    TinyImageNetDataModule,
    Food101DataModule,
    ADE20KDataModule,
)


def test_datamodule_basic_interface():
    """Test basic data module interface without real data."""
    print("=" * 80)
    print("Test 1: Data Module Basic Interface")
    print("=" * 80)

    # Create a minimal mock split file for testing
    import json
    import tempfile

    mock_splits = {
        "clients": {
            "client_0": {
                "train": [
                    {"image": "/fake/path/img1.jpg", "label": "class_a"},
                    {"image": "/fake/path/img2.jpg", "label": "class_b"},
                ],
                "val": [
                    {"image": "/fake/path/img3.jpg", "label": "class_a"},
                ],
            },
            "client_1": {
                "train": [
                    {"image": "/fake/path/img4.jpg", "label": "class_b"},
                    {"image": "/fake/path/img5.jpg", "label": "class_a"},
                ],
                "val": [],
            },
        },
        "global_val": [
            {"image": "/fake/path/val1.jpg", "label": "class_a"},
            {"image": "/fake/path/val2.jpg", "label": "class_b"},
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_splits, f)
        split_file = f.name

    try:
        # Test TinyImageNet data module
        print("\nTesting TinyImageNetDataModule...")
        dm = TinyImageNetDataModule(
            data_dir="./data",
            split_file=split_file,
            client_idx=0,
            central=False,
            batch_size=2,
            num_workers=0,  # No workers for testing
        )

        # Load splits
        splits = dm.load_splits()
        print(f"✓ Loaded splits: {len(splits['clients'])} clients")

        # Get client data
        train_data, val_data = dm.get_client_data(splits)
        print(f"✓ Client 0: {len(train_data)} train, {len(val_data)} val samples")

        # Create label map
        label_map = dm.create_label_map(train_data)
        print(f"✓ Label map: {label_map}")

        # Get transforms
        train_transform = dm.get_train_transform()
        val_transform = dm.get_val_transform()
        print(f"✓ Transforms created")

        print("\nTinyImageNetDataModule interface test passed!")

    finally:
        Path(split_file).unlink()


def test_central_vs_federated_mode():
    """Test central vs federated mode data loading."""
    print("\n" + "=" * 80)
    print("Test 2: Central vs Federated Mode")
    print("=" * 80)

    import json
    import tempfile

    mock_splits = {
        "clients": {
            "client_0": {
                "train": [
                    {"image": "/fake/img1.jpg", "label": "cat"},
                    {"image": "/fake/img2.jpg", "label": "dog"},
                ],
                "val": [],
            },
            "client_1": {
                "train": [
                    {"image": "/fake/img3.jpg", "label": "cat"},
                ],
                "val": [],
            },
        },
        "global_val": [{"image": "/fake/val1.jpg", "label": "cat"}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_splits, f)
        split_file = f.name

    try:
        # Federated mode
        dm_fed = TinyImageNetDataModule(
            data_dir="./data",
            split_file=split_file,
            client_idx=0,
            central=False,
            batch_size=2,
            num_workers=0,
        )
        splits = dm_fed.load_splits()
        train_fed, val_fed = dm_fed.get_client_data(splits)
        print(f"Federated (client_0): {len(train_fed)} train samples")
        assert len(train_fed) == 2, "Client 0 should have 2 training samples"

        # Central mode
        dm_central = TinyImageNetDataModule(
            data_dir="./data",
            split_file=split_file,
            client_idx=None,
            central=True,
            batch_size=2,
            num_workers=0,
        )
        train_central, val_central = dm_central.get_client_data(splits)
        print(f"Central mode: {len(train_central)} train samples")
        assert len(train_central) == 3, "Central should have 3 training samples (all clients)"

        print("\n✓ Central vs Federated mode test passed!")

    finally:
        Path(split_file).unlink()


def test_factory_function():
    """Test factory function for creating data modules."""
    print("\n" + "=" * 80)
    print("Test 3: Factory Function")
    print("=" * 80)

    import json
    import tempfile

    mock_splits = {
        "clients": {
            "client_0": {
                "train": [{"image": "/fake/img.jpg", "label": "class_a"}],
                "val": [],
            }
        },
        "global_val": None,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_splits, f)
        split_file = f.name

    try:
        # Test creating different dataset modules
        datasets = ["tiny_imagenet", "food101", "ade20k"]

        for dataset_name in datasets:
            print(f"\nCreating {dataset_name} data module...")
            dm = get_datamodule(
                dataset_name=dataset_name,
                data_dir="./data",
                split_file=split_file,
                client_idx=0,
                batch_size=2,
                num_workers=0,
            )
            print(f"✓ Created: {dm}")

        print("\n✓ Factory function test passed!")

    finally:
        Path(split_file).unlink()


def test_label_map_creation():
    """Test label map creation from training data."""
    print("\n" + "=" * 80)
    print("Test 4: Label Map Creation")
    print("=" * 80)

    from data_modules.data_modules import BaseDataModule

    # Mock data module for testing
    class MockDataModule(BaseDataModule):
        def get_train_transform(self):
            pass

        def get_val_transform(self):
            pass

    import json
    import tempfile

    mock_splits = {
        "clients": {"client_0": {"train": [], "val": []}},
        "global_val": None,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_splits, f)
        split_file = f.name

    try:
        dm = MockDataModule(
            data_dir="./data",
            split_file=split_file,
            client_idx=0,
        )

        # Test label map creation
        train_data = [
            {"image": "/path/img1.jpg", "label": "zebra"},
            {"image": "/path/img2.jpg", "label": "apple"},
            {"image": "/path/img3.jpg", "label": "zebra"},
            {"image": "/path/img4.jpg", "label": "banana"},
        ]

        label_map = dm.create_label_map(train_data)
        print(f"Label map: {label_map}")

        # Check label map is correct
        assert "apple" in label_map
        assert "banana" in label_map
        assert "zebra" in label_map
        assert len(label_map) == 3

        # Check labels are alphabetically sorted (apple, banana, zebra)
        assert label_map["apple"] == 0
        assert label_map["banana"] == 1
        assert label_map["zebra"] == 2

        print("✓ Label map creation test passed!")

    finally:
        Path(split_file).unlink()


def test_repr():
    """Test string representation of data modules."""
    print("\n" + "=" * 80)
    print("Test 5: String Representation")
    print("=" * 80)

    import json
    import tempfile

    mock_splits = {
        "clients": {"client_0": {"train": [], "val": []}},
        "global_val": None,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_splits, f)
        split_file = f.name

    try:
        # Federated mode
        dm_fed = TinyImageNetDataModule(
            data_dir="./data",
            split_file=split_file,
            client_idx=5,
            central=False,
            batch_size=64,
        )
        print(f"Federated: {dm_fed}")
        assert "Client 5" in str(dm_fed)

        # Central mode
        dm_central = TinyImageNetDataModule(
            data_dir="./data",
            split_file=split_file,
            central=True,
            batch_size=128,
        )
        print(f"Central: {dm_central}")
        assert "Central" in str(dm_central)

        print("\n✓ String representation test passed!")

    finally:
        Path(split_file).unlink()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING DATA MODULE TESTS")
    print("=" * 80)

    try:
        test_datamodule_basic_interface()
        test_central_vs_federated_mode()
        test_factory_function()
        test_label_map_creation()
        test_repr()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
