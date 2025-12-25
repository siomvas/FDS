"""Shared pytest fixtures for FDS test suite.

This module provides reusable fixtures for:
- Mock datasets (balanced, imbalanced, with various sizes)
- Embeddings (2D pooled, 3D with tokens, clusterable)
- VLM annotations (JSONL format for all 5 criteria)
- Cache management (temporary directories, CacheManager instances)
- Configuration (embedding configs for different test scenarios)
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def simple_dataset():
    """30 samples, 3 classes, balanced distribution (10 per class).

    Returns:
        List of dicts with 'image' (path) and 'label' (string) keys.
    """
    dataset = []
    for class_id in range(3):
        for sample_id in range(10):
            dataset.append({
                "image": f"/tmp/class_{class_id}/sample_{sample_id}.jpg",
                "label": f"class_{class_id}"
            })
    return dataset


@pytest.fixture
def imbalanced_dataset():
    """100 samples, 3 classes, highly imbalanced (90-5-5 distribution).

    Returns:
        List of dicts with 'image' (path) and 'label' (string) keys.
    """
    dataset = []

    # Class 0: 90 samples
    for i in range(90):
        dataset.append({
            "image": f"/tmp/class_0/sample_{i}.jpg",
            "label": "class_0"
        })

    # Class 1: 5 samples
    for i in range(5):
        dataset.append({
            "image": f"/tmp/class_1/sample_{i}.jpg",
            "label": "class_1"
        })

    # Class 2: 5 samples
    for i in range(5):
        dataset.append({
            "image": f"/tmp/class_2/sample_{i}.jpg",
            "label": "class_2"
        })

    return dataset


@pytest.fixture
def large_dataset():
    """1000 samples, 10 classes, balanced distribution.

    Useful for property-based testing and performance tests.

    Returns:
        List of dicts with 'image' (path) and 'label' (string) keys.
    """
    dataset = []
    for class_id in range(10):
        for sample_id in range(100):
            dataset.append({
                "image": f"/tmp/class_{class_id}/sample_{sample_id}.jpg",
                "label": f"class_{class_id}"
            })
    return dataset


class MockDataset:
    """Mock BaseDataset for testing without file I/O.

    Simulates the BaseDataset interface for testing splitters.
    """

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["image"], item["label"]

    def get_summary(self):
        labels = [item["label"] for item in self.data]
        unique_labels = sorted(set(labels))
        return {
            "dataset_size": len(self.data),
            "num_classes": len(unique_labels),
            "labels": unique_labels
        }


@pytest.fixture
def mock_train_dataset(simple_dataset):
    """Mock training dataset (30 samples, 3 classes)."""
    return MockDataset(simple_dataset)


@pytest.fixture
def mock_val_dataset():
    """Mock validation dataset (15 samples, 3 classes, balanced)."""
    val_data = []
    for class_id in range(3):
        for sample_id in range(5):
            val_data.append({
                "image": f"/tmp/val/class_{class_id}/sample_{sample_id}.jpg",
                "label": f"class_{class_id}"
            })
    return MockDataset(val_data)


# ============================================================================
# Embedding Fixtures
# ============================================================================

@pytest.fixture
def sample_embeddings_2d():
    """2D embeddings: 100 samples, 768 dimensions (already pooled).

    Returns:
        np.ndarray of shape (100, 768).
    """
    np.random.seed(42)
    return np.random.randn(100, 768).astype(np.float32)


@pytest.fixture
def sample_embeddings_3d():
    """3D embeddings: 50 samples, 197 tokens, 768 dimensions.

    Simulates Vision Transformer (ViT) output with CLS token + patch tokens.

    Returns:
        np.ndarray of shape (50, 197, 768).
    """
    np.random.seed(42)
    return np.random.randn(50, 197, 768).astype(np.float32)


@pytest.fixture
def clusterable_embeddings():
    """Embeddings with 3 well-separated clusters for testing clustering.

    - Cluster 0: 40 samples centered at [10, 0, 0, ...]
    - Cluster 1: 30 samples centered at [-10, 0, 0, ...]
    - Cluster 2: 30 samples centered at [0, 10, 0, ...]

    Returns:
        Tuple of (embeddings: np.ndarray, true_labels: np.ndarray).
    """
    np.random.seed(42)
    n_dims = 768

    # Cluster 0: 40 samples
    cluster_0 = np.random.randn(40, n_dims) * 0.5 + np.array([10.0] + [0.0] * (n_dims - 1))

    # Cluster 1: 30 samples
    cluster_1 = np.random.randn(30, n_dims) * 0.5 + np.array([-10.0] + [0.0] * (n_dims - 1))

    # Cluster 2: 30 samples
    cluster_2 = np.random.randn(30, n_dims) * 0.5 + np.array([0.0, 10.0] + [0.0] * (n_dims - 2))

    embeddings = np.vstack([cluster_0, cluster_1, cluster_2]).astype(np.float32)
    true_labels = np.array([0] * 40 + [1] * 30 + [2] * 30)

    return embeddings, true_labels


@pytest.fixture
def precomputed_dino_embeddings():
    """Simulated DINOv2 embeddings for deterministic testing.

    Returns:
        Dict mapping image paths to embedding vectors (768-dim).
    """
    np.random.seed(42)
    embeddings_dict = {}
    for i in range(20):
        path = f"/tmp/img_{i}.jpg"
        embeddings_dict[path] = np.random.randn(768).astype(np.float32)

    return embeddings_dict


# ============================================================================
# VLM Annotation Fixtures
# ============================================================================

@pytest.fixture
def sample_vlm_annotations_color():
    """Sample VLM annotations for 'color' criterion.

    Returns:
        List of dicts with 'image' and 'class' keys (JSONL format).
    """
    return [
        {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "class": "warm_colors"},
        {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_1.JPEG", "class": "cool_colors"},
        {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_2.JPEG", "class": "warm_colors"},
        {"image": "/users/user1/FDS/tiny-imagenet-200/train/n02/images/img_3.JPEG", "class": "monochrome"},
        {"image": "/users/user1/FDS/tiny-imagenet-200/train/n02/images/img_4.JPEG", "class": "cool_colors"},
    ]


@pytest.fixture
def sample_vlm_annotations_all_criteria():
    """Sample VLM annotations for all 5 criteria.

    Returns:
        Dict mapping criterion name to list of annotation dicts.
    """
    base_images = [
        "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG",
        "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_1.JPEG",
        "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_2.JPEG",
        "/users/user1/FDS/tiny-imagenet-200/train/n02/images/img_3.JPEG",
        "/users/user1/FDS/tiny-imagenet-200/train/n02/images/img_4.JPEG",
    ]

    return {
        "color": [
            {"image": base_images[0], "class": "warm_colors"},
            {"image": base_images[1], "class": "cool_colors"},
            {"image": base_images[2], "class": "warm_colors"},
            {"image": base_images[3], "class": "monochrome"},
            {"image": base_images[4], "class": "cool_colors"},
        ],
        "main_object": [
            {"image": base_images[0], "class": "animal"},
            {"image": base_images[1], "class": "vehicle"},
            {"image": base_images[2], "class": "animal"},
            {"image": base_images[3], "class": "building"},
            {"image": base_images[4], "class": "person"},
        ],
        "size": [
            {"image": base_images[0], "class": "close_up"},
            {"image": base_images[1], "class": "wide_angle"},
            {"image": base_images[2], "class": "medium_shot"},
            {"image": base_images[3], "class": "wide_angle"},
            {"image": base_images[4], "class": "close_up"},
        ],
        "location": [
            {"image": base_images[0], "class": "outdoor"},
            {"image": base_images[1], "class": "urban"},
            {"image": base_images[2], "class": "natural"},
            {"image": base_images[3], "class": "indoor"},
            {"image": base_images[4], "class": "outdoor"},
        ],
        "time": [
            {"image": base_images[0], "class": "day"},
            {"image": base_images[1], "class": "night"},
            {"image": base_images[2], "class": "day"},
            {"image": base_images[3], "class": "day"},
            {"image": base_images[4], "class": "dusk"},
        ],
    }


@pytest.fixture
def vlm_annotation_directory(tmp_path, sample_vlm_annotations_all_criteria):
    """Create temporary VLM annotation directory structure.

    Mimics partitioning/vlm/data/tiny_imagenet/ structure with JSONL files.

    Args:
        tmp_path: Pytest temporary directory fixture.
        sample_vlm_annotations_all_criteria: Annotation data fixture.

    Returns:
        Path to the root annotation directory.
    """
    vlm_dir = tmp_path / "vlm_annotations"

    criterion_to_dir = {
        "color": "output_color_train",
        "main_object": "output_main_object_train",
        "size": "output_size_train",
        "location": "output_location_train",
        "time": "output_time_train",
    }

    for criterion, dirname in criterion_to_dir.items():
        criterion_dir = vlm_dir / dirname
        criterion_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = criterion_dir / "step3_result.jsonl"
        annotations = sample_vlm_annotations_all_criteria[criterion]

        # Write as JSONL (one JSON object per line)
        with open(jsonl_path, 'w') as f:
            for entry in annotations:
                f.write(json.dumps(entry) + '\n')

    return vlm_dir


# ============================================================================
# Cache Fixtures
# ============================================================================

@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary cache directory for testing CacheManager.

    Automatically cleaned up after test.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to temporary cache directory.
    """
    cache_dir = tmp_path / ".embedding_cache"
    cache_dir.mkdir()
    yield cache_dir
    # Cleanup handled by tmp_path


@pytest.fixture
def cache_manager(tmp_cache_dir):
    """CacheManager instance with temporary directory.

    Args:
        tmp_cache_dir: Temporary cache directory fixture.

    Returns:
        CacheManager instance configured with temp directory.
    """
    from embedding.cache_manager import CacheManager
    return CacheManager(cache_dir=tmp_cache_dir, enabled=True)


@pytest.fixture
def disabled_cache_manager(tmp_cache_dir):
    """CacheManager instance with caching disabled.

    Useful for testing cache-disabled code paths.

    Args:
        tmp_cache_dir: Temporary cache directory fixture.

    Returns:
        CacheManager instance with enabled=False.
    """
    from embedding.cache_manager import CacheManager
    return CacheManager(cache_dir=tmp_cache_dir, enabled=False)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_embedding_config():
    """Fast EmbeddingConfig for testing.

    Uses 'fast' preset with small batch size for quick tests.

    Returns:
        EmbeddingConfig instance.
    """
    from partitioning.embedding_utils.config import get_preset_config
    return get_preset_config("fast", n_clients=10)


@pytest.fixture
def all_embedding_configs():
    """All preset embedding configs for parameterized testing.

    Returns:
        List of EmbeddingConfig instances: [default, fast, high_quality, spectral].
    """
    from partitioning.embedding_utils.config import get_preset_config
    return [
        get_preset_config("default", n_clients=10),
        get_preset_config("fast", n_clients=10),
        get_preset_config("high_quality", n_clients=10),
        get_preset_config("spectral", n_clients=10),
    ]


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config):
    """Register custom markers for test organization."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (fast, no I/O)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (cross-component workflows)"
    )
    config.addinivalue_line(
        "markers",
        "property: marks property-based tests using Hypothesis"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"' on CPU-only)"
    )
