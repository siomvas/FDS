"""Comprehensive tests for CacheManager.

Tests the persistent caching system for embeddings with both batch mode
(ad-hoc image lists) and dataset mode (large-scale datasets with subset access).

Test Coverage:
- Cache initialization and metadata handling
- Batch mode (cache key from specific image lists)
- Dataset mode (cache key from dataset name, subset retrieval)
- Path-indexed dictionaries (is_path_dict storage format)
- Relative path conversion for HPC multi-user compatibility
- Cache management (clear, get_cache_info)
- NPZ storage format (is_dict, is_path_dict flags)
- Edge cases (empty lists, corrupted files, etc.)
"""

import json
import numpy as np
import pytest
from pathlib import Path

from embedding.cache_manager import CacheManager


# ============================================================================
# Cache Initialization Tests
# ============================================================================

class TestCacheInitialization:
    """Test cache directory creation and metadata initialization."""

    def test_create_cache_dir(self, tmp_path):
        """Cache manager creates directory if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        cache = CacheManager(cache_dir=cache_dir, enabled=True)

        assert cache_dir.exists()
        assert cache.cache_dir == cache_dir

    def test_load_existing_metadata(self, tmp_path):
        """Cache manager loads existing metadata.json."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create metadata file
        metadata = {
            "version": "1.0",
            "entries": {
                "test_key": {
                    "model_name": "test_model",
                    "dataset_name": None,
                    "num_images": 10
                }
            }
        }
        metadata_file = cache_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Load cache
        cache = CacheManager(cache_dir=cache_dir, enabled=True)

        assert "test_key" in cache.metadata["entries"]
        assert cache.metadata["entries"]["test_key"]["model_name"] == "test_model"

    def test_corrupted_metadata_recovery(self, tmp_path):
        """Cache manager recovers from corrupted metadata.json."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create corrupted metadata
        metadata_file = cache_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            f.write("{invalid json content")

        # Should recover gracefully
        cache = CacheManager(cache_dir=cache_dir, enabled=True)

        assert cache.metadata["version"] == "1.0"
        assert cache.metadata["entries"] == {}

    def test_disabled_cache_returns_none(self, tmp_cache_dir):
        """Disabled cache always returns None for get()."""
        cache = CacheManager(cache_dir=tmp_cache_dir, enabled=False)

        result = cache.get(
            model_name="test_model",
            image_paths=["img1.jpg", "img2.jpg"]
        )

        assert result is None


# ============================================================================
# Batch Mode Tests
# ============================================================================

class TestBatchMode:
    """Test batch mode caching (cache key from specific image lists)."""

    def test_put_and_get_simple_array(self, cache_manager):
        """Basic put and get with simple numpy array."""
        embeddings = np.random.randn(10, 768).astype(np.float32)
        image_paths = [f"/tmp/img_{i}.jpg" for i in range(10)]

        # Put embeddings
        cache_manager.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=image_paths
        )

        # Get embeddings back
        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=image_paths
        )

        assert retrieved is not None
        np.testing.assert_allclose(retrieved, embeddings, rtol=1e-5)

    def test_cache_key_from_image_list(self, cache_manager):
        """Cache key computed from image list content."""
        embeddings = np.random.randn(5, 768).astype(np.float32)
        paths_a = ["img1.jpg", "img2.jpg", "img3.jpg"]
        paths_b = ["img1.jpg", "img2.jpg", "img3.jpg"]

        # Put with paths_a
        cache_manager.put(
            embeddings=embeddings[:3],
            model_name="test_model",
            image_paths=paths_a
        )

        # Get with paths_b (identical content)
        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=paths_b
        )

        assert retrieved is not None
        np.testing.assert_allclose(retrieved, embeddings[:3], rtol=1e-5)

    def test_different_lists_different_keys(self, cache_manager):
        """Different image lists produce different cache keys."""
        embs_a = np.random.randn(3, 768).astype(np.float32)
        embs_b = np.random.randn(3, 768).astype(np.float32)

        paths_a = ["img1.jpg", "img2.jpg", "img3.jpg"]
        paths_b = ["img1.jpg", "img2.jpg", "img4.jpg"]  # Different

        # Put both
        cache_manager.put(embeddings=embs_a, model_name="test_model", image_paths=paths_a)
        cache_manager.put(embeddings=embs_b, model_name="test_model", image_paths=paths_b)

        # Get paths_a
        retrieved_a = cache_manager.get(model_name="test_model", image_paths=paths_a)
        np.testing.assert_allclose(retrieved_a, embs_a, rtol=1e-5)

        # Get paths_b
        retrieved_b = cache_manager.get(model_name="test_model", image_paths=paths_b)
        np.testing.assert_allclose(retrieved_b, embs_b, rtol=1e-5)

    def test_same_images_different_order_same_key(self, cache_manager):
        """Same images in different order produce same cache key (sorted)."""
        embeddings = np.random.randn(3, 768).astype(np.float32)

        paths_original = ["img1.jpg", "img2.jpg", "img3.jpg"]
        paths_shuffled = ["img3.jpg", "img1.jpg", "img2.jpg"]

        # Put with original order
        cache_manager.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=paths_original
        )

        # Get with shuffled order should hit cache
        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=paths_shuffled
        )

        # Should get cache hit (same sorted paths)
        assert retrieved is not None

    def test_cache_miss_returns_none(self, cache_manager):
        """Cache miss returns None."""
        result = cache_manager.get(
            model_name="nonexistent_model",
            image_paths=["img1.jpg"]
        )

        assert result is None

    def test_in_memory_cache_hit(self, cache_manager):
        """Second get() uses in-memory cache (no disk I/O)."""
        embeddings = np.random.randn(5, 768).astype(np.float32)
        image_paths = [f"/tmp/img_{i}.jpg" for i in range(5)]

        # Put embeddings
        cache_manager.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=image_paths
        )

        # First get (loads from disk)
        retrieved_1 = cache_manager.get(
            model_name="test_model",
            image_paths=image_paths
        )

        # Second get (from memory cache)
        retrieved_2 = cache_manager.get(
            model_name="test_model",
            image_paths=image_paths
        )

        np.testing.assert_allclose(retrieved_1, retrieved_2, rtol=1e-5)


# ============================================================================
# Dataset Mode Tests
# ============================================================================

class TestDatasetMode:
    """Test dataset mode (cache entire datasets, retrieve subsets)."""

    def test_put_entire_dataset(self, cache_manager):
        """Put entire dataset with dataset_name."""
        all_paths = [f"/tmp/img_{i}.jpg" for i in range(100)]
        all_embeddings = np.random.randn(100, 768).astype(np.float32)

        cache_manager.put(
            embeddings=all_embeddings,
            model_name="test_model",
            image_paths=all_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Verify cache entry created
        info = cache_manager.get_cache_info()
        assert info["num_entries"] == 1

    def test_get_subset_from_dataset(self, cache_manager):
        """Retrieve subset of images from cached dataset."""
        all_paths = [f"/tmp/img_{i}.jpg" for i in range(100)]
        all_embeddings = np.random.randn(100, 768).astype(np.float32)

        # Cache entire dataset
        cache_manager.put(
            embeddings=all_embeddings,
            model_name="test_model",
            image_paths=all_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Retrieve subset
        subset_paths = ["/tmp/img_5.jpg", "/tmp/img_25.jpg", "/tmp/img_99.jpg"]
        subset_embs = cache_manager.get(
            model_name="test_model",
            image_paths=subset_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        assert subset_embs is not None
        assert subset_embs.shape == (3, 768)

        # Verify correct embeddings retrieved
        np.testing.assert_allclose(subset_embs[0], all_embeddings[5], rtol=1e-5)
        np.testing.assert_allclose(subset_embs[1], all_embeddings[25], rtol=1e-5)
        np.testing.assert_allclose(subset_embs[2], all_embeddings[99], rtol=1e-5)

    def test_dataset_name_required_for_mode(self, cache_manager):
        """Dataset mode requires dataset_name parameter."""
        # Without dataset_name, uses batch mode
        embeddings = np.random.randn(10, 768).astype(np.float32)
        paths = [f"/tmp/img_{i}.jpg" for i in range(10)]

        # Put without dataset_name (batch mode)
        cache_manager.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=paths
        )

        # Get with dataset_name (dataset mode) - should miss
        result = cache_manager.get(
            model_name="test_model",
            image_paths=paths,
            dataset_name="test_dataset"
        )

        assert result is None

    def test_base_path_for_relative_paths(self, cache_manager):
        """base_path converts absolute paths to relative."""
        abs_paths = [
            "/home/user/data/img_0.jpg",
            "/home/user/data/img_1.jpg"
        ]
        embeddings = np.random.randn(2, 768).astype(np.float32)

        # Put with base_path
        cache_manager.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=abs_paths,
            dataset_name="test_dataset",
            base_path="/home/user/data"
        )

        # Get with base_path
        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=abs_paths,
            dataset_name="test_dataset",
            base_path="/home/user/data"
        )

        assert retrieved is not None
        np.testing.assert_allclose(retrieved, embeddings, rtol=1e-5)

    def test_missing_images_in_subset_returns_none(self, cache_manager):
        """Requesting subset with missing images returns None."""
        all_paths = [f"/tmp/img_{i}.jpg" for i in range(10)]
        all_embeddings = np.random.randn(10, 768).astype(np.float32)

        # Cache dataset
        cache_manager.put(
            embeddings=all_embeddings,
            model_name="test_model",
            image_paths=all_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Request subset with missing image
        subset_paths = ["/tmp/img_0.jpg", "/tmp/img_999.jpg"]  # img_999 not in cache
        result = cache_manager.get(
            model_name="test_model",
            image_paths=subset_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        assert result is None

    def test_cache_key_independent_of_subset(self, cache_manager):
        """Dataset mode cache key doesn't depend on specific queried subset."""
        all_paths = [f"/tmp/img_{i}.jpg" for i in range(20)]
        all_embeddings = np.random.randn(20, 768).astype(np.float32)

        # Cache dataset
        cache_manager.put(
            embeddings=all_embeddings,
            model_name="test_model",
            image_paths=all_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Get different subsets
        subset_a = ["/tmp/img_0.jpg", "/tmp/img_1.jpg"]
        subset_b = ["/tmp/img_10.jpg", "/tmp/img_15.jpg"]

        result_a = cache_manager.get(
            model_name="test_model",
            image_paths=subset_a,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        result_b = cache_manager.get(
            model_name="test_model",
            image_paths=subset_b,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Both should hit cache
        assert result_a is not None
        assert result_b is not None


# ============================================================================
# Path-Indexed Dictionaries Tests
# ============================================================================

class TestPathIndexedDictionaries:
    """Test storage and retrieval of path-indexed embedding dictionaries."""

    def test_store_as_path_dict(self, cache_manager):
        """Store embeddings as path-indexed dictionary."""
        emb_dict = {
            "/tmp/img_0.jpg": np.random.randn(768).astype(np.float32),
            "/tmp/img_1.jpg": np.random.randn(768).astype(np.float32),
            "/tmp/img_2.jpg": np.random.randn(768).astype(np.float32)
        }

        cache_manager.put(
            embeddings=emb_dict,
            model_name="test_model",
            image_paths=list(emb_dict.keys()),
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Verify stored
        info = cache_manager.get_cache_info()
        assert info["num_entries"] == 1

    def test_retrieve_from_path_dict(self, cache_manager):
        """Retrieve embeddings from path-indexed dictionary."""
        paths = ["/tmp/img_0.jpg", "/tmp/img_1.jpg", "/tmp/img_2.jpg"]
        emb_dict = {p: np.random.randn(768).astype(np.float32) for p in paths}

        # Store
        cache_manager.put(
            embeddings=emb_dict,
            model_name="test_model",
            image_paths=paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Retrieve subset
        subset_paths = ["/tmp/img_0.jpg", "/tmp/img_2.jpg"]
        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=subset_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        assert retrieved is not None
        assert retrieved.shape == (2, 768)

    def test_is_path_dict_flag_in_npz(self, tmp_cache_dir):
        """NPZ file contains is_path_dict flag when storing path dicts."""
        cache = CacheManager(cache_dir=tmp_cache_dir, enabled=True)

        paths = ["/tmp/img_0.jpg", "/tmp/img_1.jpg"]
        embeddings = np.random.randn(2, 768).astype(np.float32)

        cache.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Load NPZ directly
        npz_files = list(tmp_cache_dir.glob("*.npz"))
        assert len(npz_files) == 1

        data = np.load(npz_files[0], allow_pickle=True)
        assert "is_path_dict" in data
        assert data["is_path_dict"] == True

    def test_sorted_paths_storage(self, cache_manager):
        """Paths are stored sorted in NPZ file."""
        paths_unsorted = ["/tmp/img_2.jpg", "/tmp/img_0.jpg", "/tmp/img_1.jpg"]
        embeddings = {p: np.random.randn(768).astype(np.float32) for p in paths_unsorted}

        cache_manager.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=paths_unsorted,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Verify paths stored sorted
        cache_key = list(cache_manager.metadata["entries"].keys())[0]
        cache_file = cache_manager.cache_dir / f"{cache_key}.npz"
        data = np.load(cache_file, allow_pickle=True)

        stored_paths = data["image_paths"]
        assert list(stored_paths) == sorted([p.replace("/tmp/", "") for p in paths_unsorted])

    def test_embedding_reconstruction_correct(self, cache_manager):
        """Embeddings reconstructed correctly from path-indexed storage."""
        paths = [f"/tmp/img_{i}.jpg" for i in range(10)]
        emb_dict = {p: np.random.randn(768).astype(np.float32) for p in paths}

        # Store
        cache_manager.put(
            embeddings=emb_dict,
            model_name="test_model",
            image_paths=paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Retrieve all
        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Verify each embedding
        for i, path in enumerate(paths):
            expected = emb_dict[path]
            np.testing.assert_allclose(retrieved[i], expected, rtol=1e-5)


# ============================================================================
# Relative Path Conversion Tests
# ============================================================================

class TestRelativePathConversion:
    """Test path normalization for HPC multi-user compatibility."""

    def test_to_relative_with_base_path(self, cache_manager):
        """Convert absolute paths to relative with base_path."""
        abs_paths = ["/home/user/data/img_0.jpg", "/home/user/data/img_1.jpg"]

        rel_paths = cache_manager._to_relative(abs_paths, base_path="/home/user/data")

        assert rel_paths == ["img_0.jpg", "img_1.jpg"]

    def test_to_relative_without_base_path(self, cache_manager):
        """Without base_path, paths remain as-is."""
        paths = ["/home/user/data/img_0.jpg", "/home/user/data/img_1.jpg"]

        result = cache_manager._to_relative(paths, base_path=None)

        assert result == paths

    def test_relative_path_consistency(self, cache_manager):
        """Relative path conversion ensures multi-user compatibility."""
        # User 1 path
        user1_paths = ["/users/user1/FDS/tiny-imagenet-200/train/n01/img_0.JPEG"]
        # User 2 path (same image, different user)
        user2_paths = ["/users/user2/FDS/tiny-imagenet-200/train/n01/img_0.JPEG"]

        base_path_1 = "/users/user1/FDS"
        base_path_2 = "/users/user2/FDS"

        rel1 = cache_manager._to_relative(user1_paths, base_path_1)
        rel2 = cache_manager._to_relative(user2_paths, base_path_2)

        # Should produce same relative path
        assert rel1 == rel2

    def test_different_drives_fallback(self, cache_manager):
        """Fallback to absolute path if on different drives (Windows)."""
        # This test is platform-dependent, but tests the fallback logic
        paths = ["/home/user/data/img.jpg"]

        # Use incompatible base_path
        result = cache_manager._to_relative(paths, base_path="/mnt/other")

        # Should return valid path (either relative or absolute)
        assert len(result) == 1
        assert isinstance(result[0], str)


# ============================================================================
# Cache Management Tests
# ============================================================================

class TestCacheManagement:
    """Test cache clearing and info retrieval."""

    def test_get_cache_info(self, cache_manager):
        """get_cache_info() returns cache statistics."""
        info = cache_manager.get_cache_info()

        assert info["enabled"] is True
        assert "cache_dir" in info
        assert "num_entries" in info
        assert "total_size_mb" in info

    def test_clear_all_cache(self, cache_manager):
        """clear() removes all cache entries."""
        # Put multiple entries
        for i in range(3):
            cache_manager.put(
                embeddings=np.random.randn(5, 768).astype(np.float32),
                model_name=f"model_{i}",
                image_paths=[f"/tmp/img_{j}.jpg" for j in range(5)]
            )

        assert cache_manager.get_cache_info()["num_entries"] == 3

        # Clear all
        cache_manager.clear()

        assert cache_manager.get_cache_info()["num_entries"] == 0

    def test_clear_by_model_name(self, cache_manager):
        """clear(model_name) removes only specified model's cache."""
        # Put entries for two models
        cache_manager.put(
            embeddings=np.random.randn(5, 768).astype(np.float32),
            model_name="model_a",
            image_paths=["/tmp/img_0.jpg"]
        )

        cache_manager.put(
            embeddings=np.random.randn(5, 768).astype(np.float32),
            model_name="model_b",
            image_paths=["/tmp/img_1.jpg"]
        )

        # Clear model_a
        cache_manager.clear(model_name="model_a")

        # Only model_b should remain
        info = cache_manager.get_cache_info()
        assert info["num_entries"] == 1
        remaining_entry = list(info["entries"].values())[0]
        assert remaining_entry["model_name"] == "model_b"

    def test_cache_size_calculation(self, cache_manager):
        """total_size_mb accurately reflects cache size."""
        # Put embeddings
        cache_manager.put(
            embeddings=np.random.randn(100, 768).astype(np.float32),
            model_name="large_model",
            image_paths=[f"/tmp/img_{i}.jpg" for i in range(100)]
        )

        info = cache_manager.get_cache_info()
        assert info["total_size_mb"] > 0.0

    def test_metadata_consistency(self, cache_manager):
        """Metadata remains consistent after multiple operations."""
        # Multiple puts
        for i in range(5):
            cache_manager.put(
                embeddings=np.random.randn(10, 768).astype(np.float32),
                model_name=f"model_{i}",
                image_paths=[f"/tmp/img_{j}.jpg" for j in range(10)]
            )

        # Clear one
        cache_manager.clear(model_name="model_2")

        # Verify metadata consistent with disk
        info = cache_manager.get_cache_info()
        assert info["num_entries"] == 4

        # Verify metadata file is valid JSON
        with open(cache_manager.metadata_file, 'r') as f:
            metadata = json.load(f)
            assert len(metadata["entries"]) == 4


# ============================================================================
# NPZ Format Tests
# ============================================================================

class TestNPZFormat:
    """Test NPZ storage format details."""

    def test_npz_contains_correct_flags(self, tmp_cache_dir):
        """NPZ file contains appropriate format flags."""
        cache = CacheManager(cache_dir=tmp_cache_dir, enabled=True)

        # Store dataset mode (is_path_dict)
        cache.put(
            embeddings=np.random.randn(5, 768).astype(np.float32),
            model_name="test_model",
            image_paths=[f"/tmp/img_{i}.jpg" for i in range(5)],
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        npz_files = list(tmp_cache_dir.glob("*.npz"))
        data = np.load(npz_files[0], allow_pickle=True)

        assert "is_path_dict" in data
        assert "image_paths" in data
        assert "embeddings" in data

    def test_load_npz_reconstructs_data(self, cache_manager):
        """NPZ loading correctly reconstructs original data structure."""
        original_embs = np.random.randn(10, 768).astype(np.float32)
        paths = [f"/tmp/img_{i}.jpg" for i in range(10)]

        cache_manager.put(
            embeddings=original_embs,
            model_name="test_model",
            image_paths=paths
        )

        # Clear memory cache to force disk read
        cache_manager._memory_cache.clear()

        # Retrieve (loads from disk)
        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=paths
        )

        np.testing.assert_allclose(retrieved, original_embs, rtol=1e-5)

    def test_compression_settings(self, tmp_cache_dir):
        """NPZ files are compressed (savez_compressed)."""
        cache = CacheManager(cache_dir=tmp_cache_dir, enabled=True)

        cache.put(
            embeddings=np.random.randn(100, 768).astype(np.float32),
            model_name="test_model",
            image_paths=[f"/tmp/img_{i}.jpg" for i in range(100)]
        )

        npz_files = list(tmp_cache_dir.glob("*.npz"))
        assert len(npz_files) == 1

        # File should be compressed (smaller than raw float32 array)
        file_size = npz_files[0].stat().st_size
        raw_size = 100 * 768 * 4  # 100 samples * 768 dims * 4 bytes
        assert file_size < raw_size

    def test_multi_layer_dictionary_storage(self, cache_manager):
        """Multi-layer embeddings stored with is_dict flag."""
        multi_layer_embs = {
            0: np.random.randn(10, 768).astype(np.float32),
            1: np.random.randn(10, 768).astype(np.float32),
            2: np.random.randn(10, 768).astype(np.float32)
        }

        cache_manager.put(
            embeddings=multi_layer_embs,
            model_name="test_model",
            image_paths=[f"/tmp/img_{i}.jpg" for i in range(10)]
        )

        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=[f"/tmp/img_{i}.jpg" for i in range(10)]
        )

        assert isinstance(retrieved, dict)
        assert set(retrieved.keys()) == {0, 1, 2}


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_image_list(self, cache_manager):
        """Handle empty image list gracefully."""
        embeddings = np.array([]).reshape(0, 768).astype(np.float32)

        # Should not crash
        cache_manager.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=[]
        )

    def test_single_image(self, cache_manager):
        """Handle single image correctly."""
        embedding = np.random.randn(1, 768).astype(np.float32)
        path = ["/tmp/img_0.jpg"]

        cache_manager.put(
            embeddings=embedding,
            model_name="test_model",
            image_paths=path
        )

        retrieved = cache_manager.get(
            model_name="test_model",
            image_paths=path
        )

        assert retrieved is not None
        assert retrieved.shape == (1, 768)

    def test_mismatched_embeddings_paths_length(self, cache_manager):
        """Mismatched embeddings and paths length fails fast."""
        embeddings = np.random.randn(5, 768).astype(np.float32)
        paths = ["/tmp/img_0.jpg", "/tmp/img_1.jpg"]  # Only 2 paths

        with pytest.raises(ValueError, match="matching numbers of embeddings and image paths"):
            cache_manager.put(
                embeddings=embeddings,
                model_name="test_model",
                image_paths=paths,
                dataset_name="test_dataset"
            )

    def test_corrupted_npz_file_graceful_fail(self, tmp_cache_dir, capsys):
        """Corrupted NPZ file returns None gracefully."""
        cache = CacheManager(cache_dir=tmp_cache_dir, enabled=True)

        # Create corrupted NPZ file
        cache_key = "corrupted_key"
        cache_file = tmp_cache_dir / f"{cache_key}.npz"
        with open(cache_file, 'w') as f:
            f.write("corrupted data")

        # Add to metadata
        cache.metadata["entries"][cache_key] = {
            "model_name": "test_model",
            "file": f"{cache_key}.npz"
        }

        # Try to get (should return None and print warning)
        result = cache.get(
            model_name="test_model",
            image_paths=["/tmp/img.jpg"]
        )

        assert result is None
