"""
Unit tests for embedding extractors.

Tests verify correct behavior of extractors without testing internal model logic.
Run with: pytest tests/test_extractors.py -v
Mark slow tests with: pytest tests/test_extractors.py -v -m "not slow"
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding import (
    ModelRegistry,
    EmbeddingExtractor,
    InceptionExtractor,
    DINOv2Extractor,
    CLIPExtractor,
    SigLIPExtractor,
    MAEExtractor,
    RawPixelResizerExtractor,
    SparseRandomProjectionResizerExtractor,
    UntrainedAutoEncoderExtractor,
    WaveletScatteringExtractor,
    CacheManager,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_images(tmp_path):
    """Create sample RGB images for testing."""
    images = []
    for i in range(5):
        # Create small images with different colors
        img = Image.new('RGB', (64, 64), color=(i*50, i*50, i*50))
        path = tmp_path / f"img_{i}.jpg"
        img.save(path)
        images.append(str(path))
    return images


@pytest.fixture
def cache_manager(tmp_path):
    """Create cache manager for testing."""
    return CacheManager(cache_dir=str(tmp_path / ".cache"), enabled=True)


@pytest.fixture
def disabled_cache():
    """Cache manager with caching disabled."""
    return CacheManager(cache_dir="./tmp_cache", enabled=False)


# ============================================================================
# Test ModelRegistry
# ============================================================================

class TestModelRegistry:
    """Test ModelRegistry factory pattern."""

    def test_list_models(self):
        """Test that available models can be listed."""
        models = ModelRegistry.list_models()

        assert isinstance(models, list)
        assert len(models) > 0

        # Check for known models
        expected_models = [
            "inception_v3",
            "dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant",
            "clip-vit-b-32", "clip-vit-b-16", "clip-vit-l-14",
            "siglip-base-patch16-224", "siglip-base-patch16-384",
            "mae-base", "mae-large", "mae-huge",
            "raw_pixel",
            "sparse_random_projection",
            "untrained_autoencoder",
            "wavelet_scattering",
        ]

        for model in expected_models:
            assert model in models, f"Expected model '{model}' not in registry"

    def test_create_known_model(self, cache_manager):
        """Test creating a known extractor."""
        # Use InceptionV3 as it's commonly available
        extractor = ModelRegistry.create(
            "inception_v3",
            batch_size=16,
            cache_manager=cache_manager
        )

        assert extractor is not None
        assert isinstance(extractor, InceptionExtractor)
        assert extractor.batch_size == 16
        assert extractor.cache_manager == cache_manager

    def test_create_unknown_model(self, cache_manager):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.create("nonexistent_model", cache_manager=cache_manager)

    def test_create_with_device(self, cache_manager):
        """Test creating extractor with device specification."""
        extractor = ModelRegistry.create(
            "inception_v3",
            device="cpu",
            cache_manager=cache_manager
        )

        assert extractor.device == "cpu"

    def test_create_multiple_extractors(self, cache_manager):
        """Test creating multiple extractors."""
        extractor1 = ModelRegistry.create("inception_v3", cache_manager=cache_manager)
        extractor2 = ModelRegistry.create("raw_pixel", cache_manager=cache_manager)

        assert isinstance(extractor1, InceptionExtractor)
        assert isinstance(extractor2, RawPixelResizerExtractor)
        assert extractor1 != extractor2


# ============================================================================
# Test InceptionExtractor
# ============================================================================

@pytest.mark.slow  # Requires model download
class TestInceptionExtractor:
    """Test InceptionV3 extractor."""

    def test_extract_embeddings(self, sample_images, cache_manager):
        """Test basic embedding extraction."""
        extractor = InceptionExtractor(
            batch_size=2,
            cache_manager=cache_manager,
            device="cpu"
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        # Verify output shape
        assert embeddings.shape == (5, 2048), f"Expected (5, 2048), got {embeddings.shape}"
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64

    def test_caching(self, sample_images, cache_manager):
        """Test that caching works correctly."""
        extractor = InceptionExtractor(
            cache_manager=cache_manager,
            device="cpu"
        )

        # First call - should compute
        emb1 = extractor.extract_embeddings(sample_images, use_cache=True)

        # Second call - should load from cache
        emb2 = extractor.extract_embeddings(sample_images, use_cache=True)

        # Should be identical
        np.testing.assert_array_equal(emb1, emb2)

    def test_cache_disabled(self, sample_images, disabled_cache):
        """Test extraction with caching disabled."""
        extractor = InceptionExtractor(
            cache_manager=disabled_cache,
            device="cpu"
        )

        emb = extractor.extract_embeddings(sample_images, use_cache=True)

        # Should still work, just not use cache
        assert emb.shape == (5, 2048)

    def test_reproducibility(self, sample_images, cache_manager):
        """Test that extraction is reproducible."""
        extractor1 = InceptionExtractor(cache_manager=cache_manager, device="cpu")
        extractor2 = InceptionExtractor(cache_manager=cache_manager, device="cpu")

        emb1 = extractor1.extract_embeddings(sample_images, use_cache=False)
        emb2 = extractor2.extract_embeddings(sample_images, use_cache=False)

        # Should produce same embeddings (within floating point precision)
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5, atol=1e-7)

    def test_batch_processing(self, sample_images, cache_manager):
        """Test batched processing."""
        extractor = InceptionExtractor(
            batch_size=2,  # Process 2 images at a time
            cache_manager=cache_manager,
            device="cpu"
        )

        emb = extractor.extract_embeddings(sample_images, use_cache=False)

        assert emb.shape[0] == len(sample_images)

    def test_single_image(self, sample_images, cache_manager):
        """Test extracting embedding for single image."""
        extractor = InceptionExtractor(cache_manager=cache_manager, device="cpu")

        emb = extractor.extract_embeddings([sample_images[0]], use_cache=False)

        assert emb.shape == (1, 2048)


# ============================================================================
# Test DINOv2Extractor
# ============================================================================

@pytest.mark.slow
class TestDINOv2Extractor:
    """Test DINOv2 extractor."""

    @pytest.mark.parametrize("variant", ["small", "base", "large", "giant"])
    def test_extract_embeddings_variants(self, sample_images, cache_manager, variant):
        """Test extraction with different DINOv2 variants."""
        extractor = DINOv2Extractor(
            variant=variant,
            batch_size=2,
            cache_manager=cache_manager,
            device="cpu"
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        # Verify shape
        assert embeddings.shape[0] == 5

        # Check embedding dimension based on variant
        expected_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536
        }
        assert embeddings.shape[1] == expected_dims[variant]

    def test_return_cls_token(self, sample_images, cache_manager):
        """Test returning CLS token."""
        extractor = DINOv2Extractor(
            variant="base",
            cache_manager=cache_manager,
            device="cpu"
        )

        embeddings = extractor.extract_embeddings(
            sample_images,
            use_cache=False,
            return_all_tokens=False
        )

        assert embeddings.shape == (5, 768)

    def test_return_all_tokens(self, sample_images, cache_manager):
        """Test returning all tokens."""
        extractor = DINOv2Extractor(
            variant="base",
            cache_manager=cache_manager,
            device="cpu"
        )

        embeddings = extractor.extract_embeddings(
            sample_images,
            use_cache=False,
            return_all_tokens=True
        )

        # Should return sequence length > 1 (CLS + patch tokens)
        assert embeddings.ndim == 3
        assert embeddings.shape[0] == 5
        assert embeddings.shape[2] == 768


# ============================================================================
# Test CLIPExtractor
# ============================================================================

@pytest.mark.slow
class TestCLIPExtractor:
    """Test CLIP extractor."""

    @pytest.mark.parametrize("variant", ["vit-b-32", "vit-b-16", "vit-l-14"])
    def test_extract_embeddings_variants(self, sample_images, cache_manager, variant):
        """Test extraction with different CLIP variants."""
        extractor = CLIPExtractor(
            variant=variant,
            batch_size=2,
            cache_manager=cache_manager,
            device="cpu"
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        assert embeddings.shape[0] == 5
        # CLIP embeddings are normalized to 512 or 768 dims depending on model
        assert embeddings.shape[1] in [512, 768]

    def test_caching(self, sample_images, cache_manager):
        """Test caching with CLIP."""
        extractor = CLIPExtractor(
            variant="vit-b-32",
            cache_manager=cache_manager,
            device="cpu"
        )

        emb1 = extractor.extract_embeddings(sample_images, use_cache=True)
        emb2 = extractor.extract_embeddings(sample_images, use_cache=True)

        np.testing.assert_array_equal(emb1, emb2)


# ============================================================================
# Test SigLIPExtractor
# ============================================================================

@pytest.mark.slow
class TestSigLIPExtractor:
    """Test SigLIP extractor."""

    def test_extract_embeddings(self, sample_images, cache_manager):
        """Test basic SigLIP extraction."""
        extractor = SigLIPExtractor(
            variant="base-patch16-224",
            batch_size=2,
            cache_manager=cache_manager,
            device="cpu"
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] == 768  # SigLIP base dimension


# ============================================================================
# Test MAEExtractor
# ============================================================================

@pytest.mark.slow
class TestMAEExtractor:
    """Test MAE extractor."""

    @pytest.mark.parametrize("variant", ["base", "large", "huge"])
    def test_extract_embeddings_variants(self, sample_images, cache_manager, variant):
        """Test extraction with different MAE variants."""
        extractor = MAEExtractor(
            variant=variant,
            batch_size=2,
            cache_manager=cache_manager,
            device="cpu"
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        assert embeddings.shape[0] == 5

        # Check embedding dimensions
        expected_dims = {
            "base": 768,
            "large": 1024,
            "huge": 1280
        }
        assert embeddings.shape[1] == expected_dims[variant]


# ============================================================================
# Test Untrained/Alternative Extractors
# ============================================================================

class TestRawPixelExtractor:
    """Test raw pixel extractor."""

    def test_extract_embeddings(self, sample_images, cache_manager):
        """Test raw pixel extraction."""
        extractor = RawPixelResizerExtractor(
            resize_dim=32,  # Resize to 32x32
            cache_manager=cache_manager
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        # 32x32 RGB = 3072 dimensions
        assert embeddings.shape == (5, 3072)
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64

    def test_different_resize_dims(self, sample_images, cache_manager):
        """Test with different resize dimensions."""
        extractor = RawPixelResizerExtractor(
            resize_dim=16,
            cache_manager=cache_manager
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        # 16x16 RGB = 768 dimensions
        assert embeddings.shape == (5, 768)

    def test_reproducibility(self, sample_images, cache_manager):
        """Test reproducibility."""
        extractor1 = RawPixelResizerExtractor(resize_dim=32, cache_manager=cache_manager)
        extractor2 = RawPixelResizerExtractor(resize_dim=32, cache_manager=cache_manager)

        emb1 = extractor1.extract_embeddings(sample_images, use_cache=False)
        emb2 = extractor2.extract_embeddings(sample_images, use_cache=False)

        np.testing.assert_array_equal(emb1, emb2)


class TestSparseRandomProjectionExtractor:
    """Test sparse random projection extractor."""

    def test_extract_embeddings(self, sample_images, cache_manager):
        """Test sparse random projection."""
        extractor = SparseRandomProjectionResizerExtractor(
            resize_dim=32,
            output_dim=512,
            cache_manager=cache_manager,
            seed=42
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        assert embeddings.shape == (5, 512)

    def test_reproducibility_with_seed(self, sample_images, cache_manager):
        """Test reproducibility with same seed."""
        extractor1 = SparseRandomProjectionResizerExtractor(
            resize_dim=32,
            output_dim=512,
            cache_manager=cache_manager,
            seed=42
        )
        extractor2 = SparseRandomProjectionResizerExtractor(
            resize_dim=32,
            output_dim=512,
            cache_manager=cache_manager,
            seed=42
        )

        emb1 = extractor1.extract_embeddings(sample_images, use_cache=False)
        emb2 = extractor2.extract_embeddings(sample_images, use_cache=False)

        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_different_seeds(self, sample_images, cache_manager):
        """Test that different seeds produce different projections."""
        extractor1 = SparseRandomProjectionResizerExtractor(
            resize_dim=32,
            output_dim=512,
            cache_manager=cache_manager,
            seed=42
        )
        extractor2 = SparseRandomProjectionResizerExtractor(
            resize_dim=32,
            output_dim=512,
            cache_manager=cache_manager,
            seed=123
        )

        emb1 = extractor1.extract_embeddings(sample_images, use_cache=False)
        emb2 = extractor2.extract_embeddings(sample_images, use_cache=False)

        # Should be different
        assert not np.allclose(emb1, emb2)


class TestUntrainedAutoEncoderExtractor:
    """Test untrained autoencoder extractor."""

    def test_extract_embeddings(self, sample_images, cache_manager):
        """Test untrained autoencoder extraction."""
        extractor = UntrainedAutoEncoderExtractor(
            latent_dim=256,
            cache_manager=cache_manager,
            seed=42
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        assert embeddings.shape == (5, 256)

    def test_reproducibility(self, sample_images, cache_manager):
        """Test reproducibility with same seed."""
        extractor1 = UntrainedAutoEncoderExtractor(
            latent_dim=256,
            cache_manager=cache_manager,
            seed=42
        )
        extractor2 = UntrainedAutoEncoderExtractor(
            latent_dim=256,
            cache_manager=cache_manager,
            seed=42
        )

        emb1 = extractor1.extract_embeddings(sample_images, use_cache=False)
        emb2 = extractor2.extract_embeddings(sample_images, use_cache=False)

        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)


class TestWaveletScatteringExtractor:
    """Test wavelet scattering extractor."""

    def test_extract_embeddings(self, sample_images, cache_manager):
        """Test wavelet scattering extraction."""
        extractor = WaveletScatteringExtractor(
            J=2,  # Scale
            cache_manager=cache_manager
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] > 0  # Dimension depends on J and image size

    def test_reproducibility(self, sample_images, cache_manager):
        """Test reproducibility."""
        extractor1 = WaveletScatteringExtractor(J=2, cache_manager=cache_manager)
        extractor2 = WaveletScatteringExtractor(J=2, cache_manager=cache_manager)

        emb1 = extractor1.extract_embeddings(sample_images, use_cache=False)
        emb2 = extractor2.extract_embeddings(sample_images, use_cache=False)

        np.testing.assert_array_almost_equal(emb1, emb2)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestExtractorEdgeCases:
    """Test edge cases for extractors."""

    def test_empty_image_list(self, cache_manager):
        """Test extraction with empty image list."""
        extractor = RawPixelResizerExtractor(cache_manager=cache_manager)

        with pytest.raises((ValueError, IndexError, AssertionError)):
            extractor.extract_embeddings([], use_cache=False)

    def test_single_image_extraction(self, sample_images, cache_manager):
        """Test extraction with single image."""
        extractor = RawPixelResizerExtractor(cache_manager=cache_manager)

        embeddings = extractor.extract_embeddings([sample_images[0]], use_cache=False)

        assert embeddings.shape[0] == 1

    def test_large_batch(self, sample_images, cache_manager):
        """Test extraction with large batch size."""
        extractor = RawPixelResizerExtractor(
            cache_manager=cache_manager,
            batch_size=100  # Larger than number of images
        )

        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)

        assert embeddings.shape[0] == len(sample_images)

    def test_grayscale_conversion(self, tmp_path, cache_manager):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale image
        img = Image.new('L', (64, 64), color=128)
        path = tmp_path / "gray.jpg"
        img.save(path)

        extractor = RawPixelResizerExtractor(
            resize_dim=32,
            cache_manager=cache_manager
        )

        embeddings = extractor.extract_embeddings([str(path)], use_cache=False)

        # Should still produce RGB dimensions (3072 = 32*32*3)
        assert embeddings.shape == (1, 3072)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
