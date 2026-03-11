"""Tests for embedding extractors."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import List, Optional, Dict, Any, Type
import os
import tempfile


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_images(tmp_path) -> List[Dict[str, str]]:
    """Create sample RGB images of varying sizes and content for testing."""
    images = []
    for i in range(5):
        img = Image.new('RGB', (128, 128), color=(i * 50, 100 - i * 10, i * 30))
        path = tmp_path / f"img_{i}.jpg"
        img.save(path, quality=95)
        images.append(str(path))
    return images


@pytest.fixture
def small_images(tmp_path) -> List[Dict[str, str]]:
    """Create small 32x32 images for fast tests."""
    images = []
    for i in range(3):
        img = Image.new('RGB', (32, 32), color=(i * 80, i * 60, i * 40))
        path = tmp_path / f"small_img_{i}.jpg"
        img.save(path)
        images.append(str(path))
    return images


@pytest.fixture
def single_image(tmp_path) -> List[Dict[str, str]]:
    """Create a single image for single-image tests."""
    img = Image.new('RGB', (64, 64), color=(100, 100, 100))
    path = tmp_path / "single.jpg"
    img.save(path)
    return [str(path)]


@pytest.fixture
def grayscale_image(tmp_path) -> List[Dict[str, str]]:
    """Create a grayscale image for mode conversion testing."""
    img = Image.new('L', (64, 64), color=128)
    path = tmp_path / "gray.jpg"
    img.save(path)
    return [str(path)]


@pytest.fixture
def rgba_image(tmp_path) -> List[Dict[str, str]]:
    """Create an RGBA image for alpha channel handling."""
    img = Image.new('RGBA', (64, 64), color=(100, 150, 200, 255))
    path = tmp_path / "rgba.png"
    img.save(path)
    return [str(path)]


@pytest.fixture
def corrupt_image(tmp_path) -> List[Dict[str, str]]:
    """Create a corrupt/invalid image file."""
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not a valid image file content")
    return [str(path)]


@pytest.fixture
def tiny_image(tmp_path) -> List[Dict[str, str]]:
    """Create a 1x1 pixel image."""
    img = Image.new('RGB', (1, 1), color=(128, 128, 128))
    path = tmp_path / "tiny.jpg"
    img.save(path)
    return [str(path)]


@pytest.fixture
def large_image(tmp_path) -> List[Dict[str, str]]:
    """Create a large 2048x2048 image."""
    img = Image.new('RGB', (2048, 2048), color=(64, 128, 192))
    path = tmp_path / "large.jpg"
    img.save(path)
    return [str(path)]


@pytest.fixture
def large_images(tmp_path) -> List[Dict[str, str]]:
    """Create larger images for realistic testing."""
    images = []
    for i in range(2):
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, 512).reshape(1, -1)
        arr[:, :, 1] = np.linspace(0, 255, 512).reshape(-1, 1)
        arr[:, :, 2] = 128
        img = Image.fromarray(arr)
        path = tmp_path / f"large_img_{i}.jpg"
        img.save(path)
        images.append(str(path))
    return images


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager for testing cache behavior."""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)  # Default: cache miss
    cache.put = MagicMock()
    return cache


@pytest.fixture
def model_dir(tmp_path) -> Path:
    """Create a temporary model directory."""
    model_path = tmp_path / "models"
    model_path.mkdir()
    return model_path


# ============================================================================
# Test collate_fn Helper Function
# ============================================================================

class TestCollateFn:
    """Test the custom collate function for DataLoader."""

    def test_collate_valid_tensors(self):
        """Test collating a batch of valid tensors."""
        from embedding.embedding_extractors import collate_fn
        
        batch = [torch.randn(3, 64, 64) for _ in range(4)]
        result = collate_fn(batch)
        
        assert result is not None
        assert result.shape == (4, 3, 64, 64)

    def test_collate_with_none_values(self):
        """Test that None values (corrupt images) are filtered out."""
        from embedding.embedding_extractors import collate_fn
        
        batch = [torch.randn(3, 64, 64), None, torch.randn(3, 64, 64), None]
        result = collate_fn(batch)
        
        assert result is not None
        assert result.shape == (2, 3, 64, 64)

    def test_collate_all_none(self):
        """Test handling of batch where all images are corrupt."""
        from embedding.embedding_extractors import collate_fn
        
        batch = [None, None, None]
        result = collate_fn(batch)
        
        assert result is None

    def test_collate_empty_batch(self):
        """Test handling of empty batch."""
        from embedding.embedding_extractors import collate_fn
        
        result = collate_fn([])
        
        assert result is None

    def test_collate_single_tensor(self):
        """Test collating a single tensor."""
        from embedding.embedding_extractors import collate_fn
        
        batch = [torch.randn(3, 32, 32)]
        result = collate_fn(batch)
        
        assert result.shape == (1, 3, 32, 32)

    def test_collate_preserves_tensor_dtype(self):
        """Test that collate preserves tensor dtype."""
        from embedding.embedding_extractors import collate_fn
        
        batch = [torch.randn(3, 32, 32).half() for _ in range(2)]
        result = collate_fn(batch)
        
        assert result.dtype == torch.float16

    def test_collate_mixed_none_positions(self):
        """Test collating with None at various positions."""
        from embedding.embedding_extractors import collate_fn
        
        # None at start
        batch1 = [None, torch.randn(3, 32, 32), torch.randn(3, 32, 32)]
        assert collate_fn(batch1).shape == (2, 3, 32, 32)
        
        # None at end
        batch2 = [torch.randn(3, 32, 32), torch.randn(3, 32, 32), None]
        assert collate_fn(batch2).shape == (2, 3, 32, 32)
        
        # None in middle
        batch3 = [torch.randn(3, 32, 32), None, torch.randn(3, 32, 32)]
        assert collate_fn(batch3).shape == (2, 3, 32, 32)


# ============================================================================
# Test ModelRegistry Factory Pattern
# ============================================================================

class TestModelRegistry:
    """Test ModelRegistry factory pattern functionality."""

    def test_registry_contains_expected_families(self):
        """Test that all expected model families are registered."""
        from embedding.embedding_extractors import ModelRegistry
        
        expected_families = [
            "inception_v3", "dino", "dinov2", "clip", "siglip", "mae",
            "raw_pixel_resizer", "sparse_random_projection_resizer",
            "UAE", "wavelet_scattering"
        ]
        
        for family in expected_families:
            assert family in ModelRegistry._REGISTRY, \
                f"Model family '{family}' not registered"

    def test_registry_family_count(self):
        """Test that registry has expected number of families."""
        from embedding.embedding_extractors import ModelRegistry
        
        assert len(ModelRegistry._REGISTRY) >= 10

    def test_create_with_default_variant(self):
        """Test creating extractor without specifying variant."""
        from embedding.embedding_extractors import ModelRegistry, InceptionExtractor
        
        extractor = ModelRegistry.create("inception_v3", device="cuda")
        
        assert isinstance(extractor, InceptionExtractor)
        assert extractor.model_name == "inception_v3"

    @pytest.mark.parametrize("model_id", ["dino-base", "dinov2-base"])
    def test_create_with_explicit_variant(self, model_id):
        """Test creating extractor with explicit variant aliases."""
        from embedding.embedding_extractors import ModelRegistry, DINOv2Extractor
        
        extractor = ModelRegistry.create(model_id, device="cuda")
        
        assert isinstance(extractor, DINOv2Extractor)
        assert extractor.variant == "base"

    def test_create_unknown_family_raises(self):
        """Test that unknown model family raises ValueError."""
        from embedding.embedding_extractors import ModelRegistry
        
        with pytest.raises(ValueError, match="Unknown model family"):
            ModelRegistry.create("nonexistent_model")

    def test_create_passes_kwargs(self):
        """Test that kwargs are properly passed to extractor."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create(
            "inception_v3",
            device="cuda",
            batch_size=16,
            num_workers=2
        )
        
        assert extractor.batch_size == 16
        assert extractor.num_workers == 2

    def test_create_with_cache_manager(self, mock_cache_manager):
        """Test creating extractor with cache manager."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create(
            "inception_v3",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        assert extractor.cache_manager is mock_cache_manager

    def test_create_with_proxy(self):
        """Test creating extractor with proxy configuration."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create(
            "inception_v3",
            device="cuda",
            proxy="http://proxy.example.com:8080"
        )
        
        assert extractor.proxy == "http://proxy.example.com:8080"

    def test_create_with_model_dir(self, model_dir):
        """Test creating extractor with local model directory."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create(
            "inception_v3",
            device="cuda",
            model_dir=str(model_dir)
        )
        
        assert extractor.model_dir == model_dir

    def test_register_decorator(self):
        """Test the @register decorator adds classes to registry."""
        from embedding.embedding_extractors import ModelRegistry, EmbeddingExtractor
        
        @ModelRegistry.register("test_model")
        class TestExtractor(EmbeddingExtractor):
            def __init__(self, variant="default", **kwargs):
                super().__init__(model_name=f"test-{variant}", **kwargs)
            def load_model(self): pass
            def get_embedding_dim(self): return 128
        
        assert "test_model" in ModelRegistry._REGISTRY
        assert ModelRegistry._REGISTRY["test_model"] is TestExtractor
        
        # Cleanup
        del ModelRegistry._REGISTRY["test_model"]

    @pytest.mark.parametrize("model_id,expected_family,expected_variant", [
        ("inception_v3", "inception_v3", "default"),
        ("dino-small", "dino", "small"),
        ("dino-base", "dino", "base"),
        ("dinov2-small", "dinov2", "small"),
        ("dinov2-base", "dinov2", "base"),
        ("clip-base-patch16", "clip", "base-patch16"),
        ("siglip-base-patch16-224", "siglip", "base-patch16-224"),
        ("mae-large", "mae", "large"),
        ("raw_pixel_resizer-32", "raw_pixel_resizer", "32"),
        ("sparse_random_projection_resizer-64-512", "sparse_random_projection_resizer", "64-512"),
        ("UAE", "UAE", "default"),
        ("wavelet_scattering-128-3", "wavelet_scattering", "128-3"),
    ])
    def test_create_all_model_types(self, model_id, expected_family, expected_variant):
        """Test creating all registered model types."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create(model_id, device="cuda")
        assert extractor is not None


# ============================================================================
# Test EmbeddingExtractor Base Class
# ============================================================================

class TestEmbeddingExtractorBase:
    """Test base EmbeddingExtractor functionality."""

    def test_default_device_selection(self):
        """Test automatic device selection."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create("raw_pixel_resizer-32")
        
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert str(extractor.device).startswith(expected_device)

    def test_explicit_device_cpu(self):
        """Test explicit CPU device selection."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create("raw_pixel_resizer-32", device="cpu")
        
        assert str(extractor.device) == "cpu"

    def test_explicit_cuda_falls_back_to_cpu_with_warning(self, monkeypatch):
        """Explicit CUDA requests degrade cleanly when CUDA is unavailable."""
        from embedding.embedding_extractors import ModelRegistry

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.warns(RuntimeWarning, match="Falling back to CPU"):
            extractor = ModelRegistry.create("raw_pixel_resizer-32", device="cuda")

        assert str(extractor.device) == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_explicit_device_cuda(self):
        """Test explicit CUDA device selection."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create("raw_pixel_resizer-32", device="cuda")
        
        assert str(extractor.device).startswith("cuda")

    def test_proxy_setup_and_cleanup(self):
        """Test proxy environment variable handling."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create(
            "raw_pixel_resizer-32",
            device="cuda",
            proxy="http://test.proxy:3128"
        )
        
        # Test setup
        extractor._setup_proxy_for_download()
        assert os.environ.get('https_proxy') == "http://test.proxy:3128"
        
        # Test cleanup
        extractor._cleanup_proxy_after_download()
        assert 'https_proxy' not in os.environ

    def test_proxy_cleanup_when_not_set(self):
        """Test that cleanup doesn't fail when proxy wasn't set."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create("raw_pixel_resizer-32", device="cuda")
        
        # Should not raise even if https_proxy was never set
        extractor._cleanup_proxy_after_download()

    def test_get_dataloader_raises_without_transform(self, sample_images):
        """Test that _get_dataloader raises if transform not initialized."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create("raw_pixel_resizer-32", device="cuda")
        # Don't call load_model() - transform will be None
        
        with pytest.raises(RuntimeError, match="Processor or Transform not initialized"):
            extractor._get_dataloader(sample_images)

    def test_lazy_model_loading(self, sample_images):
        """Test that model is loaded lazily on first extraction."""
        from embedding.embedding_extractors import ModelRegistry
        
        extractor = ModelRegistry.create("raw_pixel_resizer-32", device="cuda")
        
        # Model should be None before extraction
        assert extractor.model is None
        assert extractor.transform is None
        
        # After extraction, transform should be set
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert extractor.transform is not None


# ============================================================================
# Test InceptionExtractor
# ============================================================================

@pytest.mark.slow
class TestInceptionExtractor:
    """Test InceptionV3 extractor."""

    def test_embedding_dimension(self):
        """Test that embedding dimension is 2048."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda")
        
        assert extractor.get_embedding_dim() == 2048

    def test_extract_embeddings_shape(self, sample_images, mock_cache_manager):
        """Test output shape of embeddings."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(
            device="cuda",
            batch_size=2,
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert embeddings.shape == (5, 2048)
        assert embeddings.dtype in (np.float32, np.float64)

    def test_extract_single_image(self, single_image, mock_cache_manager):
        """Test extraction for single image."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        
        assert embeddings.shape == (1, 2048)

    def test_reproducibility(self, sample_images, mock_cache_manager):
        """Test that extraction produces deterministic results."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        
        emb1 = extractor.extract_embeddings(sample_images, use_cache=False)
        emb2 = extractor.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5)

    def test_batch_independence(self, sample_images, mock_cache_manager):
        """Test that batch size doesn't affect results."""
        from embedding.embedding_extractors import InceptionExtractor

        ext_batch2 = InceptionExtractor(
            device="cpu", batch_size=2, cache_manager=mock_cache_manager
        )
        ext_batch4 = InceptionExtractor(
            device="cpu", batch_size=4, cache_manager=mock_cache_manager
        )
        
        emb1 = ext_batch2.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext_batch4.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_allclose(emb1, emb2, rtol=1e-3)

    def test_model_fc_replaced_with_identity(self, mock_cache_manager):
        """Test that fc layer is replaced with Identity for feature extraction."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        extractor.load_model()
        
        assert isinstance(extractor.model.fc, nn.Identity)

    def test_model_eval_mode(self, mock_cache_manager):
        """Test that model is in eval mode after loading."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        extractor.load_model()
        
        assert not extractor.model.training

    def test_cache_miss_triggers_computation(self, sample_images, mock_cache_manager):
        """Test cache miss behavior."""
        from embedding.embedding_extractors import InceptionExtractor
        
        mock_cache_manager.get.return_value = None
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()
        assert embeddings.shape == (5, 2048)

    def test_cache_hit_returns_cached(self, sample_images):
        """Test cache hit behavior."""
        from embedding.embedding_extractors import InceptionExtractor
        
        cached = np.random.randn(5, 2048).astype(np.float32)
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache)
        embeddings = extractor.extract_embeddings(sample_images, use_cache=True)
        
        np.testing.assert_array_equal(embeddings, cached)
        mock_cache.put.assert_not_called()

    def test_empty_image_list(self, mock_cache_manager):
        """Test empty image list returns empty array."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        extractor.load_model()
        
        embeddings = extractor.extract_embeddings([], use_cache=False)
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 0

    def test_grayscale_handling(self, grayscale_image, mock_cache_manager):
        """Test grayscale image handling."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        embeddings = extractor.extract_embeddings(grayscale_image, use_cache=False)
        
        assert embeddings.shape == (1, 2048)

    def test_rgba_handling(self, rgba_image, mock_cache_manager):
        """Test RGBA image handling."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        embeddings = extractor.extract_embeddings(rgba_image, use_cache=False)
        
        assert embeddings.shape == (1, 2048)

    def test_tiny_image_handling(self, tiny_image, mock_cache_manager):
        """Test tiny image handling."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        embeddings = extractor.extract_embeddings(tiny_image, use_cache=False)
        
        assert embeddings.shape == (1, 2048)

    def test_embeddings_finite(self, sample_images, mock_cache_manager):
        """Test that embeddings are finite."""
        from embedding.embedding_extractors import InceptionExtractor
        
        extractor = InceptionExtractor(device="cuda", cache_manager=mock_cache_manager)
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert np.all(np.isfinite(embeddings))


# ============================================================================
# Test DINOv2Extractor
# ============================================================================

@pytest.mark.slow
class TestDINOv2Extractor:
    """Test DINOv2 extractor with various configurations."""

    @pytest.mark.parametrize("variant,expected_dim", [
        ("small", 384),
        ("base", 768),
        ("large", 1024),
        ("giant", 1536),
    ])
    def test_embedding_dimensions(self, variant, expected_dim):
        """Test embedding dimension for each variant."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(variant=variant, device="cuda")
        
        assert extractor.get_embedding_dim() == expected_dim

    def test_invalid_variant_raises(self):
        """Test that invalid variant raises ValueError."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        with pytest.raises(ValueError, match="Unknown DINOv2 variant"):
            DINOv2Extractor(variant="xlarge", device="cuda")

    def test_extract_cls_token(self, small_images, mock_cache_manager):
        """Test extracting CLS token (default mode)."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(
            small_images,
            return_which_tokens="cls",
            use_cache=False
        )
        
        assert embeddings.shape == (3, 384)

    def test_extract_k_last_cls(self, small_images, mock_cache_manager):
        """Test extracting CLS from last k layers."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(
            small_images,
            return_which_tokens="k_last_cls",
            k_last=2,
            use_cache=False
        )
        
        assert embeddings.shape == (3, 768)

    def test_extract_k_last_all(self, small_images, mock_cache_manager):
        """Test extracting all tokens from last k layers."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(
            small_images,
            return_which_tokens="k_last_all",
            k_last=2,
            use_cache=False
        )
        
        # Shape: (batch, num_tokens, hidden_dim * k_last)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[2] == 384 * 2

    def test_extract_all_last(self, small_images, mock_cache_manager):
        """Test extracting all tokens from last layer."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(
            small_images,
            return_which_tokens="all_last",
            use_cache=False
        )
        
        # Shape: (batch, num_tokens, hidden_dim)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[2] == 384

    def test_invalid_token_mode_raises(self, small_images, mock_cache_manager):
        """Test that invalid token mode raises ValueError."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        with pytest.raises(ValueError, match="Invalid return_which_tokens"):
            extractor.extract_embeddings(
                small_images,
                return_which_tokens="invalid_mode",
                use_cache=False
            )

    def test_single_image(self, single_image, mock_cache_manager):
        """Test single image extraction."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        assert embeddings.shape == (1, 384)

    def test_reproducibility(self, small_images, mock_cache_manager):
        """Test reproducibility."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        emb1 = extractor.extract_embeddings(small_images, use_cache=False)
        emb2 = extractor.extract_embeddings(small_images, use_cache=False)
        
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5)

    def test_batch_independence(self, sample_images, mock_cache_manager):
        """Test batch independence."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        ext1 = DINOv2Extractor(
            variant="small", device="cpu", batch_size=2,
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        ext2 = DINOv2Extractor(
            variant="small", device="cpu", batch_size=5,
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_allclose(emb1, emb2, rtol=1e-3)

    def test_cache_behavior(self, small_images, mock_cache_manager):
        """Test cache miss/hit behavior."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        mock_cache_manager.get.return_value = None
        extractor = DINOv2Extractor(
            variant="small", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        extractor.extract_embeddings(small_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()

    def test_empty_list(self, mock_cache_manager):
        """Test empty list handling."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        extractor.load_model()
        
        embeddings = extractor.extract_embeddings([], use_cache=False)
        assert len(embeddings) == 0

    def test_embeddings_finite(self, small_images, mock_cache_manager):
        """Test embeddings are finite."""
        from embedding.embedding_extractors import DINOv2Extractor
        
        extractor = DINOv2Extractor(
            variant="small", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        assert np.all(np.isfinite(embeddings))


# ============================================================================
# Test CLIPExtractor
# ============================================================================

@pytest.mark.slow
class TestCLIPExtractor:
    """Test CLIP vision extractor."""

    @pytest.mark.parametrize("variant,expected_dim", [
        ("base-patch32", 768),
        ("base-patch16", 768),
        ("large-patch14", 1024),
    ])
    def test_embedding_dimensions(self, variant, expected_dim):
        """Test embedding dimension for each variant."""
        from embedding.embedding_extractors import CLIPExtractor
        
        extractor = CLIPExtractor(variant=variant, device="cuda", proxy='http://proxy.example:3128')
        
        assert extractor.get_embedding_dim() == expected_dim

    def test_invalid_variant_raises(self):
        """Test that invalid variant raises ValueError."""
        from embedding.embedding_extractors import CLIPExtractor
        
        with pytest.raises(ValueError, match="Unknown CLIP variant"):
            CLIPExtractor(variant="huge-patch8", device="cuda")

    def test_hf_model_path_mapping(self):
        """Test that HuggingFace model paths are correctly mapped."""
        from embedding.embedding_extractors import CLIPExtractor
        
        extractor = CLIPExtractor(variant="large-patch14", device="cuda", proxy='http://proxy.example:3128')
        
        assert extractor.hf_model_path == "openai/clip-vit-large-patch14"

    def test_extract_embeddings_shape(self, small_images, mock_cache_manager):
        """Test output shape."""
        from embedding.embedding_extractors import CLIPExtractor
        
        extractor = CLIPExtractor(
            variant="base-patch32",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        assert embeddings.shape == (3, 768)

    def test_single_image(self, single_image, mock_cache_manager):
        """Test single image extraction."""
        from embedding.embedding_extractors import CLIPExtractor
        
        extractor = CLIPExtractor(
            variant="base-patch32",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        assert embeddings.shape == (1, 768)

    def test_reproducibility(self, small_images, mock_cache_manager):
        """Test reproducibility."""
        from embedding.embedding_extractors import CLIPExtractor
        
        extractor = CLIPExtractor(
            variant="base-patch32",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        emb1 = extractor.extract_embeddings(small_images, use_cache=False)
        emb2 = extractor.extract_embeddings(small_images, use_cache=False)
        
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5)

    def test_batch_independence(self, sample_images, mock_cache_manager):
        """Test batch independence."""
        from embedding.embedding_extractors import CLIPExtractor
        
        ext1 = CLIPExtractor(
            variant="base-patch32", device="cpu", batch_size=2,
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        ext2 = CLIPExtractor(
            variant="base-patch32", device="cpu", batch_size=5,
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_allclose(emb1, emb2, rtol=2e-3)

    def test_token_modes(self, small_images, mock_cache_manager):
        """Test different token extraction modes."""
        from embedding.embedding_extractors import CLIPExtractor
        
        extractor = CLIPExtractor(
            variant="base-patch32",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        # CLS token
        emb_cls = extractor.extract_embeddings(
            small_images, return_which_tokens="cls", use_cache=False
        )
        assert emb_cls.shape == (3, 768)
        
        # k_last_cls
        emb_k = extractor.extract_embeddings(
            small_images, return_which_tokens="k_last_cls", k_last=2, use_cache=False
        )
        assert emb_k.shape == (3, 768 * 2)

    def test_cache_behavior(self, small_images, mock_cache_manager):
        """Test cache behavior."""
        from embedding.embedding_extractors import CLIPExtractor
        
        mock_cache_manager.get.return_value = None
        extractor = CLIPExtractor(
            variant="base-patch32", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        extractor.extract_embeddings(small_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()

    def test_embeddings_finite(self, small_images, mock_cache_manager):
        """Test embeddings are finite."""
        from embedding.embedding_extractors import CLIPExtractor
        
        extractor = CLIPExtractor(
            variant="base-patch32", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        assert np.all(np.isfinite(embeddings))


# ============================================================================
# Test SigLIPExtractor
# ============================================================================

@pytest.mark.slow
class TestSigLIPExtractor:
    """Test SigLIP extractor."""

    @pytest.mark.parametrize("variant,expected_dim", [
        ("base-patch16-224", 768),
        ("base-patch16-384", 768),
        ("large-patch16-256", 1024),
    ])
    def test_embedding_dimensions(self, variant, expected_dim):
        """Test embedding dimension for each variant."""
        from embedding.embedding_extractors import SigLIPExtractor
        
        extractor = SigLIPExtractor(variant=variant, device="cuda", proxy='http://proxy.example:3128')
        
        assert extractor.get_embedding_dim() == expected_dim

    def test_invalid_variant_raises(self):
        """Test that invalid variant raises ValueError."""
        from embedding.embedding_extractors import SigLIPExtractor
        
        with pytest.raises(ValueError, match="Unknown SigLIP variant"):
            SigLIPExtractor(variant="giant-patch8-512", device="cuda")

    def test_extract_embeddings_shape(self, small_images, mock_cache_manager):
        """Test output shape."""
        from embedding.embedding_extractors import SigLIPExtractor
        
        extractor = SigLIPExtractor(
            variant="base-patch16-224",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        assert embeddings.shape == (3, 768)

    def test_single_image(self, single_image, mock_cache_manager):
        """Test single image extraction."""
        from embedding.embedding_extractors import SigLIPExtractor
        
        extractor = SigLIPExtractor(
            variant="base-patch16-224",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        assert embeddings.shape == (1, 768)

    def test_reproducibility(self, small_images, mock_cache_manager):
        """Test reproducibility."""
        from embedding.embedding_extractors import SigLIPExtractor
        
        extractor = SigLIPExtractor(
            variant="base-patch16-224",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        emb1 = extractor.extract_embeddings(small_images, use_cache=False)
        emb2 = extractor.extract_embeddings(small_images, use_cache=False)
        
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5)

    def test_batch_independence(self, sample_images, mock_cache_manager):
        """Test batch independence."""
        from embedding.embedding_extractors import SigLIPExtractor
        
        ext1 = SigLIPExtractor(
            variant="base-patch16-224", device="cpu", batch_size=2,
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        ext2 = SigLIPExtractor(
            variant="base-patch16-224", device="cpu", batch_size=5,
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_allclose(emb1, emb2, rtol=1e-3)

    def test_cache_behavior(self, small_images, mock_cache_manager):
        """Test cache behavior."""
        from embedding.embedding_extractors import SigLIPExtractor
        
        mock_cache_manager.get.return_value = None
        extractor = SigLIPExtractor(
            variant="base-patch16-224", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        extractor.extract_embeddings(small_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()

    def test_embeddings_finite(self, small_images, mock_cache_manager):
        """Test embeddings are finite."""
        from embedding.embedding_extractors import SigLIPExtractor
        
        extractor = SigLIPExtractor(
            variant="base-patch16-224", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        assert np.all(np.isfinite(embeddings))


# ============================================================================
# Test MAEExtractor
# ============================================================================

@pytest.mark.slow
class TestMAEExtractor:
    """Test Masked Autoencoder extractor."""

    @pytest.mark.parametrize("variant,expected_dim", [
        ("base", 768),
        ("large", 1024),
        ("huge", 1280),
    ])
    def test_embedding_dimensions(self, variant, expected_dim):
        """Test embedding dimension for each variant."""
        from embedding.embedding_extractors import MAEExtractor
        
        extractor = MAEExtractor(variant=variant, device="cuda")
        
        assert extractor.get_embedding_dim() == expected_dim

    def test_invalid_variant_raises(self):
        """Test that invalid variant raises ValueError."""
        from embedding.embedding_extractors import MAEExtractor
        
        with pytest.raises(ValueError, match="Unknown MAE variant"):
            MAEExtractor(variant="giant", device="cuda")

    def test_extract_embeddings_shape(self, small_images, mock_cache_manager):
        """Test output shape."""
        from embedding.embedding_extractors import MAEExtractor
        
        extractor = MAEExtractor(
            variant="base",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        assert embeddings.shape == (3, 768)

    def test_single_image(self, single_image, mock_cache_manager):
        """Test single image extraction."""
        from embedding.embedding_extractors import MAEExtractor
        
        extractor = MAEExtractor(
            variant="base",
            device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        assert embeddings.shape == (1, 768)

    # MAE is by design stochastic due to masking, so batch independence test is not applicable

    # def test_reproducibility(self, small_images, mock_cache_manager):
    #     """Test reproducibility."""
    #     from embedding.embedding_extractors import MAEExtractor
        
    #     extractor = MAEExtractor(
    #         variant="base",
    #         device="cpu",
    #         cache_manager=mock_cache_manager,
    #         proxy='http://proxy.example:3128'
    #     )
        
    #     emb1 = extractor.extract_embeddings(small_images, use_cache=False)
    #     emb2 = extractor.extract_embeddings(small_images, use_cache=False)
        
    #     np.testing.assert_allclose(emb1, emb2, rtol=1e-5)
    

    # def test_batch_independence(self, sample_images, mock_cache_manager):
    #     """Test batch independence."""
    #     from embedding.embedding_extractors import MAEExtractor
        
    #     ext1 = MAEExtractor(
    #         variant="base", device="cpu", batch_size=2,
    #         cache_manager=mock_cache_manager,
    #         proxy='http://proxy.example:3128'
    #     )
    #     ext2 = MAEExtractor(
    #         variant="base", device="cpu", batch_size=5,
    #         cache_manager=mock_cache_manager,
    #         proxy='http://proxy.example:3128'
    #     )
        
    #     emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
    #     emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
    #     np.testing.assert_allclose(emb1, emb2, rtol=1e-3)

    def test_cache_behavior(self, small_images, mock_cache_manager):
        """Test cache behavior."""
        from embedding.embedding_extractors import MAEExtractor
        
        mock_cache_manager.get.return_value = None
        extractor = MAEExtractor(
            variant="base", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        extractor.extract_embeddings(small_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()

    def test_embeddings_finite(self, small_images, mock_cache_manager):
        """Test embeddings are finite."""
        from embedding.embedding_extractors import MAEExtractor
        
        extractor = MAEExtractor(
            variant="base", device="cuda",
            cache_manager=mock_cache_manager,
            proxy='http://proxy.example:3128'
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        assert np.all(np.isfinite(embeddings))


# ============================================================================
# Test RawPixelResizerExtractor (Fast - No Model Download)
# ============================================================================

class TestRawPixelResizerExtractor:
    """Test raw pixel extractor (no neural network, fast tests)."""

    @pytest.mark.parametrize("size,expected_dim", [
        ("16", 768),
        ("32", 3072),
        ("64", 12288),
        ("128", 49152),
    ])
    def test_embedding_dimensions(self, size, expected_dim):
        """Test embedding dimension calculation."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(variant=size, device="cuda")
        
        assert extractor.get_embedding_dim() == expected_dim

    def test_invalid_variant_raises(self):
        """Test that non-integer variant raises ValueError."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        with pytest.raises(ValueError, match="Invalid target size"):
            RawPixelResizerExtractor(variant="abc", device="cuda")

    def test_extract_embeddings(self, sample_images, mock_cache_manager):
        """Test basic embedding extraction."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert embeddings.shape == (5, 3072)
        assert embeddings.dtype in (np.float32, np.float64)

    def test_perfect_reproducibility(self, sample_images, mock_cache_manager):
        """Test that identical inputs produce identical outputs."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        ext1 = RawPixelResizerExtractor(variant="32", device="cuda", cache_manager=mock_cache_manager)
        ext2 = RawPixelResizerExtractor(variant="32", device="cuda", cache_manager=mock_cache_manager)
        
        emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_images_different_embeddings(self, sample_images, mock_cache_manager):
        """Test that different images produce different embeddings."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        for i in range(len(sample_images)):
            for j in range(i + 1, len(sample_images)):
                assert not np.allclose(embeddings[i], embeddings[j]), \
                    f"Images {i} and {j} have identical embeddings"

    def test_single_image(self, single_image, mock_cache_manager):
        """Test extraction for single image."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        
        assert embeddings.shape == (1, 3072)

    def test_batch_independence(self, sample_images, mock_cache_manager):
        """Test that batch size doesn't affect results."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        ext1 = RawPixelResizerExtractor(variant="32", device="cpu", batch_size=2, cache_manager=mock_cache_manager)
        ext2 = RawPixelResizerExtractor(variant="32", device="cpu", batch_size=5, cache_manager=mock_cache_manager)
        
        emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_array_equal(emb1, emb2)

    def test_large_batch_size(self, sample_images, mock_cache_manager):
        """Test with batch size larger than dataset."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32",
            device="cuda",
            batch_size=100,
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert embeddings.shape == (5, 3072)

    def test_batch_size_one(self, sample_images, mock_cache_manager):
        """Test with batch size of 1."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32",
            device="cuda",
            batch_size=1,
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert embeddings.shape == (5, 3072)

    def test_values_normalized(self, sample_images, mock_cache_manager):
        """Test that pixel values are in expected range [0, 1]."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert embeddings.min() >= 0.0
        assert embeddings.max() <= 1.0

    def test_cache_miss(self, sample_images, mock_cache_manager):
        """Test cache miss triggers computation."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        mock_cache_manager.get.return_value = None
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(sample_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()

    def test_cache_hit(self, sample_images):
        """Test cache hit returns cached data."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        cached = np.random.randn(5, 3072).astype(np.float32)
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=True)
        
        np.testing.assert_array_equal(embeddings, cached)
        mock_cache.put.assert_not_called()

    def test_use_cache_false(self, sample_images, mock_cache_manager):
        """Test use_cache=False bypasses cache."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(sample_images, use_cache=False)
        
        mock_cache_manager.get.assert_not_called()
        mock_cache_manager.put.assert_not_called()

    def test_empty_list(self, mock_cache_manager):
        """Test empty list returns empty array."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        extractor.load_model()
        
        embeddings = extractor.extract_embeddings([], use_cache=False)
        
        assert len(embeddings) == 0

    def test_grayscale_handling(self, grayscale_image, mock_cache_manager):
        """Test grayscale image handling."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(grayscale_image, use_cache=False)
        assert embeddings.shape == (1, 3072)

    def test_rgba_handling(self, rgba_image, mock_cache_manager):
        """Test RGBA image handling."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(rgba_image, use_cache=False)
        assert embeddings.shape == (1, 3072)

    def test_tiny_image(self, tiny_image, mock_cache_manager):
        """Test tiny image handling."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(tiny_image, use_cache=False)
        assert embeddings.shape == (1, 3072)

    def test_large_image(self, large_image, mock_cache_manager):
        """Test large image handling."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(large_image, use_cache=False)
        assert embeddings.shape == (1, 3072)

    def test_embeddings_finite(self, sample_images, mock_cache_manager):
        """Test embeddings are finite."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        assert np.all(np.isfinite(embeddings))

    def test_embeddings_not_all_zero(self, sample_images, mock_cache_manager):
        """Test embeddings are not all zeros."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        assert not np.allclose(embeddings, 0)

    def test_cache_params_include_target_size(self, sample_images, mock_cache_manager):
        """Test cache params include target_size."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(sample_images, use_cache=True)
        
        put_kwargs = mock_cache_manager.put.call_args[1]
        assert put_kwargs.get("target_size") == 32


# ============================================================================
# Test SparseRandomProjectionResizerExtractor
# ============================================================================

class TestSparseRandomProjectionExtractor:
    """Test sparse random projection extractor."""

    def test_variant_parsing(self):
        """Test variant string parsing."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="64-512",
            device="cuda"
        )
        
        assert extractor.target_size == 64
        assert extractor.target_dim == 512

    def test_default_variant(self):
        """Test default variant parsing."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="default",
            device="cuda"
        )
        
        assert extractor.target_size == 64
        assert extractor.target_dim == 1000

    def test_invalid_variant_format(self):
        """Test that invalid variant format raises ValueError."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        with pytest.raises(ValueError, match="Invalid variant"):
            SparseRandomProjectionResizerExtractor(variant="invalid", device="cuda")

    def test_embedding_dimension(self):
        """Test get_embedding_dim returns target dimension."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256",
            device="cuda"
        )
        
        assert extractor.get_embedding_dim() == 256

    def test_extract_embeddings_shape(self, sample_images, mock_cache_manager):
        """Test output shape."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-512",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert embeddings.shape == (5, 512)

    def test_single_image(self, single_image, mock_cache_manager):
        """Test single image extraction."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        assert embeddings.shape == (1, 256)

    def test_reproducibility_with_fixed_seed(self, sample_images, mock_cache_manager):
        """Test reproducibility due to fixed random_state=42."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        ext1 = SparseRandomProjectionResizerExtractor(
            variant="32-256",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        ext2 = SparseRandomProjectionResizerExtractor(
            variant="32-256",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_batch_independence(self, sample_images, mock_cache_manager):
        """Test batch independence."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        ext1 = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cpu", batch_size=2, cache_manager=mock_cache_manager
        )
        ext2 = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cpu", batch_size=5, cache_manager=mock_cache_manager
        )
        
        emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_dimensionality_reduction(self, sample_images, mock_cache_manager):
        """Test that output dimension is less than input."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="64-512",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert embeddings.shape[1] == 512
        assert embeddings.shape[1] < 64 * 64 * 3

    def test_cache_miss(self, sample_images, mock_cache_manager):
        """Test cache miss triggers computation."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        mock_cache_manager.get.return_value = None
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(sample_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()

    def test_cache_hit(self, sample_images):
        """Test cache hit returns cached data."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        cached = np.random.randn(5, 256).astype(np.float32)
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=True)
        
        np.testing.assert_array_equal(embeddings, cached)
        mock_cache.put.assert_not_called()

    def test_empty_list(self, mock_cache_manager):
        """Test empty list returns empty array."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        extractor.load_model()
        
        embeddings = extractor.extract_embeddings([], use_cache=False)
        assert len(embeddings) == 0

    def test_grayscale_handling(self, grayscale_image, mock_cache_manager):
        """Test grayscale image handling."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(grayscale_image, use_cache=False)
        assert embeddings.shape == (1, 256)

    def test_rgba_handling(self, rgba_image, mock_cache_manager):
        """Test RGBA image handling."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(rgba_image, use_cache=False)
        assert embeddings.shape == (1, 256)

    def test_tiny_image(self, tiny_image, mock_cache_manager):
        """Test tiny image handling."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(tiny_image, use_cache=False)
        assert embeddings.shape == (1, 256)

    def test_embeddings_finite(self, sample_images, mock_cache_manager):
        """Test embeddings are finite."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        assert np.all(np.isfinite(embeddings))

    def test_cache_params_include_dimensions(self, sample_images, mock_cache_manager):
        """Test cache params include target_size and target_dim."""
        from embedding.embedding_extractors import SparseRandomProjectionResizerExtractor
        
        extractor = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(sample_images, use_cache=True)
        
        put_kwargs = mock_cache_manager.put.call_args[1]
        assert put_kwargs.get("target_size") == 32
        assert put_kwargs.get("target_dim") == 256


# ============================================================================
# Test UntrainedAutoEncoderExtractor
# ============================================================================

class TestUntrainedAutoEncoderExtractor:
    """Test untrained autoencoder extractor."""

    @pytest.mark.parametrize("input_size,expected_dim", [
        (32, 256),    # 16 * 4 * 4
        (64, 1024),   # 16 * 8 * 8
        (128, 4096),  # 16 * 16 * 16
    ])
    def test_embedding_dimension_calculation(self, input_size, expected_dim):
        """Test embedding dimension calculation based on architecture."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default",
            input_size=input_size,
            device="cuda"
        )
        
        assert extractor.get_embedding_dim() == expected_dim

    def test_invalid_variant_raises(self):
        """Test that non-default variant raises ValueError."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        with pytest.raises(ValueError, match="Unknown variant"):
            UntrainedAutoEncoderExtractor(variant="large", device="cuda")

    def test_extract_embeddings_shape(self, sample_images, mock_cache_manager):
        """Test output shape."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default",
            input_size=64,
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert embeddings.shape == (5, 1024)

    def test_single_image(self, single_image, mock_cache_manager):
        """Test single image extraction."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        assert embeddings.shape == (1, 1024)

    def test_autoencoder_architecture(self, mock_cache_manager):
        """Test that autoencoder has expected layer structure."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default",
            input_size=64,
            device="cuda",
            cache_manager=mock_cache_manager
        )
        extractor.load_model()
        
        layers = list(extractor.autoencoder.children())
        assert isinstance(layers[0], nn.Conv2d)
        assert isinstance(layers[1], nn.ReLU)
        assert isinstance(layers[2], nn.MaxPool2d)
        assert isinstance(layers[3], nn.Conv2d)
        assert isinstance(layers[4], nn.ReLU)
        assert isinstance(layers[5], nn.MaxPool2d)
        assert isinstance(layers[6], nn.Conv2d)
        assert isinstance(layers[7], nn.ReLU)
        assert isinstance(layers[8], nn.MaxPool2d)

    def test_model_is_not_trained(self, sample_images, mock_cache_manager):
        """Test that model parameters are random (not pretrained)."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        ext1 = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        ext2 = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        
        ext1.load_model()
        ext2.load_model()
        
        w1 = list(ext1.autoencoder.parameters())[0].detach().cpu().numpy()
        w2 = list(ext2.autoencoder.parameters())[0].detach().cpu().numpy()
        
        assert not np.allclose(w1, w2)

    def test_cache_miss(self, sample_images, mock_cache_manager):
        """Test cache miss triggers computation."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        mock_cache_manager.get.return_value = None
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(sample_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()

    def test_cache_hit(self, sample_images):
        """Test cache hit returns cached data."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        cached = np.random.randn(5, 1024).astype(np.float32)
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=True)
        
        np.testing.assert_array_equal(embeddings, cached)
        mock_cache.put.assert_not_called()

    def test_empty_list(self, mock_cache_manager):
        """Test empty list returns empty array."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        extractor.load_model()
        
        embeddings = extractor.extract_embeddings([], use_cache=False)
        assert len(embeddings) == 0

    def test_grayscale_handling(self, grayscale_image, mock_cache_manager):
        """Test grayscale image handling."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(grayscale_image, use_cache=False)
        assert embeddings.shape == (1, 1024)

    def test_rgba_handling(self, rgba_image, mock_cache_manager):
        """Test RGBA image handling."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(rgba_image, use_cache=False)
        assert embeddings.shape == (1, 1024)

    def test_tiny_image(self, tiny_image, mock_cache_manager):
        """Test tiny image handling."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(tiny_image, use_cache=False)
        assert embeddings.shape == (1, 1024)

    def test_embeddings_finite(self, sample_images, mock_cache_manager):
        """Test embeddings are finite."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        assert np.all(np.isfinite(embeddings))

    def test_cache_params_include_input_size(self, sample_images, mock_cache_manager):
        """Test cache params include input_size."""
        from embedding.embedding_extractors import UntrainedAutoEncoderExtractor
        
        extractor = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda",
            cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(sample_images, use_cache=True)
        
        put_kwargs = mock_cache_manager.put.call_args[1]
        assert put_kwargs.get("input_size") == 64


# ============================================================================
# Test WaveletScatteringExtractor
# ============================================================================

class TestWaveletScatteringExtractor:
    """Test wavelet scattering extractor."""

    def test_variant_parsing(self):
        """Test variant string parsing."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        extractor = WaveletScatteringExtractor(variant="128-3", device="cuda")
        
        assert extractor.input_size == 128
        assert extractor.J == 3

    def test_default_variant(self):
        """Test default variant."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        extractor = WaveletScatteringExtractor(variant="default", device="cuda")
        
        assert extractor.input_size == 224
        assert extractor.J == 3

    def test_invalid_variant_format(self):
        """Test that invalid variant format raises ValueError."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        with pytest.raises(ValueError, match="Invalid variant"):
            WaveletScatteringExtractor(variant="invalid", device="cuda")

    def test_extract_embeddings_shape(self, small_images, mock_cache_manager):
        """Test output shape."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        extractor = WaveletScatteringExtractor(
            variant="64-2",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0

    def test_single_image(self, single_image, mock_cache_manager):
        """Test single image extraction."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        extractor = WaveletScatteringExtractor(
            variant="64-2", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(single_image, use_cache=False)
        assert embeddings.shape[0] == 1

    def test_reproducibility(self, small_images, mock_cache_manager):
        """Test that wavelet transform is deterministic."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        ext1 = WaveletScatteringExtractor(variant="64-2", device="cuda", cache_manager=mock_cache_manager)
        ext2 = WaveletScatteringExtractor(variant="64-2", device="cuda", cache_manager=mock_cache_manager)
        
        emb1 = ext1.extract_embeddings(small_images, use_cache=False)
        emb2 = ext2.extract_embeddings(small_images, use_cache=False)
        
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_batch_independence(self, sample_images, mock_cache_manager):
        """Test batch independence."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        ext1 = WaveletScatteringExtractor(
            variant="64-2", device="cpu", batch_size=2, cache_manager=mock_cache_manager
        )
        ext2 = WaveletScatteringExtractor(
            variant="64-2", device="cpu", batch_size=5, cache_manager=mock_cache_manager
        )
        
        emb1 = ext1.extract_embeddings(sample_images, use_cache=False)
        emb2 = ext2.extract_embeddings(sample_images, use_cache=False)
        
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_cache_miss(self, small_images, mock_cache_manager):
        """Test cache miss triggers computation."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        mock_cache_manager.get.return_value = None
        extractor = WaveletScatteringExtractor(
            variant="64-2", device="cuda", cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(small_images, use_cache=True)
        
        mock_cache_manager.get.assert_called_once()
        mock_cache_manager.put.assert_called_once()

    def test_cache_hit(self, small_images):
        """Test cache hit returns cached data."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        cached = np.random.randn(3, 1000).astype(np.float32)
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached
        
        extractor = WaveletScatteringExtractor(
            variant="64-2", device="cuda", cache_manager=mock_cache
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=True)
        
        np.testing.assert_array_equal(embeddings, cached)
        mock_cache.put.assert_not_called()

    def test_empty_list(self, mock_cache_manager):
        """Test empty list returns empty array."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        extractor = WaveletScatteringExtractor(
            variant="64-2", device="cuda", cache_manager=mock_cache_manager
        )
        extractor.load_model()
        
        embeddings = extractor.extract_embeddings([], use_cache=False)
        assert len(embeddings) == 0

    def test_embeddings_finite(self, small_images, mock_cache_manager):
        """Test embeddings are finite."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        extractor = WaveletScatteringExtractor(
            variant="64-2", device="cuda", cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(small_images, use_cache=False)
        assert np.all(np.isfinite(embeddings))

    def test_get_embedding_dim_loads_model(self, mock_cache_manager):
        """Test that get_embedding_dim triggers model load if needed."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        extractor = WaveletScatteringExtractor(
            variant="64-2", device="cuda", cache_manager=mock_cache_manager
        )
        
        assert extractor.scattering is None
        dim = extractor.get_embedding_dim()
        assert extractor.scattering is not None
        assert dim > 0

    def test_cache_params_include_J(self, small_images, mock_cache_manager):
        """Test cache params include J parameter."""
        from embedding.embedding_extractors import WaveletScatteringExtractor
        
        extractor = WaveletScatteringExtractor(
            variant="64-2", device="cuda", cache_manager=mock_cache_manager
        )
        
        extractor.extract_embeddings(small_images, use_cache=True)
        
        put_kwargs = mock_cache_manager.put.call_args[1]
        assert put_kwargs.get("J") == 2
        assert put_kwargs.get("input_size") == 64


# ============================================================================
# Test Mixed Valid/Invalid Images for All Fast Extractors
# ============================================================================

class TestCorruptImageHandling:
    """Test corrupt image handling for all fast extractors."""

    @pytest.mark.parametrize("extractor_class,variant,kwargs", [
        ("RawPixelResizerExtractor", "32", {}),
        ("SparseRandomProjectionResizerExtractor", "32-256", {}),
        ("UntrainedAutoEncoderExtractor", "default", {"input_size": 64}),
    ])
    def test_mixed_valid_corrupt(self, sample_images, corrupt_image, mock_cache_manager, extractor_class, variant, kwargs):
        """Test handling of mixed valid and corrupt images."""
        from embedding import embedding_extractors
        
        cls = getattr(embedding_extractors, extractor_class)
        extractor = cls(
            variant=variant,
            device="cuda",
            cache_manager=mock_cache_manager,
            **kwargs
        )
        
        mixed = sample_images[:2] + corrupt_image + sample_images[2:]
        
        # Should not crash
        embeddings = extractor.extract_embeddings(mixed, use_cache=False)
        assert isinstance(embeddings, np.ndarray)

    @pytest.mark.parametrize("extractor_class,variant,kwargs", [
        ("RawPixelResizerExtractor", "32", {}),
        ("SparseRandomProjectionResizerExtractor", "32-256", {}),
        ("UntrainedAutoEncoderExtractor", "default", {"input_size": 64}),
    ])
    def test_nonexistent_path(self, tmp_path, mock_cache_manager, extractor_class, variant, kwargs):
        """Test handling of nonexistent image paths."""
        from embedding import embedding_extractors
        
        cls = getattr(embedding_extractors, extractor_class)
        extractor = cls(
            variant=variant,
            device="cuda",
            cache_manager=mock_cache_manager,
            **kwargs
        )
        
        nonexistent = [{"image": str(tmp_path / "nonexistent.jpg")}]
        
        # Should not crash
        embeddings = extractor.extract_embeddings(nonexistent, use_cache=False)
        assert isinstance(embeddings, np.ndarray)


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_multiple_extractors_same_images(self, sample_images, mock_cache_manager):
        """Test using multiple extractors on same images."""
        from embedding.embedding_extractors import (
            RawPixelResizerExtractor,
            SparseRandomProjectionResizerExtractor,
            UntrainedAutoEncoderExtractor
        )
        
        raw_ext = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        proj_ext = SparseRandomProjectionResizerExtractor(
            variant="32-256", device="cuda", cache_manager=mock_cache_manager
        )
        ae_ext = UntrainedAutoEncoderExtractor(
            variant="default", input_size=64, device="cuda", cache_manager=mock_cache_manager
        )
        
        raw_emb = raw_ext.extract_embeddings(sample_images, use_cache=False)
        proj_emb = proj_ext.extract_embeddings(sample_images, use_cache=False)
        ae_emb = ae_ext.extract_embeddings(sample_images, use_cache=False)
        
        assert raw_emb.shape == (5, 3072)
        assert proj_emb.shape == (5, 256)
        assert ae_emb.shape == (5, 1024)

    def test_same_extractor_different_batches(self, sample_images, mock_cache_manager):
        """Test reusing same extractor for different image sets."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        emb1 = extractor.extract_embeddings(sample_images[:2], use_cache=False)
        emb2 = extractor.extract_embeddings(sample_images[2:], use_cache=False)
        
        assert emb1.shape == (2, 3072)
        assert emb2.shape == (3, 3072)

    def test_extraction_with_dataset_name(self, sample_images, mock_cache_manager):
        """Test extraction with dataset_name for cache namespacing."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32",
            device="cuda",
            cache_manager=mock_cache_manager
        )
        
        embeddings = extractor.extract_embeddings(
            sample_images,
            use_cache=True,
            dataset_name="test_dataset",
            base_path="/data/images"
        )
        
        put_call_kwargs = mock_cache_manager.put.call_args[1]
        assert put_call_kwargs["dataset_name"] == "test_dataset"
        assert put_call_kwargs["base_path"] == "/data/images"

    def test_extractor_reuse_across_extractions(self, sample_images, mock_cache_manager):
        """Test that extractor can be reused multiple times."""
        from embedding.embedding_extractors import RawPixelResizerExtractor
        
        extractor = RawPixelResizerExtractor(
            variant="32", device="cuda", cache_manager=mock_cache_manager
        )
        
        for _ in range(5):
            emb = extractor.extract_embeddings(sample_images, use_cache=False)
            assert emb.shape == (5, 3072)


# ============================================================================
# Test Performance Characteristics
# ============================================================================

class TestPerformanceCharacteristics:
    """Tests that verify performance characteristics."""

    @pytest.mark.parametrize("extractor_class,variant,kwargs,expected_shape", [
        ("RawPixelResizerExtractor", "32", {}, (5, 3072)),
        ("SparseRandomProjectionResizerExtractor", "32-256", {}, (5, 256)),
        ("UntrainedAutoEncoderExtractor", "default", {"input_size": 64}, (5, 1024)),
    ])
    def test_embeddings_are_float(self, sample_images, mock_cache_manager, extractor_class, variant, kwargs, expected_shape):
        """Test that embeddings are floating point."""
        from embedding import embedding_extractors
        
        cls = getattr(embedding_extractors, extractor_class)
        extractor = cls(variant=variant, device="cuda", cache_manager=mock_cache_manager, **kwargs)
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        assert np.issubdtype(embeddings.dtype, np.floating)

    @pytest.mark.parametrize("extractor_class,variant,kwargs", [
        ("RawPixelResizerExtractor", "32", {}),
        ("SparseRandomProjectionResizerExtractor", "32-256", {}),
        ("UntrainedAutoEncoderExtractor", "default", {"input_size": 64}),
    ])
    def test_embedding_variance(self, sample_images, mock_cache_manager, extractor_class, variant, kwargs):
        """Test that embeddings have reasonable variance across images."""
        from embedding import embedding_extractors
        
        cls = getattr(embedding_extractors, extractor_class)
        extractor = cls(variant=variant, device="cuda", cache_manager=mock_cache_manager, **kwargs)
        
        embeddings = extractor.extract_embeddings(sample_images, use_cache=False)
        
        variance = np.var(embeddings, axis=0)
        nonzero_variance_dims = np.sum(variance > 1e-10)
        
        assert nonzero_variance_dims > embeddings.shape[1] * 0.1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow"
    ])
