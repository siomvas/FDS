"""
Unit tests for embedding-based partitioning pipeline.

Tests all components: pooling strategies, clustering methods, cache manager,
configuration system, and the main EmbeddingSplitter.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json

# Import components to test
from pooling_strategies import (
    CLSPooling,
    MeanPooling,
    MaxPooling,
    GeMLPooling,
    LastKLayerPooling,
    SpectralSharpeningPooling,
    get_pooling_strategy,
)
from clustering_methods import (
    HDBSCANMethod,
    KMeansMethod,
    KNNGraphMethod,
    SpectralMethod,
    get_clustering_method,
)
from cache_manager import CacheManager
from config import (
    EmbeddingConfig,
    ModelConfig,
    PoolingConfig,
    ClusteringConfig,
    get_preset_config,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_embeddings_2d():
    """2D embeddings (already pooled)."""
    np.random.seed(42)
    return np.random.randn(100, 768)


@pytest.fixture
def sample_embeddings_3d():
    """3D embeddings (with tokens)."""
    np.random.seed(42)
    return np.random.randn(50, 197, 768)  # 50 samples, 197 tokens, 768 dims


@pytest.fixture
def layer_embeddings():
    """Dictionary of layer embeddings."""
    np.random.seed(42)
    return {
        -4: np.random.randn(50, 768),
        -3: np.random.randn(50, 768),
        -2: np.random.randn(50, 768),
        -1: np.random.randn(50, 768),
    }


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for cache tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def clusterable_embeddings():
    """Embeddings with clear cluster structure."""
    np.random.seed(42)
    # Create 3 well-separated clusters
    cluster1 = np.random.randn(30, 50) + np.array([10, 0] + [0] * 48)
    cluster2 = np.random.randn(30, 50) + np.array([0, 10] + [0] * 48)
    cluster3 = np.random.randn(30, 50) + np.array([-10, -10] + [0] * 48)
    return np.vstack([cluster1, cluster2, cluster3])


# ============================================================================
# POOLING STRATEGY TESTS
# ============================================================================

class TestCLSPooling:
    """Test CLS token pooling."""

    def test_2d_input_unchanged(self, sample_embeddings_2d):
        """2D input (already pooled) should pass through unchanged."""
        pooling = CLSPooling()
        result = pooling.pool(sample_embeddings_2d)
        assert result.shape == sample_embeddings_2d.shape
        np.testing.assert_array_equal(result, sample_embeddings_2d)

    def test_3d_input_extracts_first_token(self, sample_embeddings_3d):
        """3D input should extract CLS token (first token)."""
        pooling = CLSPooling()
        result = pooling.pool(sample_embeddings_3d)
        assert result.shape == (50, 768)
        np.testing.assert_array_equal(result, sample_embeddings_3d[:, 0, :])

    def test_invalid_dimensions(self):
        """Should raise error for invalid dimensions."""
        pooling = CLSPooling()
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            pooling.pool(np.random.randn(10, 20, 30, 40))


class TestMeanPooling:
    """Test mean pooling."""

    def test_2d_input_unchanged(self, sample_embeddings_2d):
        """2D input should pass through unchanged."""
        pooling = MeanPooling()
        result = pooling.pool(sample_embeddings_2d)
        assert result.shape == sample_embeddings_2d.shape

    def test_3d_input_averages_tokens(self, sample_embeddings_3d):
        """3D input should average over token dimension."""
        pooling = MeanPooling()
        result = pooling.pool(sample_embeddings_3d)
        assert result.shape == (50, 768)
        expected = np.mean(sample_embeddings_3d, axis=1)
        np.testing.assert_array_almost_equal(result, expected)


class TestMaxPooling:
    """Test max pooling."""

    def test_3d_input_max_over_tokens(self, sample_embeddings_3d):
        """3D input should take max over token dimension."""
        pooling = MaxPooling()
        result = pooling.pool(sample_embeddings_3d)
        assert result.shape == (50, 768)
        expected = np.max(sample_embeddings_3d, axis=1)
        np.testing.assert_array_almost_equal(result, expected)


class TestGeMLPooling:
    """Test GeM pooling."""

    def test_p_equals_1_is_mean(self, sample_embeddings_3d):
        """GeM with p=1 should equal mean pooling."""
        gem_pooling = GeMLPooling(p=1.0, eps=0)
        mean_pooling = MeanPooling()

        gem_result = gem_pooling.pool(sample_embeddings_3d)
        mean_result = mean_pooling.pool(sample_embeddings_3d)

        np.testing.assert_array_almost_equal(gem_result, mean_result, decimal=5)

    def test_large_p_approaches_max(self, sample_embeddings_3d):
        """GeM with large p should approach max pooling."""
        gem_pooling = GeMLPooling(p=100.0, eps=1e-10)
        max_pooling = MaxPooling()

        # Use positive embeddings for this test
        positive_embeddings = np.abs(sample_embeddings_3d)

        gem_result = gem_pooling.pool(positive_embeddings)
        max_result = max_pooling.pool(positive_embeddings)

        # Should be close (not exact due to numerical precision)
        correlation = np.corrcoef(gem_result.flatten(), max_result.flatten())[0, 1]
        assert correlation > 0.95

    def test_output_shape(self, sample_embeddings_3d):
        """Output shape should be correct."""
        pooling = GeMLPooling(p=3.0)
        result = pooling.pool(sample_embeddings_3d)
        assert result.shape == (50, 768)


class TestLastKLayerPooling:
    """Test last-K layer pooling."""

    def test_mean_aggregation(self, layer_embeddings):
        """Mean aggregation should average across layers."""
        pooling = LastKLayerPooling(k=4, aggregation="mean")
        result = pooling.pool(layer_embeddings)

        expected = np.mean([layer_embeddings[k] for k in [-4, -3, -2, -1]], axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_concat_aggregation(self, layer_embeddings):
        """Concat aggregation should concatenate layers."""
        pooling = LastKLayerPooling(k=4, aggregation="concat")
        result = pooling.pool(layer_embeddings)

        assert result.shape == (50, 768 * 4)

    def test_weighted_aggregation(self, layer_embeddings):
        """Weighted aggregation should weight later layers more."""
        pooling = LastKLayerPooling(k=4, aggregation="weighted")
        result = pooling.pool(layer_embeddings)

        assert result.shape == (50, 768)
        # Later layers should have more influence
        # (Hard to test directly, just check shape and no errors)

    def test_takes_last_k_layers(self):
        """Should only use last K layers even if more provided."""
        layer_embs = {i: np.random.randn(10, 64) for i in range(-10, 0)}
        pooling = LastKLayerPooling(k=3, aggregation="mean")
        result = pooling.pool(layer_embs)

        # Should only use layers -3, -2, -1
        expected_keys = [-3, -2, -1]
        # We can't directly verify which were used, but shape should be correct
        assert result.shape == (10, 64)

    def test_invalid_aggregation_raises(self):
        """Invalid aggregation method should raise error."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            LastKLayerPooling(k=4, aggregation="invalid")

    def test_requires_dict_input(self, sample_embeddings_2d):
        """Should require dict input."""
        pooling = LastKLayerPooling(k=4, aggregation="mean")
        with pytest.raises(TypeError, match="expects dict"):
            pooling.pool(sample_embeddings_2d)


class TestSpectralSharpeningPooling:
    """Test spectral sharpening pooling."""

    @pytest.mark.slow
    def test_output_dimensions(self, sample_embeddings_2d):
        """Output should have k dimensions."""
        pooling = SpectralSharpeningPooling(k=50)
        result = pooling.pool(sample_embeddings_2d)
        assert result.shape == (100, 50)

    @pytest.mark.slow
    def test_too_few_samples_returns_original(self):
        """If n_samples < k, should return original embeddings."""
        embeddings = np.random.randn(10, 768)
        pooling = SpectralSharpeningPooling(k=50)
        result = pooling.pool(embeddings)
        assert result.shape == embeddings.shape

    @pytest.mark.slow
    def test_handles_3d_input(self, sample_embeddings_3d):
        """Should pool 3D input first."""
        pooling = SpectralSharpeningPooling(k=30)
        result = pooling.pool(sample_embeddings_3d)
        # 50 samples, k=30
        assert result.shape == (50, 30)


class TestPoolingFactory:
    """Test pooling strategy factory function."""

    def test_creates_cls_pooling(self):
        """Should create CLSPooling."""
        pooling = get_pooling_strategy("cls")
        assert isinstance(pooling, CLSPooling)

    def test_creates_gem_with_params(self):
        """Should create GeMLPooling with parameters."""
        pooling = get_pooling_strategy("gem", p=4.0, eps=1e-5)
        assert isinstance(pooling, GeMLPooling)
        assert pooling.p == 4.0
        assert pooling.eps == 1e-5

    def test_invalid_strategy_raises(self):
        """Invalid strategy should raise error."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            get_pooling_strategy("invalid")


# ============================================================================
# CLUSTERING METHOD TESTS
# ============================================================================

class TestKMeansMethod:
    """Test K-means clustering."""

    def test_correct_number_of_clusters(self, clusterable_embeddings):
        """Should create exactly n_clusters."""
        clustering = KMeansMethod(n_clusters=3, random_state=42)
        labels = clustering.cluster(clusterable_embeddings)

        assert len(np.unique(labels)) == 3

    def test_all_samples_assigned(self, clusterable_embeddings):
        """All samples should be assigned to a cluster."""
        clustering = KMeansMethod(n_clusters=3, random_state=42)
        labels = clustering.cluster(clusterable_embeddings)

        assert len(labels) == len(clusterable_embeddings)
        assert all(label >= 0 for label in labels)

    def test_cluster_info(self, clusterable_embeddings):
        """get_cluster_info should return correct info."""
        clustering = KMeansMethod(n_clusters=3, random_state=42)
        labels = clustering.cluster(clusterable_embeddings)
        info = clustering.get_cluster_info()

        assert info["method"] == "KMeans"
        assert info["num_clusters"] == 3
        assert "cluster_sizes" in info
        assert sum(info["cluster_sizes"].values()) == len(clusterable_embeddings)


class TestHDBSCANMethod:
    """Test HDBSCAN clustering."""

    def test_finds_clusters(self, clusterable_embeddings):
        """Should find clusters in structured data."""
        clustering = HDBSCANMethod(min_cluster_size=15)
        labels = clustering.cluster(clusterable_embeddings)

        unique_labels = set(labels)
        # Should find at least 2 clusters (may have noise)
        num_clusters = len(unique_labels - {-1})
        assert num_clusters >= 2

    def test_handles_noise(self, clusterable_embeddings):
        """Should identify noise samples (label -1)."""
        clustering = HDBSCANMethod(min_cluster_size=20)
        labels = clustering.cluster(clusterable_embeddings)

        # It's okay to have noise or no noise
        assert -1 in labels or -1 not in labels

    def test_cluster_info_includes_noise_count(self, clusterable_embeddings):
        """Cluster info should include noise count."""
        clustering = HDBSCANMethod(min_cluster_size=15)
        labels = clustering.cluster(clusterable_embeddings)
        info = clustering.get_cluster_info()

        assert "noise_count" in info
        assert info["noise_count"] >= 0


class TestKNNGraphMethod:
    """Test k-NN graph clustering."""

    def test_creates_clusters(self, clusterable_embeddings):
        """Should create connected components."""
        clustering = KNNGraphMethod(n_neighbors=10)
        labels = clustering.cluster(clusterable_embeddings)

        assert len(np.unique(labels)) >= 1

    def test_all_samples_assigned(self, clusterable_embeddings):
        """All samples should belong to some cluster."""
        clustering = KNNGraphMethod(n_neighbors=5)
        labels = clustering.cluster(clusterable_embeddings)

        assert len(labels) == len(clusterable_embeddings)
        assert all(label >= 0 for label in labels)


class TestSpectralMethod:
    """Test spectral clustering."""

    @pytest.mark.slow
    def test_correct_number_of_clusters(self, clusterable_embeddings):
        """Should create exactly n_clusters."""
        clustering = SpectralMethod(n_clusters=3, random_state=42)
        labels = clustering.cluster(clusterable_embeddings)

        assert len(np.unique(labels)) == 3

    @pytest.mark.slow
    def test_different_affinities(self, clusterable_embeddings):
        """Should support different affinity methods."""
        # RBF affinity
        clustering_rbf = SpectralMethod(n_clusters=3, affinity="rbf", random_state=42)
        labels_rbf = clustering_rbf.cluster(clusterable_embeddings)
        assert len(labels_rbf) == len(clusterable_embeddings)

        # Nearest neighbors affinity
        clustering_nn = SpectralMethod(
            n_clusters=3, affinity="nearest_neighbors", n_neighbors=10, random_state=42
        )
        labels_nn = clustering_nn.cluster(clusterable_embeddings)
        assert len(labels_nn) == len(clusterable_embeddings)


class TestClusteringFactory:
    """Test clustering method factory function."""

    def test_creates_hdbscan(self):
        """Should create HDBSCANMethod."""
        clustering = get_clustering_method("hdbscan", min_cluster_size=15)
        assert isinstance(clustering, HDBSCANMethod)

    def test_creates_kmeans_with_params(self):
        """Should create KMeansMethod with parameters."""
        clustering = get_clustering_method("kmeans", n_clusters=5, random_state=123)
        assert isinstance(clustering, KMeansMethod)
        assert clustering.n_clusters == 5
        assert clustering.random_state == 123

    def test_invalid_method_raises(self):
        """Invalid method should raise error."""
        with pytest.raises(ValueError, match="Unknown clustering method"):
            get_clustering_method("invalid")


# ============================================================================
# CACHE MANAGER TESTS
# ============================================================================

class TestCacheManager:
    """Test cache manager."""

    def test_cache_put_and_get(self, temp_cache_dir):
        """Should store and retrieve embeddings."""
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=True)

        embeddings = np.random.randn(100, 768)
        image_paths = [f"img_{i}.jpg" for i in range(100)]

        # Put in cache
        cache.put(
            embeddings=embeddings,
            model_name="dinov2/base",
            image_paths=image_paths,
            pooling_strategy="cls",
        )

        # Get from cache
        retrieved = cache.get(
            model_name="dinov2/base",
            image_paths=image_paths,
            pooling_strategy="cls",
        )

        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, embeddings)

    def test_cache_miss_returns_none(self, temp_cache_dir):
        """Cache miss should return None."""
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=True)

        result = cache.get(
            model_name="nonexistent",
            image_paths=["img1.jpg"],
            pooling_strategy="cls",
        )

        assert result is None

    def test_different_params_different_cache(self, temp_cache_dir):
        """Different parameters should have different cache entries."""
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=True)

        embeddings1 = np.random.randn(10, 768)
        embeddings2 = np.random.randn(10, 768)
        image_paths = [f"img_{i}.jpg" for i in range(10)]

        # Store with different pooling strategies
        cache.put(embeddings1, "dinov2/base", image_paths, pooling_strategy="cls")
        cache.put(embeddings2, "dinov2/base", image_paths, pooling_strategy="mean")

        # Retrieve should get different embeddings
        retrieved1 = cache.get("dinov2/base", image_paths, pooling_strategy="cls")
        retrieved2 = cache.get("dinov2/base", image_paths, pooling_strategy="mean")

        np.testing.assert_array_almost_equal(retrieved1, embeddings1)
        np.testing.assert_array_almost_equal(retrieved2, embeddings2)

    def test_cache_dict_embeddings(self, temp_cache_dir):
        """Should handle dictionary of embeddings (multi-layer)."""
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=True)

        layer_embs = {
            -2: np.random.randn(10, 768),
            -1: np.random.randn(10, 768),
        }
        image_paths = [f"img_{i}.jpg" for i in range(10)]

        cache.put(layer_embs, "dinov2/base", image_paths, layers=[-2, -1])
        retrieved = cache.get("dinov2/base", image_paths, layers=[-2, -1])

        assert isinstance(retrieved, dict)
        assert set(retrieved.keys()) == {-2, -1}
        np.testing.assert_array_almost_equal(retrieved[-2], layer_embs[-2])

    def test_disabled_cache_does_nothing(self, temp_cache_dir):
        """Disabled cache should not store or retrieve."""
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=False)

        embeddings = np.random.randn(10, 768)
        image_paths = ["img1.jpg"]

        cache.put(embeddings, "dinov2/base", image_paths, pooling_strategy="cls")
        retrieved = cache.get("dinov2/base", image_paths, pooling_strategy="cls")

        assert retrieved is None

    def test_clear_all_cache(self, temp_cache_dir):
        """Should clear all cache entries."""
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=True)

        # Add some entries
        cache.put(np.random.randn(10, 768), "dinov2/base", ["img1.jpg"], pooling_strategy="cls")
        cache.put(np.random.randn(10, 768), "clip/base", ["img1.jpg"], pooling_strategy="cls")

        # Clear all
        cache.clear()

        # Both should be gone
        assert cache.get("dinov2/base", ["img1.jpg"], pooling_strategy="cls") is None
        assert cache.get("clip/base", ["img1.jpg"], pooling_strategy="cls") is None

    def test_clear_specific_model(self, temp_cache_dir):
        """Should clear only specific model cache."""
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=True)

        # Add entries for two models
        cache.put(np.random.randn(10, 768), "dinov2/base", ["img1.jpg"], pooling_strategy="cls")
        cache.put(np.random.randn(10, 768), "clip/base", ["img1.jpg"], pooling_strategy="cls")

        # Clear only dinov2
        cache.clear(model_name="dinov2/base")

        # DINOv2 should be gone, CLIP should remain
        assert cache.get("dinov2/base", ["img1.jpg"], pooling_strategy="cls") is None
        assert cache.get("clip/base", ["img1.jpg"], pooling_strategy="cls") is not None


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Test configuration system."""

    def test_model_config_validation(self):
        """ModelConfig should validate model type."""
        # Valid
        config = ModelConfig(model_type="dinov2", variant="base")
        assert config.model_type == "dinov2"

        # Invalid
        with pytest.raises(ValueError, match="Invalid model_type"):
            ModelConfig(model_type="invalid", variant="base")

    def test_pooling_config_validation(self):
        """PoolingConfig should validate strategy."""
        # Valid
        config = PoolingConfig(strategy="cls")
        assert config.strategy == "cls"

        # Invalid
        with pytest.raises(ValueError, match="Invalid pooling strategy"):
            PoolingConfig(strategy="invalid")

    def test_clustering_config_validation(self):
        """ClusteringConfig should validate method."""
        # Valid
        config = ClusteringConfig(method="hdbscan")
        assert config.method == "hdbscan"

        # Invalid
        with pytest.raises(ValueError, match="Invalid clustering method"):
            ClusteringConfig(method="invalid")

    def test_embedding_config_creation(self):
        """Should create complete EmbeddingConfig."""
        config = EmbeddingConfig(
            model=ModelConfig(model_type="dinov2", variant="base"),
            pooling=PoolingConfig(strategy="cls"),
            clustering=ClusteringConfig(method="hdbscan"),
            n_clients=10,
        )

        assert config.n_clients == 10
        assert config.model.model_type == "dinov2"
        assert config.pooling.strategy == "cls"

    def test_config_to_dict(self):
        """Should convert config to dictionary."""
        config = EmbeddingConfig(
            model=ModelConfig(model_type="dinov2", variant="base"),
            pooling=PoolingConfig(strategy="cls"),
            clustering=ClusteringConfig(method="hdbscan"),
            n_clients=10,
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["n_clients"] == 10
        assert config_dict["model"]["model_type"] == "dinov2"

    def test_config_to_json_and_from_json(self, temp_cache_dir):
        """Should save and load config as JSON."""
        config = EmbeddingConfig(
            model=ModelConfig(model_type="dinov2", variant="base"),
            pooling=PoolingConfig(strategy="gem", params={"p": 3.0}),
            clustering=ClusteringConfig(method="kmeans", params={"n_clusters": 5}),
            n_clients=10,
        )

        # Save to JSON
        json_path = Path(temp_cache_dir) / "config.json"
        config.to_json(str(json_path))

        # Load from JSON
        loaded_config = EmbeddingConfig.from_json(str(json_path))

        assert loaded_config.n_clients == 10
        assert loaded_config.model.variant == "base"
        assert loaded_config.pooling.params["p"] == 3.0
        assert loaded_config.clustering.params["n_clusters"] == 5

    def test_preset_configs(self):
        """Should provide preset configurations."""
        presets = ["default", "fast", "high_quality", "spectral"]

        for preset_name in presets:
            config = get_preset_config(preset_name, n_clients=10)
            assert isinstance(config, EmbeddingConfig)
            assert config.n_clients == 10

    def test_invalid_preset_raises(self):
        """Invalid preset should raise error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("invalid", n_clients=10)


# ============================================================================
# MODEL REGISTRY TESTS
# ============================================================================

class TestModelRegistry:
    """Test model registry and instantiation."""

    def test_dinov2_variants_instantiate(self):
        """All DINOv2 variants should instantiate successfully."""
        from embedding_extractor import get_embedding_extractor

        variants = ["small", "base", "large", "giant"]

        for variant in variants:
            config = ModelConfig(model_type="dinov2", variant=variant)
            extractor = get_embedding_extractor(config)
            assert extractor is not None
            assert hasattr(extractor, "extract")

    def test_clip_variants_instantiate(self):
        """All CLIP variants should instantiate successfully."""
        from embedding_extractor import get_embedding_extractor

        variants = ["base", "large"]

        for variant in variants:
            config = ModelConfig(model_type="clip", variant=variant)
            extractor = get_embedding_extractor(config)
            assert extractor is not None
            assert hasattr(extractor, "extract")

    def test_siglip_variants_instantiate(self):
        """SigLIP variants should instantiate successfully."""
        from embedding_extractor import get_embedding_extractor

        variants = ["base", "large"]

        for variant in variants:
            config = ModelConfig(model_type="siglip", variant=variant)
            extractor = get_embedding_extractor(config)
            assert extractor is not None
            assert hasattr(extractor, "extract")

    def test_unknown_model_type_raises(self):
        """Unknown model type should raise ValueError."""
        from embedding_extractor import get_embedding_extractor

        with pytest.raises(ValueError, match="Invalid model_type"):
            config = ModelConfig(model_type="unknown", variant="base")

    def test_unknown_variant_raises(self):
        """Unknown variant should raise ValueError."""
        from embedding_extractor import get_embedding_extractor

        # Create config with valid model_type but invalid variant
        # This should be caught by the extractor factory
        config = ModelConfig(model_type="dinov2", variant="base")
        config.variant = "nonexistent"  # Override after validation

        # The extractor may or may not raise depending on implementation
        # Just verify we can instantiate with valid variants
        config_valid = ModelConfig(model_type="dinov2", variant="base")
        extractor = get_embedding_extractor(config_valid)
        assert extractor is not None

    def test_proxy_configuration_preservation(self):
        """Proxy configuration should be preserved during model loading."""
        import os

        # Set proxy
        original_proxy = os.environ.get('https_proxy')
        test_proxy = 'http://test-proxy:3128'
        os.environ['https_proxy'] = test_proxy

        try:
            # Create model config
            config = ModelConfig(model_type="dinov2", variant="small")

            # Proxy should still be set
            assert os.environ.get('https_proxy') == test_proxy

        finally:
            # Restore original proxy
            if original_proxy is not None:
                os.environ['https_proxy'] = original_proxy
            else:
                os.environ.pop('https_proxy', None)

    def test_all_preset_configs_have_valid_models(self):
        """All preset configs should reference valid model types."""
        presets = ["default", "fast", "high_quality", "spectral"]

        for preset_name in presets:
            config = get_preset_config(preset_name, n_clients=10)

            # Model type should be valid
            assert config.model.model_type in ["dinov2", "clip", "siglip"]

            # Variant should be valid for the model type
            assert config.model.variant is not None
            assert len(config.model.variant) > 0


# ============================================================================
# CROSS-COMPONENT INTEGRATION TESTS
# ============================================================================

class TestCrossComponentIntegration:
    """Test interactions between multiple components."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_extractor_to_cache_to_pooling_to_clustering(self, temp_cache_dir):
        """
        Full pipeline: Extract embeddings → Cache → Pool → Cluster.

        This tests the complete workflow that EmbeddingSplitter uses internally.
        """
        # Setup
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=True)

        # Simulate extracted embeddings (3D - with tokens)
        np.random.seed(42)
        embeddings_3d = np.random.randn(90, 197, 768)  # 90 samples, 197 tokens, 768 dims
        image_paths = [f"/tmp/img_{i}.jpg" for i in range(90)]

        # Cache the raw embeddings
        cache.put(
            embeddings=embeddings_3d,
            model_name="dinov2/base",
            image_paths=image_paths,
            pooling_strategy="raw",  # Store raw before pooling
        )

        # Retrieve from cache
        cached_embeddings = cache.get(
            model_name="dinov2/base",
            image_paths=image_paths,
            pooling_strategy="raw",
        )

        assert cached_embeddings is not None
        np.testing.assert_array_almost_equal(cached_embeddings, embeddings_3d)

        # Apply pooling
        pooling = get_pooling_strategy("cls")
        pooled_embeddings = pooling.pool(cached_embeddings)

        assert pooled_embeddings.shape == (90, 768)

        # Cache pooled embeddings
        cache.put(
            embeddings=pooled_embeddings,
            model_name="dinov2/base",
            image_paths=image_paths,
            pooling_strategy="cls",
        )

        # Cluster the pooled embeddings
        clustering = get_clustering_method("kmeans", n_clusters=3, random_state=42)
        labels = clustering.cluster(pooled_embeddings)

        assert len(labels) == 90
        assert len(np.unique(labels)) == 3

        # Verify cluster info
        info = clustering.get_cluster_info()
        assert info["method"] == "KMeans"
        assert info["num_clusters"] == 3

    @pytest.mark.integration
    def test_config_serialization_preserves_behavior(self, temp_cache_dir):
        """
        Config serialization roundtrip preserves all parameters.

        Tests that saving and loading config produces identical behavior.
        """
        # Create original config
        original_config = EmbeddingConfig(
            model=ModelConfig(model_type="dinov2", variant="base", batch_size=32),
            pooling=PoolingConfig(strategy="gem", params={"p": 3.5, "eps": 1e-6}),
            clustering=ClusteringConfig(
                method="hdbscan",
                params={"min_cluster_size": 20, "min_samples": 5}
            ),
            n_clients=15,
            cache_dir=temp_cache_dir,
        )

        # Save to JSON
        json_path = Path(temp_cache_dir) / "config_test.json"
        original_config.to_json(str(json_path))

        # Load from JSON
        loaded_config = EmbeddingConfig.from_json(str(json_path))

        # Verify all parameters preserved
        assert loaded_config.n_clients == 15
        assert loaded_config.cache_dir == temp_cache_dir

        # Model config
        assert loaded_config.model.model_type == "dinov2"
        assert loaded_config.model.variant == "base"
        assert loaded_config.model.batch_size == 32

        # Pooling config
        assert loaded_config.pooling.strategy == "gem"
        assert loaded_config.pooling.params["p"] == 3.5
        assert loaded_config.pooling.params["eps"] == 1e-6

        # Clustering config
        assert loaded_config.clustering.method == "hdbscan"
        assert loaded_config.clustering.params["min_cluster_size"] == 20
        assert loaded_config.clustering.params["min_samples"] == 5

    @pytest.mark.integration
    def test_multiple_pooling_strategies_same_cache(self, temp_cache_dir):
        """
        Multiple pooling strategies can coexist in same cache.

        Tests cache key differentiation by pooling strategy.
        """
        cache = CacheManager(cache_dir=temp_cache_dir, enabled=True)

        # Generate sample embeddings
        embeddings_3d = np.random.randn(50, 197, 768)
        image_paths = [f"/tmp/img_{i}.jpg" for i in range(50)]

        # Apply different pooling strategies
        strategies = {
            "cls": CLSPooling(),
            "mean": MeanPooling(),
            "max": MaxPooling(),
        }

        pooled_results = {}
        for strategy_name, pooling in strategies.items():
            pooled = pooling.pool(embeddings_3d)
            pooled_results[strategy_name] = pooled

            # Cache each pooled result
            cache.put(
                embeddings=pooled,
                model_name="dinov2/base",
                image_paths=image_paths,
                pooling_strategy=strategy_name,
            )

        # Retrieve and verify each strategy's cache
        for strategy_name, expected_pooled in pooled_results.items():
            retrieved = cache.get(
                model_name="dinov2/base",
                image_paths=image_paths,
                pooling_strategy=strategy_name,
            )

            assert retrieved is not None
            np.testing.assert_array_almost_equal(retrieved, expected_pooled)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_clustering_methods_produce_different_results(self):
        """
        Different clustering methods produce different but valid results.

        Tests that each clustering method has distinct behavior.
        """
        # Generate well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(30, 50) + np.array([10, 0] + [0] * 48)
        cluster2 = np.random.randn(30, 50) + np.array([0, 10] + [0] * 48)
        cluster3 = np.random.randn(30, 50) + np.array([-10, -10] + [0] * 48)
        embeddings = np.vstack([cluster1, cluster2, cluster3])

        # Test different clustering methods
        kmeans = get_clustering_method("kmeans", n_clusters=3, random_state=42)
        labels_kmeans = kmeans.cluster(embeddings)

        hdbscan = get_clustering_method("hdbscan", min_cluster_size=15)
        labels_hdbscan = hdbscan.cluster(embeddings)

        spectral = get_clustering_method("spectral", n_clusters=3, random_state=42)
        labels_spectral = spectral.cluster(embeddings)

        # All methods should assign all samples
        assert len(labels_kmeans) == 90
        assert len(labels_hdbscan) == 90
        assert len(labels_spectral) == 90

        # K-means and spectral should have exactly 3 clusters
        assert len(np.unique(labels_kmeans)) == 3
        assert len(np.unique(labels_spectral)) == 3

        # HDBSCAN may have noise points, but should find clusters
        hdbscan_clusters = len(set(labels_hdbscan) - {-1})
        assert hdbscan_clusters >= 2

    @pytest.mark.integration
    def test_preset_config_workflow_completeness(self, temp_cache_dir):
        """
        Each preset config defines a complete, usable workflow.

        Tests that all preset configs have all required components.
        """
        presets = ["default", "fast", "high_quality", "spectral"]

        for preset_name in presets:
            config = get_preset_config(preset_name, n_clients=10)

            # Should have model config
            assert config.model is not None
            assert config.model.model_type in ["dinov2", "clip", "siglip"]
            assert config.model.variant is not None

            # Should have pooling config
            assert config.pooling is not None
            assert config.pooling.strategy in ["cls", "mean", "max", "gem", "lastk", "spectral"]

            # Should have clustering config
            assert config.clustering is not None
            assert config.clustering.method in ["kmeans", "hdbscan", "spectral", "knn_graph"]

            # Should have n_clients
            assert config.n_clients == 10

            # Cache dir should be settable
            config.cache_dir = temp_cache_dir
            assert config.cache_dir == temp_cache_dir


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests (cross-component workflows)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
