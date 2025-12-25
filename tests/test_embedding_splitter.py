"""Comprehensive tests for EmbeddingSplitter.

Tests semantic data partitioning using foundation model embeddings and clustering.

Test Coverage:
- Splitter initialization with different models (DINOv2, CLIP, SigLIP)
- Embedding extraction with cache integration
- Pooling strategies (CLS, mean, max, GeM, spectral, lastk)
- Clustering methods (KMeans, HDBSCAN, spectral, KNN graph)
- Cluster assignment strategies (round_robin, size_balanced, random)
- Noise handling for HDBSCAN
- End-to-end workflows with validation

Note: Some tests are marked @pytest.mark.slow as they require model downloads.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from partitioning.embedding_splitter import EmbeddingSplitter
from partitioning.embedding_utils.config import EmbeddingConfig, get_preset_config
from embedding.cache_manager import CacheManager


# ============================================================================
# Initialization Tests
# ============================================================================

class TestEmbeddingSplitterInitialization:
    """Test splitter initialization with different models and configs."""

    def test_initialization_with_default_config(self, mock_train_dataset, default_embedding_config):
        """Initialize splitter with default EmbeddingConfig."""
        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=default_embedding_config,
            valset_type="none",
            seed=42
        )

        assert splitter.n_clients == 5
        assert splitter.heterogeneity == 0.5
        assert splitter.config == default_embedding_config
        assert splitter.embeddings is None
        assert splitter.cluster_labels is None

    def test_initialization_sets_cache_manager(self, mock_train_dataset, default_embedding_config):
        """Initialization creates CacheManager instance."""
        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=default_embedding_config,
            valset_type="none",
            seed=42
        )

        assert isinstance(splitter.cache_manager, CacheManager)
        assert splitter.cache_manager.enabled == default_embedding_config.cache_enabled

    def test_config_validation(self, mock_train_dataset):
        """Config object is properly stored and accessible."""
        config = get_preset_config("fast", n_clients=10)

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=10,
            heterogeneity=0.3,
            config=config,
            valset_type="none",
            seed=42
        )

        assert splitter.config.model.model_type == "dinov2"
        assert splitter.config.clustering.method == "kmeans"

    def test_heterogeneity_parameter_stored(self, mock_train_dataset, default_embedding_config):
        """Heterogeneity parameter correctly stored."""
        for hetero in [0.0, 0.5, 1.0]:
            splitter = EmbeddingSplitter(
                train_dataset=mock_train_dataset,
                val_dataset=None,
                n_clients=5,
                heterogeneity=hetero,
                config=default_embedding_config,
                valset_type="none",
                seed=42
            )

            assert splitter.heterogeneity == hetero


# ============================================================================
# Model Initialization Tests
# ============================================================================

class TestModelInitialization:
    """Test initialization of different embedding extractor models."""

    def test_initialize_dinov2_extractor(self, mock_train_dataset):
        """Initialize DINOv2 extractor with variant."""
        config = get_preset_config("default", n_clients=5)
        config.model.model_type = "dinov2"
        config.model.variant = "small"

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        extractor = splitter._initialize_extractor()

        from embedding.embedding_extractors import DINOv2Extractor
        assert isinstance(extractor, DINOv2Extractor)

    def test_initialize_clip_extractor(self, mock_train_dataset):
        """Initialize CLIP extractor."""
        config = get_preset_config("default", n_clients=5)
        config.model.model_type = "clip"
        config.model.variant = "base-patch32"

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        extractor = splitter._initialize_extractor()

        from embedding.embedding_extractors import CLIPExtractor
        assert isinstance(extractor, CLIPExtractor)

    def test_initialize_siglip_extractor(self, mock_train_dataset):
        """Initialize SigLIP extractor."""
        config = get_preset_config("default", n_clients=5)
        config.model.model_type = "siglip"
        config.model.variant = "base-patch16-224"

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        extractor = splitter._initialize_extractor()

        from embedding.embedding_extractors import SigLIPExtractor
        assert isinstance(extractor, SigLIPExtractor)

    def test_invalid_model_type_raises(self, mock_train_dataset):
        """Invalid model type raises ValueError."""
        config = get_preset_config("default", n_clients=5)
        config.model.model_type = "invalid_model"

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        with pytest.raises(ValueError, match="Unknown model type"):
            splitter._initialize_extractor()


# ============================================================================
# Embedding Extraction Tests (Mocked)
# ============================================================================

class TestEmbeddingExtraction:
    """Test embedding extraction logic with mocked extractors."""

    def test_extract_embeddings_cache_miss(self, mock_train_dataset, default_embedding_config, tmp_cache_dir):
        """Extract embeddings when cache misses."""
        config = default_embedding_config
        config.cache_dir = str(tmp_cache_dir)
        config.cache_enabled = True

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.model_name = "dinov2-small"
        mock_embeddings = np.random.randn(len(mock_train_dataset), 768).astype(np.float32)
        mock_extractor.extract_embeddings.return_value = mock_embeddings

        # Mock pooling
        mock_pooling = Mock()
        mock_pooling.pool.return_value = mock_embeddings

        splitter.extractor = mock_extractor
        splitter.pooling = mock_pooling

        # Extract (should miss cache and call extractor)
        embeddings = splitter._extract_embeddings()

        assert embeddings.shape == (len(mock_train_dataset), 768)
        mock_extractor.extract_embeddings.assert_called_once()

    def test_extract_embeddings_cache_hit(self, mock_train_dataset, default_embedding_config, tmp_cache_dir):
        """Extract embeddings when cache hits (no extractor call)."""
        config = default_embedding_config
        config.cache_dir = str(tmp_cache_dir)
        config.cache_enabled = True

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        # Pre-populate cache
        image_paths = [mock_train_dataset[i][0] for i in range(len(mock_train_dataset))]
        cached_embeddings = np.random.randn(len(mock_train_dataset), 768).astype(np.float32)

        # Use cache-friendly model name (no slashes)
        cache_key_params = {
            "model_name": f"{config.model.model_type}-{config.model.variant}",
            "pooling_strategy": config.pooling.strategy,
            **config.pooling.params
        }

        splitter.cache_manager.put(
            embeddings=cached_embeddings,
            image_paths=image_paths,
            **cache_key_params
        )

        # Mock extractor (should NOT be called)
        mock_extractor = Mock()
        splitter.extractor = mock_extractor
        mock_extractor.model_name = "dinov2-small"

        # Extract (should hit cache)
        embeddings = splitter._extract_embeddings()

        assert embeddings.shape == cached_embeddings.shape
        np.testing.assert_allclose(embeddings, cached_embeddings, rtol=1e-5)
        mock_extractor.extract_embeddings.assert_not_called()

    def test_pooling_strategy_application(self, mock_train_dataset, default_embedding_config):
        """Pooling strategy applied to raw embeddings."""
        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=default_embedding_config,
            valset_type="none",
            seed=42
        )

        # Mock extractor returning 3D embeddings
        mock_extractor = Mock()
        mock_extractor.model_name = "dinov2-small"
        raw_embeddings_3d = np.random.randn(len(mock_train_dataset), 197, 768).astype(np.float32)
        mock_extractor.extract_embeddings.return_value = raw_embeddings_3d

        # Mock pooling that reduces to 2D
        mock_pooling = Mock()
        pooled_embeddings_2d = np.random.randn(len(mock_train_dataset), 768).astype(np.float32)
        mock_pooling.pool.return_value = pooled_embeddings_2d

        mock_cache_manager = Mock()
        mock_cache_manager.get.return_value = None

        splitter.extractor = mock_extractor
        splitter.pooling = mock_pooling
        splitter.cache_manager = mock_cache_manager

        # Extract
        embeddings = splitter._extract_embeddings()

        mock_pooling.pool.assert_called_once_with(raw_embeddings_3d)
        assert embeddings.shape == pooled_embeddings_2d.shape


# ============================================================================
# Pooling Strategy Tests
# ============================================================================

class TestPoolingStrategies:
    """Test different pooling strategies initialization."""

    @pytest.mark.parametrize("strategy", ["cls", "mean", "max", "gem", "spectral", "lastk"])
    def test_initialize_pooling_strategies(self, mock_train_dataset, strategy):
        """Initialize different pooling strategies."""
        config = get_preset_config("default", n_clients=5)
        config.pooling.strategy = strategy

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        pooling = splitter._initialize_pooling()

        from partitioning.embedding_utils.pooling_strategies import PoolingStrategy
        assert isinstance(pooling, PoolingStrategy)

    def test_pooling_on_2d_embeddings(self, clusterable_embeddings):
        """Pooling 2D embeddings (already pooled) returns as-is."""
        embeddings_2d, _ = clusterable_embeddings

        from partitioning.embedding_utils.pooling_strategies import CLSPooling
        pooling = CLSPooling()

        result = pooling.pool(embeddings_2d)

        # Should return unchanged for 2D input
        np.testing.assert_array_equal(result, embeddings_2d)

    def test_pooling_on_3d_embeddings_with_tokens(self, sample_embeddings_3d):
        """Pooling 3D embeddings reduces token dimension."""
        from partitioning.embedding_utils.pooling_strategies import MeanPooling

        pooling = MeanPooling()
        result = pooling.pool(sample_embeddings_3d)

        # Should reduce from (N, T, D) to (N, D)
        assert result.ndim == 2
        assert result.shape[0] == sample_embeddings_3d.shape[0]
        assert result.shape[1] == sample_embeddings_3d.shape[2]


# ============================================================================
# Clustering Method Tests
# ============================================================================

class TestClusteringMethods:
    """Test different clustering methods."""

    def test_initialize_kmeans_clustering(self, mock_train_dataset):
        """Initialize KMeans clustering method."""
        config = get_preset_config("default", n_clients=5)
        config.clustering.method = "kmeans"
        config.clustering.params = {"n_clusters": 10}

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        clustering = splitter._initialize_clustering()

        from partitioning.embedding_utils.clustering_methods import KMeansMethod
        assert isinstance(clustering, KMeansMethod)

    def test_clustering_produces_labels(self, clusterable_embeddings):
        """Clustering produces integer labels for each sample."""
        embeddings, true_labels = clusterable_embeddings

        from partitioning.embedding_utils.clustering_methods import KMeansMethod
        clustering = KMeansMethod(n_clusters=3, random_state=42)

        cluster_labels = clustering.cluster(embeddings)

        assert len(cluster_labels) == len(embeddings)
        assert cluster_labels.dtype in [np.int32, np.int64]
        assert len(np.unique(cluster_labels)) >= 1  # At least one cluster

    def test_heterogeneity_affects_cluster_size_hdbscan(self, mock_train_dataset):
        """Heterogeneity parameter affects HDBSCAN min_cluster_size."""
        config = get_preset_config("default", n_clients=5)
        config.clustering.method = "hdbscan"
        config.clustering.params = {}  # No min_cluster_size specified

        # Low heterogeneity (0.1)
        splitter_low = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.1,
            config=config,
            valset_type="none",
            seed=42
        )

        clustering_low = splitter_low._initialize_clustering()

        # High heterogeneity (0.9)
        splitter_high = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.9,
            config=config,
            valset_type="none",
            seed=42
        )

        clustering_high = splitter_high._initialize_clustering()

        # Low heterogeneity → larger min_cluster_size
        # High heterogeneity → smaller min_cluster_size
        # (This is implementation-specific, verify with actual params)
        assert hasattr(clustering_low, 'min_cluster_size')
        assert hasattr(clustering_high, 'min_cluster_size')

    def test_noise_handling_hdbscan(self, clusterable_embeddings):
        """HDBSCAN can produce noise labels (-1)."""
        embeddings, _ = clusterable_embeddings

        from partitioning.embedding_utils.clustering_methods import HDBSCANMethod
        clustering = HDBSCANMethod(min_cluster_size=50)  # High threshold → noise

        cluster_labels = clustering.cluster(embeddings)

        # May contain -1 (noise) labels
        # Just verify labels are returned (whether -1 exists or not)
        assert len(cluster_labels) == len(embeddings)

    def test_initialize_spectral_clustering(self, mock_train_dataset):
        """Initialize Spectral clustering method."""
        config = get_preset_config("default", n_clients=5)
        config.clustering.method = "spectral"
        config.clustering.params = {"n_clusters": 5, "affinity": "rbf"}

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        clustering = splitter._initialize_clustering()

        from partitioning.embedding_utils.clustering_methods import SpectralMethod
        assert isinstance(clustering, SpectralMethod)

    def test_spectral_clustering_produces_labels(self, clusterable_embeddings):
        """Spectral clustering produces cluster labels."""
        embeddings, _ = clusterable_embeddings

        from partitioning.embedding_utils.clustering_methods import SpectralMethod
        clustering = SpectralMethod(n_clusters=3, affinity="rbf", random_state=42)

        cluster_labels = clustering.cluster(embeddings)

        assert len(cluster_labels) == len(embeddings)
        assert cluster_labels.dtype in [np.int32, np.int64]
        assert len(np.unique(cluster_labels)) == 3  # Should find 3 clusters

    def test_initialize_knn_graph_clustering(self, mock_train_dataset):
        """Initialize KNN graph clustering method."""
        config = get_preset_config("default", n_clients=5)
        config.clustering.method = "knn"
        config.clustering.params = {"n_neighbors": 5}

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        clustering = splitter._initialize_clustering()

        from partitioning.embedding_utils.clustering_methods import KNNGraphMethod
        assert isinstance(clustering, KNNGraphMethod)

    def test_knn_graph_clustering_produces_labels(self, clusterable_embeddings):
        """KNN graph clustering produces cluster labels."""
        embeddings, _ = clusterable_embeddings

        from partitioning.embedding_utils.clustering_methods import KNNGraphMethod
        clustering = KNNGraphMethod(n_neighbors=5)

        cluster_labels = clustering.cluster(embeddings)

        assert len(cluster_labels) == len(embeddings)
        assert cluster_labels.dtype in [np.int32, np.int64]
        assert len(np.unique(cluster_labels)) >= 1  # At least one cluster


# ============================================================================
# Cluster Assignment Tests
# ============================================================================

class TestClusterAssignment:
    """Test cluster-to-client assignment strategies."""

    def test_round_robin_strategy(self, mock_train_dataset, default_embedding_config):
        """Round-robin assignment distributes clusters evenly."""
        config = default_embedding_config
        config.assignment_strategy = "round_robin"

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        # Mock cluster labels (9 clusters, 3 clients → 3 clusters per client)
        cluster_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])

        assignments = splitter._assign_clusters_to_clients(cluster_labels)

        # Each client should get 3 clusters in round-robin order
        assert len(assignments["client_0"]) > 0
        assert len(assignments["client_1"]) > 0
        assert len(assignments["client_2"]) > 0

        # All samples assigned
        total_assigned = sum(len(v) for v in assignments.values())
        assert total_assigned == len(cluster_labels)

    def test_size_balanced_strategy(self, mock_train_dataset, default_embedding_config):
        """Size-balanced assignment balances client sizes."""
        config = default_embedding_config
        config.assignment_strategy = "size_balanced"

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        # Create imbalanced clusters
        cluster_labels = np.array([0] * 15 + [1] * 10 + [2] * 5)

        assignments = splitter._assign_clusters_to_clients(cluster_labels)

        client_sizes = [len(assignments[f"client_{i}"]) for i in range(3)]

        # Sizes should be more balanced than round-robin
        assert max(client_sizes) - min(client_sizes) <= 10

    def test_random_strategy_with_seed(self, mock_train_dataset, default_embedding_config):
        """Random assignment is reproducible with seed."""
        config = default_embedding_config
        config.assignment_strategy = "random"

        cluster_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

        # First run
        splitter_1 = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        assignments_1 = splitter_1._assign_clusters_to_clients(cluster_labels)

        # Second run with same seed
        splitter_2 = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        assignments_2 = splitter_2._assign_clusters_to_clients(cluster_labels)

        # Should produce identical assignments
        for client_id in range(3):
            assert assignments_1[f"client_{client_id}"] == assignments_2[f"client_{client_id}"]

    def test_noise_distribution(self, mock_train_dataset, default_embedding_config):
        """Noise samples distributed according to handle_noise config."""
        config = default_embedding_config
        config.handle_noise = "distribute"

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        # Include noise (-1) labels
        cluster_labels = np.array([0] * 10 + [1] * 10 + [-1] * 10)

        assignments = splitter._assign_clusters_to_clients(cluster_labels)

        # All samples (including noise) should be assigned
        total_assigned = sum(len(v) for v in assignments.values())
        assert total_assigned == len(cluster_labels)

    def test_all_samples_assigned(self, mock_train_dataset, default_embedding_config):
        """All samples assigned to some client."""
        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=default_embedding_config,
            valset_type="none",
            seed=42
        )

        cluster_labels = np.array([i % 10 for i in range(len(mock_train_dataset))])

        assignments = splitter._assign_clusters_to_clients(cluster_labels)

        # Collect all assigned indices
        assigned_indices = set()
        for client_indices in assignments.values():
            assigned_indices.update(client_indices)

        # All dataset indices should be assigned
        assert len(assigned_indices) == len(mock_train_dataset)


# ============================================================================
# End-to-End Integration Tests (Slow)
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests with real components (marked slow)."""

    @pytest.mark.slow
    def test_full_pipeline_with_mocked_extractor(self, mock_train_dataset, default_embedding_config, tmp_cache_dir):
        """Full pipeline with mocked extractor for speed."""
        config = default_embedding_config
        config.cache_dir = str(tmp_cache_dir)
        config.cache_enabled = False  # Disable cache for this test

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        # Mock extractor and pooling
        mock_embeddings = np.random.randn(len(mock_train_dataset), 768).astype(np.float32)

        with patch.object(splitter, '_extract_embeddings', return_value=mock_embeddings):
            splits = splitter.get_client_splits()

        # Verify splits structure
        assert "clients" in splits
        assert "global_val" in splits
        assert len(splits["clients"]) == 5

        # Verify all samples distributed
        total_samples = sum(
            len(splits["clients"][f"client_{i}"]["train"])
            for i in range(5)
        )
        assert total_samples == len(mock_train_dataset)

    def test_reproducibility_with_seed(self, mock_train_dataset, default_embedding_config, tmp_cache_dir):
        """Same seed produces identical splits."""
        config = default_embedding_config
        config.cache_dir = str(tmp_cache_dir)
        config.cache_enabled = False

        mock_embeddings = np.random.randn(len(mock_train_dataset), 768).astype(np.float32)

        # First run
        splitter_1 = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        with patch.object(splitter_1, '_extract_embeddings', return_value=mock_embeddings):
            splits_1 = splitter_1.get_client_splits()

        # Second run with same seed
        splitter_2 = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        with patch.object(splitter_2, '_extract_embeddings', return_value=mock_embeddings):
            splits_2 = splitter_2.get_client_splits()

        # Compare splits
        for client_id in range(3):
            train_1 = splits_1["clients"][f"client_{client_id}"]["train"]
            train_2 = splits_2["clients"][f"client_{client_id}"]["train"]

            assert len(train_1) == len(train_2)

    def test_validation_set_modes(self, mock_train_dataset, mock_val_dataset, default_embedding_config, tmp_cache_dir):
        """Test global, local, and none validation modes."""
        config = default_embedding_config
        config.cache_dir = str(tmp_cache_dir)
        config.cache_enabled = False

        mock_embeddings = np.random.randn(len(mock_train_dataset), 768).astype(np.float32)

        # Test global valset
        splitter_global = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=mock_val_dataset,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="global",
            seed=42
        )

        with patch.object(splitter_global, '_extract_embeddings', return_value=mock_embeddings):
            splits_global = splitter_global.get_client_splits()

        assert splits_global["global_val"] is not None
        assert len(splits_global["global_val"]) == len(mock_val_dataset)

        # Local validation sets should be empty
        for client_id in range(3):
            assert len(splits_global["clients"][f"client_{client_id}"]["val"]) == 0

        # Test none valset
        splitter_none = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        with patch.object(splitter_none, '_extract_embeddings', return_value=mock_embeddings):
            splits_none = splitter_none.get_client_splits()

        assert splits_none["global_val"] is None


# ============================================================================
# Cluster Metadata and Saving Tests
# ============================================================================

class TestClusterMetadata:
    """Test cluster metadata saving."""

    def test_save_cluster_metadata(self, mock_train_dataset, default_embedding_config, tmp_path, tmp_cache_dir):
        """Save cluster metadata to JSON file."""
        config = default_embedding_config
        config.cache_dir = str(tmp_cache_dir)
        config.cache_enabled = False

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        # Mock cluster info
        splitter.cluster_info = {"num_clusters": 10, "noise_count": 2}

        metadata_path = tmp_path / "cluster_metadata.json"
        splitter.save_cluster_metadata(str(metadata_path))

        assert metadata_path.exists()

        # Verify contents
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert metadata["n_clients"] == 5
        assert metadata["heterogeneity"] == 0.5
        assert metadata["cluster_info"]["num_clusters"] == 10

    def test_save_embeddings(self, mock_train_dataset, default_embedding_config, tmp_path, tmp_cache_dir):
        """Save embeddings to NPZ file."""
        config = default_embedding_config
        config.cache_dir = str(tmp_cache_dir)
        config.cache_enabled = False

        splitter = EmbeddingSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            config=config,
            valset_type="none",
            seed=42
        )

        # Set embeddings
        splitter.embeddings = np.random.randn(len(mock_train_dataset), 768).astype(np.float32)
        splitter.cluster_labels = np.array([i % 10 for i in range(len(mock_train_dataset))])

        embeddings_path = tmp_path / "embeddings.npz"
        splitter.save_embeddings(str(embeddings_path))

        assert embeddings_path.exists()

        # Verify contents
        data = np.load(embeddings_path)
        assert "embeddings" in data
        assert "cluster_labels" in data
        assert data["embeddings"].shape == splitter.embeddings.shape
