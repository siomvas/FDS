"""Contract tests for embedding utilities and configuration."""

import numpy as np
import pytest

from partitioning.embedding_utils.pooling_strategies import (
    GeMLPooling,
    LastKLayerPooling,
    MaxPooling,
    MeanPooling,
    SpectralSharpeningPooling,
    get_pooling_strategy,
)
from partitioning.embedding_utils.clustering_methods import (
    KMeansMethod,
    KNNGraphMethod,
    SpectralMethod,
    get_clustering_method,
)
from partitioning.embedding_utils.config import (
    EmbeddingConfig,
    PoolingConfig,
    create_custom_config,
    get_preset_config,
)


def _tokens(n_samples=12, n_tokens=5, dim=7):
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_samples, n_tokens, dim)).astype(np.float32)


def _layers(n_samples=10, n_layers=4, n_tokens=6, dim=8):
    rng = np.random.default_rng(7)
    return rng.normal(size=(n_samples, n_layers, n_tokens, dim)).astype(np.float32)


def _clusterable_embeddings():
    rng = np.random.default_rng(13)
    c1 = rng.normal(size=(20, 6)) + np.array([10, 0, 0, 0, 0, 0])
    c2 = rng.normal(size=(20, 6)) + np.array([-10, 0, 0, 0, 0, 0])
    c3 = rng.normal(size=(20, 6)) + np.array([0, 10, 0, 0, 0, 0])
    return np.vstack([c1, c2, c3]).astype(np.float32)


def test_mean_pooling_requires_3d_input():
    pooling = MeanPooling()
    with pytest.raises(ValueError, match="expects 3D embeddings"):
        _ = pooling.pool(np.random.randn(4, 6))


def test_mean_pooling_reduces_token_dimension():
    x = _tokens()
    out = MeanPooling().pool(x)
    assert out.shape == (12, 7)
    np.testing.assert_allclose(out, x.mean(axis=1))


def test_max_pooling_reduces_token_dimension():
    x = _tokens()
    out = MaxPooling().pool(x)
    assert out.shape == (12, 7)
    np.testing.assert_allclose(out, x.max(axis=1))


def test_gem_with_p1_matches_mean_pooling():
    x = _tokens()
    gem = GeMLPooling(p=1.0, eps=0.0).pool(x)
    mean = MeanPooling().pool(x)
    np.testing.assert_allclose(gem, mean, atol=1e-5)


def test_lastk_mean_pooling_shape():
    x = _layers()
    out = LastKLayerPooling(layer_aggregation="mean", token_aggregation="mean").pool(x)
    assert out.shape == (10, 8)


def test_lastk_concat_pooling_shape():
    x = _layers()
    out = LastKLayerPooling(layer_aggregation="concat", token_aggregation="mean").pool(x)
    assert out.shape == (10, 8 * 4)


def test_lastk_weighted_pooling_shape():
    x = _layers()
    out = LastKLayerPooling(layer_aggregation="weighted", token_aggregation="max").pool(x)
    assert out.shape == (10, 8)


def test_lastk_cls_token_aggregation_uses_first_token_after_layer_pool():
    x = _layers(n_samples=3, n_layers=2, n_tokens=4, dim=5)
    out = LastKLayerPooling(layer_aggregation="mean", token_aggregation="cls").pool(x)
    expected = x.mean(axis=1)[:, 0, :]
    np.testing.assert_allclose(out, expected)


def test_lastk_invalid_layer_aggregation_raises():
    x = _layers()
    with pytest.raises(ValueError, match="Unknown aggregation"):
        _ = LastKLayerPooling(layer_aggregation="invalid", token_aggregation="mean").pool(x)


def test_pooling_factory_rejects_cls_strategy():
    with pytest.raises(ValueError, match="Unknown pooling strategy"):
        _ = get_pooling_strategy("cls")


def test_pooling_factory_builds_gem_with_params():
    pooling = get_pooling_strategy("gem", p=4.0, eps=1e-5)
    assert isinstance(pooling, GeMLPooling)
    assert pooling.p == 4.0
    assert pooling.eps == 1e-5


def test_spectral_sharpening_returns_original_for_small_sample_count():
    x = np.random.randn(8, 16)
    out = SpectralSharpeningPooling(k=20).pool(x)
    np.testing.assert_allclose(out, x)


def test_kmeans_cluster_info_matches_requested_clusters():
    x = _clusterable_embeddings()
    method = KMeansMethod(n_clusters=3, random_state=42)
    labels = method.cluster(x)
    info = method.get_cluster_info()

    assert labels.shape[0] == x.shape[0]
    assert len(np.unique(labels)) == 3
    assert info["num_clusters"] == 3
    assert sum(info["cluster_sizes"].values()) == x.shape[0]


def test_knn_graph_method_outputs_labels_and_info():
    x = _clusterable_embeddings()
    method = KNNGraphMethod(n_neighbors=5, metric="euclidean")
    labels = method.cluster(x)
    info = method.get_cluster_info()

    assert labels.shape[0] == x.shape[0]
    assert info["num_clusters"] == len(np.unique(labels))
    assert sum(info["cluster_sizes"].values()) == x.shape[0]


def test_spectral_method_outputs_requested_cluster_count():
    x = _clusterable_embeddings()
    method = SpectralMethod(n_clusters=3, affinity="nearest_neighbors", random_state=42)
    labels = method.cluster(x)
    info = method.get_cluster_info()

    assert labels.shape[0] == x.shape[0]
    assert len(np.unique(labels)) == 3
    assert info["num_clusters"] == 3


def test_clustering_factory_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown clustering method"):
        _ = get_clustering_method("invalid")


def test_clustering_factory_requires_kmeans_params():
    with pytest.raises(TypeError):
        _ = get_clustering_method("kmeans")


def test_fast_preset_maps_n_clients_to_kmeans_clusters():
    config = get_preset_config("fast", n_clients=11)
    assert config.clustering.method == "kmeans"
    assert config.clustering.params["n_clusters"] == 11


def test_embedding_config_json_round_trip(tmp_path):
    config = get_preset_config("default", n_clients=9)
    out_path = tmp_path / "embedding_config.json"
    config.to_json(str(out_path))

    loaded = EmbeddingConfig.from_json(str(out_path))

    assert loaded.n_clients == 9
    assert loaded.model.model_type == config.model.model_type
    assert loaded.pooling.strategy == config.pooling.strategy
    assert loaded.clustering.method == config.clustering.method


def test_embedding_config_from_dict_requires_core_keys():
    with pytest.raises(KeyError, match="missing required keys"):
        EmbeddingConfig.from_dict(
            {
                "model": {"model_type": "dinov2", "variant": "base"},
                "pooling": {"strategy": "cls"},
                "n_clients": 4,
            }
        )


def test_create_custom_config_rejects_unexpected_kwargs():
    with pytest.raises(TypeError, match="Unexpected custom config kwargs"):
        create_custom_config(n_clients=4, not_a_real_option=True)


def test_pooling_config_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid pooling strategy"):
        _ = PoolingConfig(strategy="not_real")
