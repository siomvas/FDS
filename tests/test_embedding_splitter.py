"""Contract tests for the current EmbeddingSplitter API."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from embedding.cache_manager import CacheManager
from partitioning.embedding_splitter import EmbeddingSplitter


def _make_splitter(train_dataset, val_dataset=None, **overrides):
    params = {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "extractor_name": "dinov2-base",
        "clustering_method": "kmeans",
        "clustering_params": {"n_clusters": 3, "random_state": 42},
        "token_extraction_strategy": "cls",
        "token_params": {},
        "pooling_strategy": "mean",
        "pooling_params": {},
        "preprocessing_params": {},
        "valset_type": "none" if val_dataset is None else "global",
        "seed": 7,
        "cache_dir": "/tmp/fds_embedding_cache_test",
        "dataset_name": "tiny_imagenet",
        "base_path": "/tmp",
        "device": "cpu",
        "batch_size": 4,
        "use_cache": False,
        "num_workers": 0,
    }
    params.update(overrides)
    return EmbeddingSplitter(**params)


def test_cls_token_strategy_forces_cls_pooling(mock_train_dataset):
    splitter = _make_splitter(
        mock_train_dataset,
        token_extraction_strategy="cls",
        pooling_strategy="max",
    )
    assert splitter.pooling_strategy == "cls"


def test_non_cls_defaults_to_mean_pooling_when_none(mock_train_dataset):
    splitter = _make_splitter(
        mock_train_dataset,
        token_extraction_strategy="all_last",
        pooling_strategy=None,
    )
    assert splitter.pooling_strategy == "mean"


def test_invalid_token_extraction_raises(mock_train_dataset):
    with pytest.raises(ValueError, match="Invalid token extraction"):
        _make_splitter(mock_train_dataset, token_extraction_strategy="bad_tokens")


def test_k_last_extraction_requires_token_params(mock_train_dataset):
    with pytest.raises(ValueError, match="token_params must include"):
        _make_splitter(
            mock_train_dataset,
            token_extraction_strategy="k_last_all",
            token_params={"lastk_k": 2},
        )


def test_invalid_clustering_method_raises(mock_train_dataset):
    with pytest.raises(ValueError, match="Invalid clustering"):
        _make_splitter(mock_train_dataset, clustering_method="bad_clusterer")


def test_get_extractor_passes_runtime_args(mock_train_dataset):
    splitter = _make_splitter(mock_train_dataset)

    with patch("partitioning.embedding_splitter.ModelRegistry.create") as create_mock:
        create_mock.return_value = object()
        _ = splitter._get_extractor()

    kwargs = create_mock.call_args.kwargs
    assert create_mock.call_args.args[0] == "dinov2-base"
    assert kwargs["device"] == "cpu"
    assert kwargs["batch_size"] == 4
    assert kwargs["num_workers"] == 0
    assert isinstance(kwargs["cache_manager"], CacheManager)


def test_get_embeddings_skips_pooling_for_cls(mock_train_dataset):
    splitter = _make_splitter(mock_train_dataset, token_extraction_strategy="cls", pooling_strategy="mean")
    n_samples = len(mock_train_dataset)
    raw = np.random.randn(n_samples, 8)

    extractor = Mock()
    extractor.extract_embeddings.return_value = raw

    with patch.object(splitter, "_get_extractor", return_value=extractor):
        with patch("partitioning.embedding_splitter.get_pooling_strategy") as pooling_factory:
            out = splitter._get_embeddings(mock_train_dataset)

    np.testing.assert_array_equal(out, raw)
    pooling_factory.assert_not_called()

    kwargs = extractor.extract_embeddings.call_args.kwargs
    assert kwargs["return_which_tokens"] == "cls"
    assert kwargs["k_last"] == -1
    assert kwargs["use_cache"] is False


def test_get_embeddings_applies_pooling_for_non_cls(mock_train_dataset):
    splitter = _make_splitter(
        mock_train_dataset,
        token_extraction_strategy="all_last",
        pooling_strategy="mean",
    )
    n_samples = len(mock_train_dataset)
    raw = np.random.randn(n_samples, 5, 8)
    pooled = np.random.randn(n_samples, 8)

    extractor = Mock()
    extractor.extract_embeddings.return_value = raw
    pooler = Mock()
    pooler.pool.return_value = pooled

    with patch.object(splitter, "_get_extractor", return_value=extractor):
        with patch("partitioning.embedding_splitter.get_pooling_strategy", return_value=pooler) as pooling_factory:
            out = splitter._get_embeddings(mock_train_dataset)

    np.testing.assert_array_equal(out, pooled)
    pooling_factory.assert_called_once_with("mean")


def test_preprocess_embeddings_requires_params(mock_train_dataset):
    splitter = _make_splitter(mock_train_dataset, preprocessing_params={})
    with pytest.raises(ValueError, match="No preprocessing_params"):
        _ = splitter._preprocess_embeddings(np.random.randn(4, 3), fit=True)


def test_preprocess_embeddings_applies_l2_normalization(mock_train_dataset):
    splitter = _make_splitter(
        mock_train_dataset,
        preprocessing_params={"normalization": "l2", "dim_reduction": "none"},
    )
    x = np.array([[3.0, 4.0], [0.0, 2.0]])
    out = splitter._preprocess_embeddings(x, fit=True)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms))


def test_preprocess_embeddings_umap_fit_then_transform(mock_train_dataset):
    splitter = _make_splitter(
        mock_train_dataset,
        preprocessing_params={
            "dim_reduction": "umap",
            "dr_dims": 2,
            "umap_neighbors": 3,
            "umap_min_dist": 0.1,
            "umap_metric": "euclidean",
        },
    )

    class DummyUMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, embeddings):
            return embeddings[:, :2] + 1.0

        def transform(self, embeddings):
            return embeddings[:, :2] - 1.0

    x = np.random.randn(6, 4)

    with patch("partitioning.embedding_splitter.umap.UMAP", DummyUMAP):
        fit_out = splitter._preprocess_embeddings(x, fit=True)
        transform_out = splitter._preprocess_embeddings(x, fit=False)

    assert fit_out.shape == (6, 2)
    assert transform_out.shape == (6, 2)


def test_preprocess_embeddings_umap_transform_requires_fit(mock_train_dataset):
    splitter = _make_splitter(
        mock_train_dataset,
        preprocessing_params={
            "dim_reduction": "umap",
            "dr_dims": 2,
            "umap_neighbors": 3,
            "umap_min_dist": 0.1,
            "umap_metric": "euclidean",
        },
    )

    with pytest.raises(ValueError, match="UMAP reducer not fitted"):
        _ = splitter._preprocess_embeddings(np.random.randn(4, 4), fit=False)


def test_cluster_rejects_non_2d_embeddings(mock_train_dataset):
    splitter = _make_splitter(mock_train_dataset)
    with pytest.raises(ValueError, match="Embeddings must be 2D"):
        _ = splitter._cluster(np.random.randn(4, 2, 2))


def test_cluster_uses_factory_and_stores_cluster_info(mock_train_dataset):
    splitter = _make_splitter(mock_train_dataset, clustering_method="kmeans", clustering_params={"n_clusters": 2})

    clusterer = Mock()
    clusterer.cluster.return_value = np.array([0, 1, 0, 1])
    clusterer.get_cluster_info.return_value = {"num_clusters": 2, "method": "KMeans"}

    with patch("partitioning.embedding_splitter.get_clustering_method", return_value=clusterer) as clustering_factory:
        labels = splitter._cluster(np.random.randn(4, 3))

    np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
    assert splitter.cluster_info_["num_clusters"] == 2
    clustering_factory.assert_called_once_with("kmeans", n_clusters=2)


def test_assign_noise_to_nearest_cluster(mock_train_dataset):
    splitter = _make_splitter(mock_train_dataset)
    embeddings = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
        [0.2, -0.1],
        [9.8, 9.9],
    ])
    labels = np.array([0, 1, -1, -1])

    out = splitter._assign_noise_to_nearest(embeddings, labels)
    np.testing.assert_array_equal(out, np.array([0, 1, 0, 1]))


def test_assign_noise_when_all_points_noise(mock_train_dataset):
    splitter = _make_splitter(mock_train_dataset)
    labels = np.array([-1, -1, -1])
    out = splitter._assign_noise_to_nearest(np.random.randn(3, 2), labels)
    np.testing.assert_array_equal(out, np.array([0, 0, 0]))


def test_compute_client_splits_global_val(mock_train_dataset, mock_val_dataset):
    splitter = _make_splitter(
        mock_train_dataset,
        val_dataset=mock_val_dataset,
        valset_type="global",
    )

    n_train = len(mock_train_dataset)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    train_embeddings = np.random.randn(n_train, 4)

    with patch.object(splitter, "_get_embeddings", return_value=train_embeddings):
        with patch.object(splitter, "_cluster", return_value=labels):
            splits = splitter.get_client_splits()

    assert splitter.n_clients == 3
    assert splits["global_val"] is not None
    assert len(splits["global_val"]) == len(mock_val_dataset)

    total_train = sum(len(v["train"]) for v in splits["clients"].values())
    total_val = sum(len(v["val"]) for v in splits["clients"].values())
    assert total_train == len(mock_train_dataset)
    assert total_val == 0
    assert splitter.validate_splits(splits) is True


def test_compute_client_splits_local_val(mock_train_dataset, mock_val_dataset):
    splitter = _make_splitter(
        mock_train_dataset,
        val_dataset=mock_val_dataset,
        valset_type="local",
    )

    n_train = len(mock_train_dataset)
    n_val = len(mock_val_dataset)
    train_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

    train_embeddings = np.vstack([
        np.random.randn(10, 2) + np.array([0.0, 0.0]),
        np.random.randn(10, 2) + np.array([10.0, 0.0]),
        np.random.randn(10, 2) + np.array([0.0, 10.0]),
    ])

    val_embeddings = np.vstack([
        np.random.randn(5, 2) + np.array([0.1, 0.1]),
        np.random.randn(5, 2) + np.array([9.9, 0.2]),
        np.random.randn(5, 2) + np.array([0.2, 9.8]),
    ])
    assert train_embeddings.shape[0] == n_train
    assert val_embeddings.shape[0] == n_val

    with patch.object(splitter, "_get_embeddings", side_effect=[train_embeddings, val_embeddings]):
        with patch.object(splitter, "_cluster", return_value=train_labels):
            splits = splitter.get_client_splits()

    assert splitter.n_clients == 3
    assert splits["global_val"] is None

    total_train = sum(len(v["train"]) for v in splits["clients"].values())
    total_val = sum(len(v["val"]) for v in splits["clients"].values())
    assert total_train == len(mock_train_dataset)
    assert total_val == len(mock_val_dataset)
    assert splitter.validate_splits(splits) is True


def test_compute_client_splits_without_valset(mock_train_dataset):
    splitter = _make_splitter(mock_train_dataset, val_dataset=None, valset_type="none")
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

    with patch.object(splitter, "_get_embeddings", return_value=np.random.randn(len(mock_train_dataset), 3)):
        with patch.object(splitter, "_cluster", return_value=labels):
            splits = splitter.get_client_splits()

    assert splitter.n_clients == 3
    assert splits["global_val"] is None
    assert sum(len(v["val"]) for v in splits["clients"].values()) == 0
    assert splitter.validate_splits(splits) is True
