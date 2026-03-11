import numpy as np
import pytest
import torch

from evaluation.gradient_metrics import (
    GradientDiversityMetric,
    analyze_gradient_diversity,
    compute_gradient_diversity,
    compute_pairwise_cosine_similarity,
    compute_pairwise_l2_distance,
    flatten_gradients,
    normalize_gradient,
)


@pytest.fixture
def simple_gradients():
    torch.manual_seed(42)
    return [
        {
            "layer1.weight": torch.randn(4, 4),
            "layer2.weight": torch.randn(2, 2),
        }
        for _ in range(4)
    ]


@pytest.fixture
def zero_gradients():
    return [
        {
            "layer1.weight": torch.zeros(4, 4),
            "layer2.weight": torch.zeros(2, 2),
        }
        for _ in range(3)
    ]


def test_flatten_gradients_respects_layer_filter(simple_gradients):
    flat = flatten_gradients(simple_gradients[0], layer_filter=["layer1"])
    assert flat.shape == (16,)


def test_flatten_gradients_raises_when_filter_matches_nothing(simple_gradients):
    with pytest.raises(ValueError, match="No gradients found"):
        flatten_gradients(simple_gradients[0], layer_filter=["missing"])


@pytest.mark.parametrize("method", ["none", "l2", "std"])
def test_normalize_gradient_returns_finite_values(method):
    grad = torch.tensor([1.0, 2.0, 3.0])
    normalized = normalize_gradient(grad, method=method)
    assert torch.isfinite(normalized).all()


def test_pairwise_helpers_return_square_matrices(simple_gradients):
    flattened = [flatten_gradients(grad) for grad in simple_gradients[:3]]
    cosine = compute_pairwise_cosine_similarity(flattened)
    l2 = compute_pairwise_l2_distance(flattened)

    assert cosine.shape == (3, 3)
    assert l2.shape == (3, 3)


@pytest.mark.parametrize("metric_type", GradientDiversityMetric.VALID_METRICS)
def test_gradient_diversity_metric_returns_finite_value(simple_gradients, metric_type):
    metric = GradientDiversityMetric(metric_type=metric_type)
    value = metric.compute(simple_gradients)

    assert isinstance(value, float)
    assert np.isfinite(value)


@pytest.mark.parametrize("aggregation", GradientDiversityMetric.VALID_AGGREGATIONS)
def test_layer_wise_metric_supports_all_aggregations(simple_gradients, aggregation):
    metric = GradientDiversityMetric(
        metric_type="cosine",
        normalization="layer_wise",
        aggregation=aggregation,
    )
    assert np.isfinite(metric.compute(simple_gradients))


def test_single_client_warns_and_returns_zero(simple_gradients):
    metric = GradientDiversityMetric(metric_type="cosine")
    with pytest.warns(UserWarning):
        value = metric.compute(simple_gradients[:1])
    assert value == 0.0


def test_zero_gradients_are_supported(zero_gradients):
    metric = GradientDiversityMetric(metric_type="cosine")
    assert np.isfinite(metric.compute(zero_gradients))


def test_layer_filter_error_surfaces_in_similarity_matrix(simple_gradients):
    metric = GradientDiversityMetric(
        metric_type="cosine",
        normalization="layer_wise",
        layer_filter=["missing"],
    )
    with pytest.raises(ValueError, match="No gradients found"):
        metric.get_pairwise_similarity_matrix(simple_gradients)


def test_compute_gradient_diversity_wrapper(simple_gradients):
    value = compute_gradient_diversity(simple_gradients, metric_type="variance", normalization="none")
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_analyze_gradient_diversity_returns_expected_structure(simple_gradients):
    analysis = analyze_gradient_diversity(simple_gradients, normalize=True)

    assert set(analysis) == {
        "metrics",
        "similarity_matrix",
        "pairwise_similarities",
        "mean_similarity",
        "std_similarity",
        "min_similarity",
        "max_similarity",
        "n_clients",
    }
    assert analysis["similarity_matrix"].shape == (len(simple_gradients), len(simple_gradients))
    assert set(analysis["metrics"]) == set(GradientDiversityMetric.VALID_METRICS)


def test_sampling_parameters_still_computes(simple_gradients):
    metric = GradientDiversityMetric(metric_type="cosine", sample_params=0.25)
    assert np.isfinite(metric.compute(simple_gradients))
