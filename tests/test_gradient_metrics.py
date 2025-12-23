"""
Unit tests for gradient diversity metrics.

Tests verify correct behavior of gradient diversity metrics used to quantify
dynamic heterogeneity during federated learning training.
"""

import pytest
import torch
import numpy as np
from evaluation.gradient_metrics import (
    GradientDiversityMetric,
    compute_gradient_diversity,
    analyze_gradient_diversity,
    flatten_gradients,
    normalize_gradient,
    compute_pairwise_cosine_similarity,
    compute_pairwise_l2_distance,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_gradients():
    """Create simple synthetic gradients for testing."""
    torch.manual_seed(42)
    gradients = []
    for i in range(5):
        grad_dict = {
            "layer1.weight": torch.randn(10, 20),
            "layer2.weight": torch.randn(20, 10),
        }
        gradients.append(grad_dict)
    return gradients


@pytest.fixture
def identical_gradients():
    """Create identical gradients (zero diversity)."""
    grad_dict = {
        "layer1.weight": torch.ones(10, 20),
        "layer2.weight": torch.ones(20, 10) * 0.5,
    }
    return [grad_dict, grad_dict.copy(), grad_dict.copy()]


@pytest.fixture
def orthogonal_gradients():
    """Create orthogonal gradients (minimal conflict, moderate diversity)."""
    gradients = []
    for i in range(5):
        # Create gradients that are orthogonal in flattened space
        grad = torch.zeros(100)
        grad[i * 20:(i + 1) * 20] = 1.0
        grad_dict = {"layer1.weight": grad.view(10, 10)}
        gradients.append(grad_dict)
    return gradients


@pytest.fixture
def opposing_gradients():
    """Create opposing gradients (high conflict)."""
    grad1 = {"layer1.weight": torch.ones(10, 10)}
    grad2 = {"layer1.weight": -torch.ones(10, 10)}
    return [grad1, grad2]


@pytest.fixture
def zero_gradients():
    """Create zero gradients (edge case)."""
    grad_dict = {"layer1.weight": torch.zeros(10, 10)}
    return [grad_dict, grad_dict.copy(), grad_dict.copy()]


@pytest.fixture
def single_client_gradient():
    """Create single client gradient (edge case)."""
    grad_dict = {"layer1.weight": torch.randn(10, 10)}
    return [grad_dict]


# ============================================================================
# Test Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Test helper functions for gradient processing."""

    def test_flatten_gradients(self, simple_gradients):
        """Test gradient flattening."""
        grad_dict = simple_gradients[0]
        flat = flatten_gradients(grad_dict)

        # Should concatenate all gradients
        expected_size = 10 * 20 + 20 * 10
        assert flat.shape == (expected_size,)
        assert flat.dtype == torch.float32 or flat.dtype == torch.float64

    def test_flatten_gradients_with_filter(self, simple_gradients):
        """Test gradient flattening with layer filter."""
        grad_dict = simple_gradients[0]
        flat = flatten_gradients(grad_dict, layer_filter=["layer1"])

        # Should only include layer1
        expected_size = 10 * 20
        assert flat.shape == (expected_size,)

    def test_flatten_gradients_empty(self):
        """Test that empty gradients raise ValueError."""
        grad_dict = {}
        with pytest.raises(ValueError, match="No gradients found"):
            flatten_gradients(grad_dict)

    def test_normalize_gradient_l2(self):
        """Test L2 normalization."""
        grad = torch.tensor([3.0, 4.0])
        normalized = normalize_gradient(grad, method="l2")

        # Should have unit norm
        norm = torch.norm(normalized)
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-6)

    def test_normalize_gradient_std(self):
        """Test standardization normalization."""
        grad = torch.randn(100)
        normalized = normalize_gradient(grad, method="std")

        # Should have zero mean and unit std
        assert torch.abs(normalized.mean()) < 1e-6
        assert torch.abs(normalized.std() - 1.0) < 1e-6

    def test_normalize_gradient_none(self):
        """Test no normalization."""
        grad = torch.tensor([1.0, 2.0, 3.0])
        normalized = normalize_gradient(grad, method="none")

        # Should be unchanged
        assert torch.equal(normalized, grad)

    def test_normalize_gradient_zero(self):
        """Test normalization of zero gradient."""
        grad = torch.zeros(10)

        # Should not raise error
        normalized_l2 = normalize_gradient(grad, method="l2")
        normalized_std = normalize_gradient(grad, method="std")

        assert torch.equal(normalized_l2, grad)
        assert torch.equal(normalized_std, grad)

    def test_compute_pairwise_cosine_similarity(self):
        """Test pairwise cosine similarity computation."""
        grads = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 1.0]),
        ]

        sim_matrix = compute_pairwise_cosine_similarity(grads, normalize=True)

        # Should be 3x3
        assert sim_matrix.shape == (3, 3)

        # Diagonal should be 1 (self-similarity)
        assert np.allclose(np.diag(sim_matrix), 1.0, atol=1e-5)

        # Should be symmetric
        assert np.allclose(sim_matrix, sim_matrix.T)

    def test_compute_pairwise_l2_distance(self):
        """Test pairwise L2 distance computation."""
        grads = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        ]

        dist_matrix = compute_pairwise_l2_distance(grads)

        # Should be 3x3
        assert dist_matrix.shape == (3, 3)

        # Diagonal should be 0 (self-distance)
        assert np.allclose(np.diag(dist_matrix), 0.0, atol=1e-6)

        # Should be symmetric
        assert np.allclose(dist_matrix, dist_matrix.T)


# ============================================================================
# Test GradientDiversityMetric Class
# ============================================================================

class TestGradientDiversityMetric:
    """Test GradientDiversityMetric class."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        metric = GradientDiversityMetric(
            metric_type="cosine",
            normalization="layer_wise",
            aggregation="weighted_mean"
        )
        assert metric.metric_type == "cosine"
        assert metric.normalization == "layer_wise"
        assert metric.aggregation == "weighted_mean"

    def test_initialization_invalid_metric_type(self):
        """Test that invalid metric_type raises ValueError."""
        with pytest.raises(ValueError, match="metric_type must be one of"):
            GradientDiversityMetric(metric_type="invalid")

    def test_initialization_invalid_normalization(self):
        """Test that invalid normalization raises ValueError."""
        with pytest.raises(ValueError, match="normalization must be one of"):
            GradientDiversityMetric(normalization="invalid")

    def test_initialization_invalid_aggregation(self):
        """Test that invalid aggregation raises ValueError."""
        with pytest.raises(ValueError, match="aggregation must be one of"):
            GradientDiversityMetric(aggregation="invalid")

    def test_initialization_invalid_sample_params(self):
        """Test that invalid sample_params raises ValueError."""
        with pytest.raises(ValueError, match="sample_params must be between 0 and 1"):
            GradientDiversityMetric(sample_params=1.5)

    def test_compute_basic(self, simple_gradients):
        """Test basic diversity computation."""
        metric = GradientDiversityMetric(metric_type="cosine")
        diversity = metric.compute(simple_gradients)

        # Should return a float
        assert isinstance(diversity, float)

        # Cosine diversity should be in [0, 2]
        assert 0.0 <= diversity <= 2.0

    def test_compute_with_return_details(self, simple_gradients):
        """Test compute with return_details=True."""
        metric = GradientDiversityMetric(metric_type="cosine")
        result = metric.compute(simple_gradients, return_details=True)

        # Should return dict
        assert isinstance(result, dict)
        assert "diversity" in result
        assert "n_clients" in result
        assert "metric_type" in result
        assert "normalization" in result
        assert result["n_clients"] == 5

    def test_compute_single_client(self, single_client_gradient):
        """Test that single client returns 0 diversity with warning."""
        metric = GradientDiversityMetric(metric_type="cosine")

        with pytest.warns(UserWarning, match="Need at least 2 clients"):
            diversity = metric.compute(single_client_gradient)

        assert diversity == 0.0

    def test_compute_identical_gradients(self, identical_gradients):
        """Test that identical gradients give zero diversity."""
        metric = GradientDiversityMetric(metric_type="cosine")
        diversity = metric.compute(identical_gradients)

        # Should be very close to 0
        assert diversity < 0.01

    def test_compute_zero_gradients(self, zero_gradients):
        """Test handling of zero gradients (edge case)."""
        metric = GradientDiversityMetric(metric_type="cosine")

        # Should handle gracefully without crashing
        diversity = metric.compute(zero_gradients)

        # Result should be numeric (not NaN or Inf)
        assert isinstance(diversity, float)
        assert not np.isnan(diversity)
        assert not np.isinf(diversity)

    def test_compute_all_metrics(self, simple_gradients):
        """Test computing all metric types."""
        metric = GradientDiversityMetric()
        all_metrics = metric.compute_all_metrics(simple_gradients)

        # Should have all 5 metric types
        assert len(all_metrics) == 5
        assert "cosine" in all_metrics
        assert "variance" in all_metrics
        assert "conflict" in all_metrics
        assert "l2" in all_metrics
        assert "spectral" in all_metrics

        # All should be numeric
        for value in all_metrics.values():
            assert isinstance(value, float)
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_get_pairwise_similarity_matrix(self, simple_gradients):
        """Test pairwise similarity matrix computation."""
        metric = GradientDiversityMetric(metric_type="cosine")
        sim_matrix = metric.get_pairwise_similarity_matrix(simple_gradients)

        # Should be square matrix
        n_clients = len(simple_gradients)
        assert sim_matrix.shape == (n_clients, n_clients)

        # Diagonal should be ~1 (self-similarity)
        diagonal = np.diag(sim_matrix)
        assert np.allclose(diagonal, 1.0, atol=0.1)

        # Should be symmetric
        assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-5)


# ============================================================================
# Test Different Metric Types
# ============================================================================

class TestCosineMetric:
    """Test cosine diversity metric."""

    def test_identical_gradients_zero_diversity(self, identical_gradients):
        """Test that identical gradients give ~0 diversity."""
        metric = GradientDiversityMetric(metric_type="cosine")
        diversity = metric.compute(identical_gradients)
        assert diversity < 0.01

    def test_orthogonal_gradients(self, orthogonal_gradients):
        """Test orthogonal gradients give moderate diversity."""
        metric = GradientDiversityMetric(metric_type="cosine")
        diversity = metric.compute(orthogonal_gradients)

        # Orthogonal gradients should have diversity around 1
        # (since 1 - mean_cos_sim, and orthogonal means cos_sim ≈ 0)
        assert 0.8 <= diversity <= 1.2

    def test_opposing_gradients_high_diversity(self, opposing_gradients):
        """Test opposing gradients give high diversity."""
        metric = GradientDiversityMetric(metric_type="cosine")
        diversity = metric.compute(opposing_gradients)

        # Opposing gradients (cos_sim = -1) should give diversity = 2
        assert diversity > 1.8


class TestConflictMetric:
    """Test conflict diversity metric."""

    def test_orthogonal_gradients_minimal_conflict(self, orthogonal_gradients):
        """Test that orthogonal gradients have minimal conflict."""
        metric = GradientDiversityMetric(metric_type="conflict")
        conflict = metric.compute(orthogonal_gradients)

        # Orthogonal gradients (cos_sim ≈ 0) should have low conflict rate
        assert conflict < 0.2

    def test_opposing_gradients_high_conflict(self, opposing_gradients):
        """Test that opposing gradients have high conflict."""
        metric = GradientDiversityMetric(metric_type="conflict")
        conflict = metric.compute(opposing_gradients)

        # Opposing gradients (cos_sim = -1) should have conflict = 1
        assert conflict > 0.9

    def test_aligned_gradients_no_conflict(self, identical_gradients):
        """Test that aligned gradients have no conflict."""
        metric = GradientDiversityMetric(metric_type="conflict")
        conflict = metric.compute(identical_gradients)

        # Aligned gradients (cos_sim = 1) should have conflict = 0
        assert conflict < 0.01

    def test_single_client_no_conflict(self, single_client_gradient):
        """Test that single client has no conflict."""
        metric = GradientDiversityMetric(metric_type="conflict")

        with pytest.warns(UserWarning):
            conflict = metric.compute(single_client_gradient)

        assert conflict == 0.0


class TestVarianceMetric:
    """Test variance diversity metric."""

    def test_identical_gradients_zero_variance(self, identical_gradients):
        """Test that identical gradients give zero variance."""
        metric = GradientDiversityMetric(metric_type="variance")
        variance = metric.compute(identical_gradients)

        assert variance < 1e-6

    def test_diverse_gradients_positive_variance(self, simple_gradients):
        """Test that diverse gradients give positive variance."""
        metric = GradientDiversityMetric(metric_type="variance")
        variance = metric.compute(simple_gradients)

        assert variance > 0


class TestL2Metric:
    """Test L2 distance diversity metric."""

    def test_identical_gradients_zero_distance(self, identical_gradients):
        """Test that identical gradients give zero L2 distance."""
        metric = GradientDiversityMetric(metric_type="l2")
        distance = metric.compute(identical_gradients)

        assert distance < 1e-6

    def test_diverse_gradients_positive_distance(self, simple_gradients):
        """Test that diverse gradients give positive L2 distance."""
        metric = GradientDiversityMetric(metric_type="l2")
        distance = metric.compute(simple_gradients)

        assert distance > 0


class TestSpectralMetric:
    """Test spectral diversity metric."""

    def test_identical_gradients_zero_spectral(self, identical_gradients):
        """Test that identical gradients give low spectral diversity."""
        metric = GradientDiversityMetric(metric_type="spectral")
        spectral = metric.compute(identical_gradients)

        # Identical gradients have low spectral diversity
        assert spectral < 0.1

    def test_diverse_gradients_positive_spectral(self, simple_gradients):
        """Test that diverse gradients give positive spectral diversity."""
        metric = GradientDiversityMetric(metric_type="spectral")
        spectral = metric.compute(simple_gradients)

        # Should be in [0, 1]
        assert 0.0 <= spectral <= 1.0


# ============================================================================
# Test Normalization Strategies
# ============================================================================

class TestNormalizationStrategies:
    """Test different normalization strategies."""

    @pytest.mark.parametrize("normalization", ["none", "l2", "layer_wise", "std"])
    def test_all_normalizations_work(self, simple_gradients, normalization):
        """Test that all normalization strategies work without error."""
        metric = GradientDiversityMetric(
            metric_type="cosine",
            normalization=normalization
        )
        diversity = metric.compute(simple_gradients)

        # Should return valid numeric result
        assert isinstance(diversity, float)
        assert not np.isnan(diversity)
        assert not np.isinf(diversity)

    def test_normalization_affects_results(self, simple_gradients):
        """Test that different normalizations give different results."""
        results = {}
        for normalization in ["none", "l2", "layer_wise", "std"]:
            metric = GradientDiversityMetric(
                metric_type="cosine",
                normalization=normalization
            )
            results[normalization] = metric.compute(simple_gradients)

        # Should have at least some variation
        assert len(set(results.values())) >= 2

    def test_zero_gradients_with_different_normalizations(self, zero_gradients):
        """Test zero gradients with different normalizations."""
        for normalization in ["none", "l2", "layer_wise", "std"]:
            metric = GradientDiversityMetric(
                metric_type="cosine",
                normalization=normalization
            )

            # Should not crash
            diversity = metric.compute(zero_gradients)
            assert not np.isnan(diversity)
            assert not np.isinf(diversity)


# ============================================================================
# Test Aggregation Strategies
# ============================================================================

class TestAggregationStrategies:
    """Test different aggregation strategies."""

    @pytest.mark.parametrize("aggregation", ["mean", "weighted_mean", "max"])
    def test_all_aggregations_work(self, simple_gradients, aggregation):
        """Test that all aggregation strategies work without error."""
        metric = GradientDiversityMetric(
            metric_type="cosine",
            normalization="layer_wise",
            aggregation=aggregation
        )
        diversity = metric.compute(simple_gradients)

        # Should return valid numeric result
        assert isinstance(diversity, float)
        assert not np.isnan(diversity)
        assert not np.isinf(diversity)

    def test_aggregation_affects_results(self, simple_gradients):
        """Test that different aggregations can give different results."""
        results = {}
        for aggregation in ["mean", "weighted_mean", "max"]:
            metric = GradientDiversityMetric(
                metric_type="cosine",
                normalization="layer_wise",
                aggregation=aggregation
            )
            results[aggregation] = metric.compute(simple_gradients)

        # All should be valid
        for value in results.values():
            assert isinstance(value, float)
            assert not np.isnan(value)


# ============================================================================
# Test Layer Filtering
# ============================================================================

class TestLayerFiltering:
    """Test layer filtering functionality."""

    def test_layer_filter_single_layer(self, simple_gradients):
        """Test filtering to single layer."""
        metric_all = GradientDiversityMetric(metric_type="cosine")
        metric_layer1 = GradientDiversityMetric(
            metric_type="cosine",
            layer_filter=["layer1"]
        )

        diversity_all = metric_all.compute(simple_gradients)
        diversity_layer1 = metric_layer1.compute(simple_gradients)

        # Both should be valid
        assert isinstance(diversity_all, float)
        assert isinstance(diversity_layer1, float)

    def test_layer_filter_multiple_layers(self):
        """Test filtering to multiple layers."""
        # Create gradients with multiple layers
        gradients = []
        for i in range(3):
            grad_dict = {
                "conv1.weight": torch.randn(10, 10),
                "conv2.weight": torch.randn(10, 10),
                "fc1.weight": torch.randn(10, 10),
                "fc2.weight": torch.randn(10, 10),
            }
            gradients.append(grad_dict)

        metric_conv = GradientDiversityMetric(
            metric_type="cosine",
            layer_filter=["conv"]
        )
        metric_fc = GradientDiversityMetric(
            metric_type="cosine",
            layer_filter=["fc"]
        )

        diversity_conv = metric_conv.compute(gradients)
        diversity_fc = metric_fc.compute(gradients)

        # Both should be valid
        assert isinstance(diversity_conv, float)
        assert isinstance(diversity_fc, float)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_client_returns_zero(self, single_client_gradient):
        """Test that single client returns 0 diversity."""
        metric = GradientDiversityMetric(metric_type="cosine")

        with pytest.warns(UserWarning):
            diversity = metric.compute(single_client_gradient)

        assert diversity == 0.0

    def test_two_clients_minimum(self):
        """Test with 2 clients (minimum for meaningful diversity)."""
        grad1 = {"layer1": torch.randn(10, 10)}
        grad2 = {"layer1": torch.randn(10, 10)}
        gradients = [grad1, grad2]

        metric = GradientDiversityMetric(metric_type="cosine")
        diversity = metric.compute(gradients)

        # Should return non-zero for different clients
        assert diversity > 0

    def test_large_number_of_clients(self):
        """Test with large number of clients."""
        gradients = []
        for i in range(100):
            grad_dict = {"layer1": torch.randn(10, 10)}
            gradients.append(grad_dict)

        metric = GradientDiversityMetric(metric_type="cosine")

        # Should handle without error
        diversity = metric.compute(gradients)
        assert isinstance(diversity, float)
        assert not np.isnan(diversity)
        assert not np.isinf(diversity)

    def test_zero_gradients_all_metrics(self, zero_gradients):
        """Test zero gradients with all metric types."""
        for metric_type in ["cosine", "variance", "conflict", "l2", "spectral"]:
            metric = GradientDiversityMetric(metric_type=metric_type)

            # Should not crash
            diversity = metric.compute(zero_gradients)
            assert not np.isnan(diversity)
            assert not np.isinf(diversity)

    def test_empty_layer_filter(self, simple_gradients):
        """Test with empty layer filter (no matching layers)."""
        metric = GradientDiversityMetric(
            metric_type="cosine",
            layer_filter=["nonexistent"]
        )

        # Should raise error when no layers match
        with pytest.raises(ValueError, match="No gradients found"):
            metric.compute(simple_gradients)


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compute_gradient_diversity(self, simple_gradients):
        """Test compute_gradient_diversity convenience function."""
        diversity = compute_gradient_diversity(
            simple_gradients,
            metric_type="cosine",
            normalization="l2"
        )

        assert isinstance(diversity, float)
        assert not np.isnan(diversity)

    def test_analyze_gradient_diversity(self, simple_gradients):
        """Test analyze_gradient_diversity convenience function."""
        analysis = analyze_gradient_diversity(simple_gradients, normalize=True)

        # Should have all expected keys
        assert "metrics" in analysis
        assert "similarity_matrix" in analysis
        assert "mean_similarity" in analysis
        assert "std_similarity" in analysis
        assert "min_similarity" in analysis
        assert "max_similarity" in analysis
        assert "n_clients" in analysis

        # Should have all 5 metrics
        assert len(analysis["metrics"]) == 5

        # Similarity matrix should be correct shape
        n_clients = len(simple_gradients)
        assert analysis["similarity_matrix"].shape == (n_clients, n_clients)

    def test_analyze_gradient_diversity_no_normalize(self, simple_gradients):
        """Test analyze_gradient_diversity without normalization."""
        analysis = analyze_gradient_diversity(simple_gradients, normalize=False)

        # Should still work
        assert "metrics" in analysis
        assert len(analysis["metrics"]) == 5


# ============================================================================
# Test Parameter Sampling
# ============================================================================

class TestParameterSampling:
    """Test parameter sampling functionality."""

    def test_sample_params_reduces_computation(self, simple_gradients):
        """Test that parameter sampling works."""
        metric = GradientDiversityMetric(
            metric_type="cosine",
            sample_params=0.1
        )

        # Should work without error
        diversity = metric.compute(simple_gradients)
        assert isinstance(diversity, float)

    @pytest.mark.parametrize("sample_fraction", [0.1, 0.5, 0.9])
    def test_different_sample_fractions(self, simple_gradients, sample_fraction):
        """Test different sampling fractions."""
        metric = GradientDiversityMetric(
            metric_type="cosine",
            sample_params=sample_fraction
        )

        diversity = metric.compute(simple_gradients)
        assert isinstance(diversity, float)
        assert not np.isnan(diversity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
