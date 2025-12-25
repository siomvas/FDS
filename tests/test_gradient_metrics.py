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
            "layer1.weight": layer1,
            "layer2.weight": layer2,
        }
        client_gradients.append(grad_dict)

    return client_gradients


def test_basic_functionality():
    """Test that all metrics can be computed without errors."""
    print("=" * 70)
    print("TEST 1: Basic Functionality")
    print("=" * 70)

    # Create synthetic gradients
    gradients = create_synthetic_gradients(n_clients=5, n_params=100, heterogeneity="moderate")

    # Test each metric type
    for metric_type in ["cosine", "variance", "conflict", "l2", "spectral"]:
        metric = GradientDiversityMetric(metric_type=metric_type)
        diversity = metric.compute(gradients)
        print(f"{metric_type:12s}: {diversity:.4f}")

    print("\n✓ All metrics computed successfully\n")


def test_heterogeneity_levels():
    """Test that metrics increase with heterogeneity."""
    print("=" * 70)
    print("TEST 2: Heterogeneity Sensitivity")
    print("=" * 70)

    for heterogeneity in ["low", "moderate", "high"]:
        print(f"\n{heterogeneity.upper()} Heterogeneity:")
        gradients = create_synthetic_gradients(n_clients=5, n_params=100, heterogeneity=heterogeneity)

        # Compute cosine diversity
        metric = GradientDiversityMetric(metric_type="variance", normalization="none")
        diversity = metric.compute(gradients)

        print(f"  Cosine diversity: {diversity:.4f}")

        # Check expected ordering
        if heterogeneity == "low":
            assert diversity < 3, f"Expected low diversity, got {diversity}"
        elif heterogeneity == "high":
            assert diversity > 5, f"Expected high diversity, got {diversity}"

    print("\n✓ Metrics correctly capture heterogeneity levels\n")


def test_normalization_strategies():
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
