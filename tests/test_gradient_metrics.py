"""
Test and demonstration script for gradient diversity metrics.

This script validates the implementation and demonstrates usage patterns.
"""

import torch
import numpy as np
from gradient_metrics import (
    GradientDiversityMetric,
    compute_gradient_diversity,
    analyze_gradient_diversity,
)


def create_synthetic_gradients(
    n_clients: int,
    n_params: int,
    heterogeneity: str = "moderate"
) -> list:
    """
    Create synthetic gradient dictionaries for testing.

    Args:
        n_clients: Number of clients
        n_params: Number of parameters per layer
        heterogeneity: Level of heterogeneity
            - "low": Similar gradients (IID-like)
            - "moderate": Some differences
            - "high": Very different gradients

    Returns:
        List of gradient dictionaries
    """
    torch.manual_seed(42)
    np.random.seed(42)

    client_gradients = []

    for i in range(n_clients):
        if heterogeneity == "low":
            # All clients have similar gradients
            base = torch.randn(n_params)
            noise = torch.randn(n_params) * 0.1
            layer1 = base + noise
            layer2 = base + noise
        elif heterogeneity == "moderate":
            # Moderate differences
            base = torch.randn(n_params)
            noise = torch.randn(n_params) * 0.5
            layer1 = base + noise
            layer2 = base + noise
        elif heterogeneity == "high":
            # Each client has independent gradients
            layer1 = torch.randn(n_params)
            layer2 = torch.randn(n_params)
        else:
            raise ValueError(f"Unknown heterogeneity: {heterogeneity}")

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
        metric = GradientDiversityMetric(metric_type="cosine", normalization="l2")
        diversity = metric.compute(gradients)

        print(f"  Cosine diversity: {diversity:.4f}")

        # Check expected ordering
        if heterogeneity == "low":
            assert diversity < 0.3, f"Expected low diversity, got {diversity}"
        elif heterogeneity == "high":
            assert diversity > 0.5, f"Expected high diversity, got {diversity}"

    print("\n✓ Metrics correctly capture heterogeneity levels\n")


def test_normalization_strategies():
    """Test different normalization strategies."""
    print("=" * 70)
    print("TEST 3: Normalization Strategies")
    print("=" * 70)

    gradients = create_synthetic_gradients(n_clients=5, n_params=100, heterogeneity="moderate")

    for normalization in ["none", "l2", "layer_wise", "std"]:
        metric = GradientDiversityMetric(
            metric_type="cosine",
            normalization=normalization
        )
        diversity = metric.compute(gradients)
        print(f"{normalization:12s}: {diversity:.4f}")

    print("\n✓ All normalization strategies work\n")


def test_pairwise_similarity_matrix():
    """Test pairwise similarity matrix computation."""
    print("=" * 70)
    print("TEST 4: Pairwise Similarity Matrix")
    print("=" * 70)

    n_clients = 4
    gradients = create_synthetic_gradients(n_clients=n_clients, n_params=100, heterogeneity="moderate")

    metric = GradientDiversityMetric(metric_type="cosine")
    sim_matrix = metric.get_pairwise_similarity_matrix(gradients)

    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Expected shape: ({n_clients}, {n_clients})")
    assert sim_matrix.shape == (n_clients, n_clients), "Incorrect matrix shape"

    # Check that diagonal is 1 (self-similarity)
    diagonal = np.diag(sim_matrix)
    print(f"Diagonal (should be ~1.0): {diagonal}")
    assert np.allclose(diagonal, 1.0, atol=0.01), "Diagonal should be 1.0"

    # Check symmetry
    assert np.allclose(sim_matrix, sim_matrix.T), "Matrix should be symmetric"

    print(f"\nSimilarity matrix:\n{sim_matrix}")
    print("\n✓ Pairwise similarity matrix correct\n")


def test_layer_filtering():
    """Test that layer filtering works correctly."""
    print("=" * 70)
    print("TEST 5: Layer Filtering")
    print("=" * 70)

    # Create gradients with multiple layers
    client_gradients = []
    for i in range(3):
        grad_dict = {
            "conv1.weight": torch.randn(100),
            "conv2.weight": torch.randn(100),
            "fc1.weight": torch.randn(100),
            "fc2.weight": torch.randn(100),
        }
        client_gradients.append(grad_dict)

    # Compute with all layers
    metric_all = GradientDiversityMetric(metric_type="cosine")
    diversity_all = metric_all.compute(client_gradients)
    print(f"All layers: {diversity_all:.4f}")

    # Compute with only conv layers
    metric_conv = GradientDiversityMetric(
        metric_type="cosine",
        layer_filter=["conv"]
    )
    diversity_conv = metric_conv.compute(client_gradients)
    print(f"Conv layers only: {diversity_conv:.4f}")

    # Compute with only fc layers
    metric_fc = GradientDiversityMetric(
        metric_type="cosine",
        layer_filter=["fc"]
    )
    diversity_fc = metric_fc.compute(client_gradients)
    print(f"FC layers only: {diversity_fc:.4f}")

    print("\n✓ Layer filtering works correctly\n")


def test_comprehensive_analysis():
    """Test the comprehensive analysis function."""
    print("=" * 70)
    print("TEST 6: Comprehensive Analysis")
    print("=" * 70)

    gradients = create_synthetic_gradients(n_clients=5, n_params=100, heterogeneity="moderate")

    analysis = analyze_gradient_diversity(gradients, normalize=True)

    print("Analysis results:")
    print(f"  Number of clients: {analysis['n_clients']}")
    print(f"  Mean similarity: {analysis['mean_similarity']:.4f}")
    print(f"  Std similarity: {analysis['std_similarity']:.4f}")
    print(f"  Min similarity: {analysis['min_similarity']:.4f}")
    print(f"  Max similarity: {analysis['max_similarity']:.4f}")
    print(f"\n  All metrics:")
    for metric_name, value in analysis['metrics'].items():
        print(f"    {metric_name:12s}: {value:.4f}")

    assert "metrics" in analysis, "Missing metrics"
    assert "similarity_matrix" in analysis, "Missing similarity matrix"
    assert len(analysis['metrics']) == 5, "Should have 5 metrics"

    print("\n✓ Comprehensive analysis works\n")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=" * 70)
    print("TEST 7: Edge Cases")
    print("=" * 70)

    # Test with 1 client (should warn and return 0)
    gradients_1 = create_synthetic_gradients(n_clients=1, n_params=100, heterogeneity="moderate")
    metric = GradientDiversityMetric(metric_type="cosine")
    diversity = metric.compute(gradients_1)
    print(f"1 client: diversity = {diversity:.4f} (expected 0.0)")
    assert diversity == 0.0, "Should return 0 for single client"

    # Test with 2 clients (minimum for meaningful diversity)
    gradients_2 = create_synthetic_gradients(n_clients=2, n_params=100, heterogeneity="high")
    diversity = metric.compute(gradients_2)
    print(f"2 clients: diversity = {diversity:.4f}")
    assert diversity > 0, "Should return non-zero for 2 different clients"

    # Test with identical gradients (should give diversity ≈ 0)
    grad_dict = {"layer1": torch.ones(100)}
    identical_grads = [grad_dict, grad_dict, grad_dict]
    diversity = metric.compute(identical_grads)
    print(f"Identical gradients: diversity = {diversity:.4f} (expected ~0.0)")
    assert diversity < 0.01, "Should return ~0 for identical gradients"

    print("\n✓ Edge cases handled correctly\n")


def test_convenience_functions():
    """Test convenience functions."""
    print("=" * 70)
    print("TEST 8: Convenience Functions")
    print("=" * 70)

    gradients = create_synthetic_gradients(n_clients=5, n_params=100, heterogeneity="moderate")

    # Test quick compute function
    diversity = compute_gradient_diversity(
        gradients,
        metric_type="cosine",
        normalization="l2"
    )
    print(f"Quick compute: {diversity:.4f}")

    # Test with all metrics
    metric = GradientDiversityMetric(metric_type="cosine")
    all_metrics = metric.compute_all_metrics(gradients)

    print("\nAll metrics:")
    for name, value in all_metrics.items():
        print(f"  {name:12s}: {value:.4f}")

    print("\n✓ Convenience functions work\n")


def demo_fl_usage():
    """Demonstrate typical FL training usage."""
    print("=" * 70)
    print("DEMO: Typical FL Training Usage")
    print("=" * 70)

    print("""
    # In FL training loop:

    metric = GradientDiversityMetric(
        metric_type="cosine",
        normalization="layer_wise",
        aggregation="weighted_mean"
    )

    diversity_history = []

    for round_num in range(n_rounds):
        # 1. Clients compute local gradients
        client_gradients = []
        for client in selected_clients:
            grads = client.compute_gradients()
            client_gradients.append(grads)

        # 2. Compute gradient diversity BEFORE aggregation
        diversity = metric.compute(client_gradients)
        diversity_history.append({
            'round': round_num,
            'diversity': diversity,
            'n_clients': len(client_gradients)
        })

        # 3. Aggregate and update
        aggregated = aggregate_gradients(client_gradients)
        global_model.apply_gradients(aggregated)

    # 4. Analyze diversity evolution
    import matplotlib.pyplot as plt
    rounds = [x['round'] for x in diversity_history]
    diversities = [x['diversity'] for x in diversity_history]
    plt.plot(rounds, diversities)
    plt.xlabel('FL Round')
    plt.ylabel('Gradient Diversity')
    plt.title('Gradient Diversity Evolution')
    plt.show()
    """)

    # Simulate this with synthetic data
    print("Simulating 10 FL rounds with decreasing heterogeneity...")
    metric = GradientDiversityMetric(metric_type="cosine", normalization="layer_wise")

    diversity_history = []
    for round_num in range(10):
        # Simulate gradients becoming more similar over time
        heterogeneity = "high" if round_num < 3 else "moderate" if round_num < 7 else "low"
        gradients = create_synthetic_gradients(n_clients=5, n_params=100, heterogeneity=heterogeneity)

        diversity = metric.compute(gradients)
        diversity_history.append({
            'round': round_num,
            'diversity': diversity
        })

        print(f"  Round {round_num}: diversity = {diversity:.4f}")

    print("\n✓ Typical FL usage demonstrated\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING GRADIENT DIVERSITY METRICS TESTS")
    print("=" * 70 + "\n")

    try:
        test_basic_functionality()
        test_heterogeneity_levels()
        test_normalization_strategies()
        test_pairwise_similarity_matrix()
        test_layer_filtering()
        test_comprehensive_analysis()
        test_edge_cases()
        test_convenience_functions()
        demo_fl_usage()

        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
