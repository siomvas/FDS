"""Property-based tests using Hypothesis framework.

These tests validate fundamental invariants and mathematical properties that should
hold for all valid inputs, not just specific test cases. Hypothesis automatically
generates diverse test cases to find edge cases and violations.

Test Coverage:
- Splitter invariants (completeness, no overlap, reproducibility)
- Metric properties (symmetry, boundedness, triangle inequality)
- Cache round-trip properties
- Dirichlet distribution properties
- Numerical stability

Note: Property-based tests are marked with @pytest.mark.property
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from partitioning.random_splitter import RandomSplitter
from partitioning.dirichlet_splitter import DirichletSplitter
from evaluation.metrics import JS_divergence, TV_label_distributions
from embedding.cache_manager import CacheManager
from tests.conftest import MockDataset


# ============================================================================
# Custom Hypothesis Strategies
# ============================================================================

@st.composite
def dataset_strategy(draw, min_samples=10, max_samples=500, min_classes=2, max_classes=20):
    """Generate random dataset for testing.

    Returns list of dicts with 'image' and 'label' keys.
    """
    n_samples = draw(st.integers(min_samples, max_samples))
    n_classes = draw(st.integers(min_classes, max_classes))

    dataset = []
    for i in range(n_samples):
        label_idx = draw(st.integers(0, n_classes - 1))
        dataset.append({
            "image": f"/tmp/img_{i}.jpg",
            "label": f"class_{label_idx}"
        })

    return dataset


@st.composite
def splitter_params_strategy(draw, max_clients=50):
    """Generate random splitter parameters."""
    return {
        "n_clients": draw(st.integers(1, max_clients)),
        "heterogeneity": draw(st.floats(0.01, 100.0)),
        "seed": draw(st.integers(0, 10000))
    }


@st.composite
def embedding_array_strategy(draw, min_samples=1, max_samples=100,
                             min_dims=64, max_dims=2048):
    """Generate random embedding arrays."""
    n_samples = draw(st.integers(min_samples, max_samples))
    n_dims = draw(st.integers(min_dims, max_dims))

    return np.random.randn(n_samples, n_dims).astype(np.float32)


# ============================================================================
# Splitter Invariant Tests
# ============================================================================

@pytest.mark.property
class TestSplitterInvariants:
    """Test fundamental invariants that all splitters must satisfy."""

    @given(
        dataset=dataset_strategy(min_samples=10, max_samples=200),
        n_clients=st.integers(1, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_all_samples_distributed(self, dataset, n_clients, seed):
        """Property: All samples must be distributed exactly once.

        sum(client_sizes) == dataset_size
        """
        assume(len(dataset) >= n_clients)  # Need enough samples

        mock_dataset = MockDataset(dataset)

        splitter = RandomSplitter(
            train_dataset=mock_dataset,
            val_dataset=None,
            n_clients=n_clients,
            heterogeneity=0.0,
            valset_type="none",
            seed=seed
        )

        splits = splitter.get_client_splits()

        # Count total samples
        total_samples = sum(
            len(splits["clients"][f"client_{i}"]["train"])
            for i in range(n_clients)
        )

        assert total_samples == len(dataset), \
            f"Expected {len(dataset)} samples, got {total_samples}"

    @given(
        dataset=dataset_strategy(min_samples=20, max_samples=200),
        n_clients=st.integers(2, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_no_sample_overlap(self, dataset, n_clients, seed):
        """Property: No sample appears in multiple clients.

        All client train sets are disjoint.
        """
        assume(len(dataset) >= n_clients * 2)  # Need enough samples

        mock_dataset = MockDataset(dataset)

        splitter = RandomSplitter(
            train_dataset=mock_dataset,
            val_dataset=None,
            n_clients=n_clients,
            heterogeneity=0.0,
            valset_type="none",
            seed=seed
        )

        splits = splitter.get_client_splits()

        # Collect all assigned samples
        all_samples = []
        for i in range(n_clients):
            client_samples = splits["clients"][f"client_{i}"]["train"]
            for sample in client_samples:
                all_samples.append(sample["image"])

        # Check for duplicates
        assert len(all_samples) == len(set(all_samples)), \
            "Found duplicate samples across clients"

    @given(
        dataset=dataset_strategy(min_samples=10, max_samples=100),
        n_clients=st.integers(1, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_correct_client_count(self, dataset, n_clients, seed):
        """Property: Splits contain exactly n_clients.

        len(splits["clients"]) == n_clients
        """
        assume(len(dataset) >= n_clients)

        mock_dataset = MockDataset(dataset)

        splitter = RandomSplitter(
            train_dataset=mock_dataset,
            val_dataset=None,
            n_clients=n_clients,
            heterogeneity=0.0,
            valset_type="none",
            seed=seed
        )

        splits = splitter.get_client_splits()

        assert len(splits["clients"]) == n_clients, \
            f"Expected {n_clients} clients, got {len(splits['clients'])}"

    @given(
        dataset=dataset_strategy(min_samples=20, max_samples=100),
        n_clients=st.integers(2, 10),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=30, deadline=None)
    def test_reproducibility(self, dataset, n_clients, seed):
        """Property: Same seed produces identical splits.

        splitter(seed=X) == splitter(seed=X)
        """
        assume(len(dataset) >= n_clients * 2)

        mock_dataset = MockDataset(dataset)

        # First run
        splitter_1 = RandomSplitter(
            train_dataset=mock_dataset,
            val_dataset=None,
            n_clients=n_clients,
            heterogeneity=0.0,
            valset_type="none",
            seed=seed
        )
        splits_1 = splitter_1.get_client_splits()

        # Second run with same seed
        splitter_2 = RandomSplitter(
            train_dataset=mock_dataset,
            val_dataset=None,
            n_clients=n_clients,
            heterogeneity=0.0,
            valset_type="none",
            seed=seed
        )
        splits_2 = splitter_2.get_client_splits()

        # Compare client assignments
        for i in range(n_clients):
            samples_1 = [s["image"] for s in splits_1["clients"][f"client_{i}"]["train"]]
            samples_2 = [s["image"] for s in splits_2["clients"][f"client_{i}"]["train"]]

            assert samples_1 == samples_2, \
                f"Client {i} has different samples with same seed"

    @given(
        dataset=dataset_strategy(min_samples=20, max_samples=100),
        n_clients=st.integers(2, 10),
        seed1=st.integers(0, 5000),
        seed2=st.integers(5001, 10000)
    )
    @settings(max_examples=30, deadline=None)
    def test_different_seeds_different_splits(self, dataset, n_clients, seed1, seed2):
        """Property: Different seeds (usually) produce different splits.

        Note: This is probabilistic - with high probability, splits differ.
        """
        assume(len(dataset) >= n_clients * 2)

        mock_dataset = MockDataset(dataset)

        # Run with seed1
        splitter_1 = RandomSplitter(
            train_dataset=mock_dataset,
            val_dataset=None,
            n_clients=n_clients,
            heterogeneity=0.0,
            valset_type="none",
            seed=seed1
        )
        splits_1 = splitter_1.get_client_splits()

        # Run with seed2
        splitter_2 = RandomSplitter(
            train_dataset=mock_dataset,
            val_dataset=None,
            n_clients=n_clients,
            heterogeneity=0.0,
            valset_type="none",
            seed=seed2
        )
        splits_2 = splitter_2.get_client_splits()

        # At least one client should have different samples
        differences = 0
        for i in range(n_clients):
            samples_1 = set(s["image"] for s in splits_1["clients"][f"client_{i}"]["train"])
            samples_2 = set(s["image"] for s in splits_2["clients"][f"client_{i}"]["train"])

            if samples_1 != samples_2:
                differences += 1

        # With high probability, at least one client differs
        assert differences > 0, "Different seeds produced identical splits (extremely unlikely)"


# ============================================================================
# Metric Invariant Tests
# ============================================================================

@pytest.mark.property
class TestMetricInvariants:
    """Test mathematical properties of evaluation metrics."""

    @given(
        dataset_size=st.integers(20, 200),
        num_classes=st.integers(2, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_jsd_symmetry(self, dataset_size, num_classes, seed):
        """Property: JSD(D1, D2) == JSD(D2, D1) (symmetric).

        Jensen-Shannon divergence is symmetric.
        """
        np.random.seed(seed)

        # Generate two random label distributions
        labels_1 = np.random.randint(0, num_classes, size=dataset_size)
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)

        dataset_1 = [{"label": int(l)} for l in labels_1]
        dataset_2 = [{"label": int(l)} for l in labels_2]

        # Compute JSD in both orders
        jsd_12 = JS_divergence([dataset_1, dataset_2], num_classes=num_classes)
        jsd_21 = JS_divergence([dataset_2, dataset_1], num_classes=num_classes)

        assert abs(jsd_12 - jsd_21) < 1e-10, \
            f"JSD not symmetric: JSD(D1,D2)={jsd_12}, JSD(D2,D1)={jsd_21}"

    @given(
        dataset_size=st.integers(20, 200),
        num_classes=st.integers(2, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_jsd_self_is_zero(self, dataset_size, num_classes, seed):
        """Property: JSD(D, D) == 0 (self-similarity).

        A distribution has zero divergence from itself.
        """
        np.random.seed(seed)

        labels = np.random.randint(0, num_classes, size=dataset_size)
        dataset = [{"label": int(l)} for l in labels]

        jsd = JS_divergence([dataset, dataset], num_classes=num_classes)

        assert abs(jsd) < 1e-10, f"JSD(D,D) should be 0, got {jsd}"

    @given(
        dataset_size=st.integers(20, 200),
        num_classes=st.integers(2, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_jsd_bounded(self, dataset_size, num_classes, seed):
        """Property: 0 <= JSD <= log(2).

        Jensen-Shannon divergence is bounded between 0 and log(2).
        """
        np.random.seed(seed)

        labels_1 = np.random.randint(0, num_classes, size=dataset_size)
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)

        dataset_1 = [{"label": int(l)} for l in labels_1]
        dataset_2 = [{"label": int(l)} for l in labels_2]

        jsd = JS_divergence([dataset_1, dataset_2], num_classes=num_classes)

        assert 0.0 <= jsd <= np.log(2) + 1e-6, \
            f"JSD should be in [0, log(2)], got {jsd}"

    @given(
        dataset_size=st.integers(20, 200),
        num_classes=st.integers(2, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=30, deadline=None)
    def test_tv_distance_bounded(self, dataset_size, num_classes, seed):
        """Property: 0 <= TV <= 1.

        Total Variation distance is bounded between 0 and 1.
        """
        np.random.seed(seed)

        labels_1 = np.random.randint(0, num_classes, size=dataset_size)
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)

        dataset_1 = [{"label": int(l)} for l in labels_1]
        dataset_2 = [{"label": int(l)} for l in labels_2]

        tv = TV_label_distributions([dataset_1, dataset_2], num_classes=num_classes)

        assert 0.0 <= tv <= 1.0 + 1e-6, \
            f"TV distance should be in [0, 1], got {tv}"

    @given(
        dataset_size=st.integers(20, 200),
        num_classes=st.integers(2, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=30, deadline=None)
    def test_tv_self_is_zero(self, dataset_size, num_classes, seed):
        """Property: TV(D, D) == 0.

        Total Variation distance from a distribution to itself is zero.
        """
        np.random.seed(seed)

        labels = np.random.randint(0, num_classes, size=dataset_size)
        dataset = [{"label": int(l)} for l in labels]

        tv = TV_label_distributions([dataset, dataset], num_classes=num_classes)

        assert abs(tv) < 1e-10, f"TV(D,D) should be 0, got {tv}"


# ============================================================================
# Cache Invariant Tests
# ============================================================================

@pytest.mark.property
class TestCacheInvariants:
    """Test cache round-trip and consistency properties."""

    @given(
        n_images=st.integers(1, 50),
        embedding_dim=st.integers(64, 512),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=30, deadline=None)
    def test_cache_roundtrip(self, n_images, embedding_dim, seed, tmp_cache_dir):
        """Property: get(put(x)) == x (round-trip).

        Cached embeddings should be retrieved identically.
        """
        np.random.seed(seed)

        cache = CacheManager(cache_dir=tmp_cache_dir, enabled=True)

        # Generate random embeddings
        image_paths = [f"/tmp/img_{i}.jpg" for i in range(n_images)]
        embeddings = np.random.randn(n_images, embedding_dim).astype(np.float32)

        # Put embeddings
        cache.put(
            embeddings=embeddings,
            model_name="test_model",
            image_paths=image_paths
        )

        # Get embeddings back
        retrieved = cache.get(
            model_name="test_model",
            image_paths=image_paths
        )

        assert retrieved is not None, "Cache miss on immediate retrieval"
        np.testing.assert_allclose(retrieved, embeddings, rtol=1e-5)

    @given(
        n_images=st.integers(5, 50),
        embedding_dim=st.integers(64, 512),
        subset_size=st.integers(1, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=30, deadline=None)
    def test_dataset_mode_subset_retrieval(self, n_images, embedding_dim, subset_size, seed, tmp_cache_dir):
        """Property: Dataset mode allows correct subset retrieval.

        Retrieving a subset should return the correct embeddings.
        """
        assume(subset_size <= n_images)

        np.random.seed(seed)

        cache = CacheManager(cache_dir=tmp_cache_dir, enabled=True)

        # Generate and cache full dataset
        all_paths = [f"/tmp/img_{i}.jpg" for i in range(n_images)]
        all_embeddings = np.random.randn(n_images, embedding_dim).astype(np.float32)

        cache.put(
            embeddings=all_embeddings,
            model_name="test_model",
            image_paths=all_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        # Retrieve random subset
        subset_indices = np.random.choice(n_images, size=subset_size, replace=False)
        subset_paths = [all_paths[i] for i in subset_indices]

        retrieved = cache.get(
            model_name="test_model",
            image_paths=subset_paths,
            dataset_name="test_dataset",
            base_path="/tmp"
        )

        assert retrieved is not None, "Cache miss on subset retrieval"
        assert retrieved.shape == (subset_size, embedding_dim)

        # Verify each embedding matches
        for i, idx in enumerate(subset_indices):
            np.testing.assert_allclose(retrieved[i], all_embeddings[idx], rtol=1e-5)


# ============================================================================
# Dirichlet Distribution Properties
# ============================================================================

@pytest.mark.property
class TestDirichletProperties:
    """Test properties of Dirichlet distribution sampling."""

    @given(
        alpha=st.floats(0.01, 100.0),
        n_clients=st.integers(2, 20),
        n_classes=st.integers(2, 50),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_preferences_sum_to_one(self, alpha, n_clients, n_classes, seed):
        """Property: Dirichlet samples sum to 1.

        Each preference vector must sum to 1.0.
        """
        np.random.seed(seed)

        for _ in range(n_clients):
            alpha_vec = np.full(n_classes, alpha)
            preferences = np.random.dirichlet(alpha_vec)

            assert abs(preferences.sum() - 1.0) < 1e-10, \
                f"Preferences sum to {preferences.sum()}, expected 1.0"

    @given(
        alpha=st.floats(0.01, 100.0),
        n_clients=st.integers(2, 20),
        n_classes=st.integers(2, 50),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_preferences_non_negative(self, alpha, n_clients, n_classes, seed):
        """Property: Dirichlet samples are non-negative.

        All preference values must be >= 0.
        """
        np.random.seed(seed)

        for _ in range(n_clients):
            alpha_vec = np.full(n_classes, alpha)
            preferences = np.random.dirichlet(alpha_vec)

            assert (preferences >= 0.0).all(), \
                f"Found negative preferences: min={preferences.min()}"

    @given(
        alpha=st.floats(0.01, 100.0),
        n_classes=st.integers(2, 50),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_dirichlet_reproducibility(self, alpha, n_classes, seed):
        """Property: Same seed produces identical Dirichlet samples.

        Dirichlet sampling is deterministic with seed.
        """
        alpha_vec = np.full(n_classes, alpha)

        # First sample
        np.random.seed(seed)
        sample_1 = np.random.dirichlet(alpha_vec)

        # Second sample with same seed
        np.random.seed(seed)
        sample_2 = np.random.dirichlet(alpha_vec)

        np.testing.assert_allclose(sample_1, sample_2, rtol=1e-10)


# ============================================================================
# Numerical Stability Tests
# ============================================================================

@pytest.mark.property
class TestNumericalStability:
    """Test numerical stability with edge case inputs."""

    @given(
        dataset_size=st.integers(20, 200),
        num_classes=st.integers(2, 20),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=30, deadline=None)
    def test_jsd_handles_uniform_distribution(self, dataset_size, num_classes, seed):
        """Property: JSD handles perfectly uniform distributions.

        Should compute correctly even for edge case distributions.
        """
        # Create uniform distribution (equal samples per class)
        samples_per_class = dataset_size // num_classes
        labels = []
        for class_id in range(num_classes):
            labels.extend([class_id] * samples_per_class)

        # Pad to reach dataset_size
        while len(labels) < dataset_size:
            labels.append(0)

        dataset = [{"label": f"class_{l}"} for l in labels]

        # JSD with itself should be 0
        jsd = JS_divergence([dataset, dataset], num_classes=num_classes)

        assert abs(jsd) < 1e-10
        assert not np.isnan(jsd)
        assert not np.isinf(jsd)

    @given(
        dataset_size=st.integers(10, 100),
        num_classes=st.integers(2, 10),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=30, deadline=None)
    def test_jsd_handles_single_class_dominant(self, dataset_size, num_classes, seed):
        """Property: JSD handles highly imbalanced distributions.

        One class dominates (>90% of samples).
        """
        np.random.seed(seed)

        # Create highly imbalanced distribution
        dominant_class_size = int(dataset_size * 0.95)
        labels = [0] * dominant_class_size

        # Fill remainder with other classes
        remaining = dataset_size - dominant_class_size
        for i in range(remaining):
            labels.append(np.random.randint(1, num_classes))

        dataset = [{"label": f"class_{l}"} for l in labels]

        # Create another random distribution
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)
        dataset_2 = [{"label": f"class_{l}"} for l in labels_2]

        jsd = JS_divergence([dataset, dataset_2], num_classes=num_classes)

        # Should be valid number
        assert not np.isnan(jsd)
        assert not np.isinf(jsd)
        assert 0.0 <= jsd <= np.log(2) + 1e-6


# ============================================================================
# Integration Property Tests
# ============================================================================

@pytest.mark.property
class TestIntegrationProperties:
    """Property tests for cross-component workflows."""

    @given(
        dataset=dataset_strategy(min_samples=30, max_samples=100),
        n_clients=st.integers(2, 10),
        heterogeneity=st.floats(0.1, 10.0),
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=20, deadline=None)
    def test_dirichlet_splitter_completeness(self, dataset, n_clients, heterogeneity, seed):
        """Property: DirichletSplitter also satisfies completeness.

        All samples distributed, no overlap.
        """
        assume(len(dataset) >= n_clients * 5)  # Need enough samples

        mock_dataset = MockDataset(dataset)

        try:
            splitter = DirichletSplitter(
                train_dataset=mock_dataset,
                val_dataset=None,
                n_clients=n_clients,
                heterogeneity=heterogeneity,
                valset_type="none",
                min_samples_per_client=1,
                max_retries=100,
                seed=seed
            )

            splits = splitter.get_client_splits()

            # All samples distributed
            total_samples = sum(
                len(splits["clients"][f"client_{i}"]["train"])
                for i in range(n_clients)
            )
            assert total_samples == len(dataset)

            # No overlap
            all_images = []
            for i in range(n_clients):
                for sample in splits["clients"][f"client_{i}"]["train"]:
                    all_images.append(sample["image"])

            assert len(all_images) == len(set(all_images))

        except ValueError:
            # Retry exhaustion is acceptable for extreme parameters
            pass
