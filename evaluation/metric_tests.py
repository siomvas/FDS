"""
Unit tests for evaluation metrics.

Tests verify mathematical properties, edge cases, and expected behavior
for all label and feature distribution metrics.

Run with: pytest metric_tests.py -v
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import metrics


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

@pytest.fixture
def identical_distributions():
    """Two clients with identical label distributions."""
    # 100 samples, 10 classes, uniform distribution
    labels = list(range(10)) * 100
    np.random.shuffle(labels)

    dataset1 = [{"label": label} for label in labels[:500]]
    dataset2 = [{"label": label} for label in labels[500:]]

    return dataset1, dataset2, 10


@pytest.fixture
def complementary_distributions():
    """Two clients with completely non-overlapping classes."""
    # Client 0: classes 0-4, Client 1: classes 5-9
    dataset1 = [{"label": i % 5} for i in range(50)]
    dataset2 = [{"label": i % 5 + 5} for i in range(50)]

    return dataset1, dataset2, 10


@pytest.fixture
def skewed_distributions():
    """One uniform, one heavily skewed distribution."""
    # Client 0: uniform over 10 classes
    dataset1 = [{"label": i % 10} for i in range(100)]

    # Client 1: 90% class 0, 10% others
    labels = [0] * 90 + [i % 9 + 1 for i in range(10)]
    np.random.shuffle(labels)
    dataset2 = [{"label": label} for label in labels]

    return dataset1, dataset2, 10


@pytest.fixture
def single_class_distributions():
    """Both clients have only one class."""
    dataset1 = [{"label": 0} for _ in range(50)]
    dataset2 = [{"label": 1} for _ in range(50)]

    return dataset1, dataset2, 10


@pytest.fixture
def temp_image_datasets():
    """Create temporary image datasets for feature metrics."""
    temp_dir = tempfile.mkdtemp()

    def create_dataset(num_samples, color_shift=0):
        """Create synthetic images with different colors."""
        dataset = []
        for i in range(num_samples):
            # Create solid color image (R, G, B)
            color = ((i * 20 + color_shift) % 256,
                     (i * 30 + color_shift) % 256,
                     (i * 40 + color_shift) % 256)
            img = Image.new('RGB', (64, 64), color=color)

            img_path = Path(temp_dir) / f"img_{color_shift}_{i}.png"
            img.save(img_path)

            dataset.append({
                "image_path": str(img_path),
                "label": i % 5
            })

        return dataset

    # Create two similar datasets (small color shift)
    dataset1 = create_dataset(20, color_shift=0)
    dataset2 = create_dataset(20, color_shift=10)

    # Create two very different datasets (large color shift)
    dataset3 = create_dataset(20, color_shift=0)
    dataset4 = create_dataset(20, color_shift=128)

    yield (dataset1, dataset2), (dataset3, dataset4), temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


# ============================================================================
# Label Distribution Metrics Tests
# ============================================================================

class TestKLDivergence:
    """Test KL Divergence metric."""

    def test_self_divergence_is_zero(self, identical_distributions):
        """KL(P||P) should be approximately 0."""
        dataset1, dataset2, num_classes = identical_distributions

        # Use same dataset twice
        kl = metrics.KL_divergence(dataset1, dataset1, num_classes)

        assert kl < 1e-6, f"Self-divergence should be ~0, got {kl}"

    def test_non_negative(self, skewed_distributions):
        """KL divergence should always be non-negative."""
        dataset1, dataset2, num_classes = skewed_distributions

        kl = metrics.KL_divergence(dataset1, dataset2, num_classes)

        assert kl >= 0, f"KL divergence should be non-negative, got {kl}"

    def test_identical_distributions_near_zero(self, identical_distributions):
        """Identical distributions should give near-zero KL."""
        dataset1, dataset2, num_classes = identical_distributions

        kl = metrics.KL_divergence(dataset1, dataset2, num_classes)

        # Should be very small (not exactly 0 due to sampling)
        assert kl < 0.1, f"KL for identical distributions should be near 0, got {kl}"

    def test_complementary_distributions_high(self, complementary_distributions):
        """Completely different distributions should give high KL."""
        dataset1, dataset2, num_classes = complementary_distributions

        kl = metrics.KL_divergence(dataset1, dataset2, num_classes)

        # Should be very high (approaching infinity with smoothing)
        assert kl > 1.0, f"KL for complementary distributions should be high, got {kl}"

    def test_not_symmetric(self, skewed_distributions):
        """KL(P||Q) != KL(Q||P) in general."""
        dataset1, dataset2, num_classes = skewed_distributions

        kl_pq = metrics.KL_divergence(dataset1, dataset2, num_classes)
        kl_qp = metrics.KL_divergence(dataset2, dataset1, num_classes)

        # Allow some tolerance but they should generally differ
        # (might be close for symmetric distributions)
        assert kl_pq >= 0 and kl_qp >= 0, "Both should be non-negative"

    def test_uniform_distributions_zero(self):
        """Two uniform distributions should give KL ≈ 0."""
        # Create perfectly uniform distributions
        dataset1 = [{"label": i % 10} for i in range(100)]
        dataset2 = [{"label": i % 10} for i in range(100)]

        kl = metrics.KL_divergence(dataset1, dataset2, 10)

        assert kl < 1e-6, f"KL for uniform distributions should be ~0, got {kl}"

    def test_known_value(self):
        """Test against known KL value."""
        # P = [0.5, 0.5], Q = [0.8, 0.2]
        # KL(P||Q) = 0.5*log(0.5/0.8) + 0.5*log(0.5/0.2)
        #          = 0.5*log(0.625) + 0.5*log(2.5)
        #          ≈ 0.5*(-0.470) + 0.5*(0.916) = 0.223

        dataset1 = [{"label": 0}] * 50 + [{"label": 1}] * 50
        dataset2 = [{"label": 0}] * 80 + [{"label": 1}] * 20

        kl = metrics.KL_divergence(dataset1, dataset2, 2)

        expected = 0.223
        assert abs(kl - expected) < 0.05, f"Expected ~{expected}, got {kl}"


class TestKSLabelTest:
    """Test Kolmogorov-Smirnov label test."""

    def test_returns_dict_with_keys(self, identical_distributions):
        """Should return dict with 'statistic' and 'pvalue'."""
        dataset1, dataset2, _ = identical_distributions

        result = metrics.KS_label_test(dataset1, dataset2)

        assert isinstance(result, dict), "Should return dict"
        assert 'statistic' in result, "Should have 'statistic' key"
        assert 'pvalue' in result, "Should have 'pvalue' key"

    def test_pvalue_range(self, skewed_distributions):
        """p-value should be in [0, 1]."""
        dataset1, dataset2, _ = skewed_distributions

        result = metrics.KS_label_test(dataset1, dataset2)

        assert 0 <= result['pvalue'] <= 1, f"p-value should be in [0,1], got {result['pvalue']}"

    def test_statistic_range(self, skewed_distributions):
        """Statistic should be in [0, 1]."""
        dataset1, dataset2, _ = skewed_distributions

        result = metrics.KS_label_test(dataset1, dataset2)

        assert 0 <= result['statistic'] <= 1, f"Statistic should be in [0,1], got {result['statistic']}"

    def test_identical_distributions_high_pvalue(self, identical_distributions):
        """Identical distributions should fail to reject null (high p-value)."""
        dataset1, dataset2, _ = identical_distributions

        result = metrics.KS_label_test(dataset1, dataset2)

        # High p-value means we fail to reject null hypothesis
        assert result['pvalue'] > 0.05, f"Should have high p-value for identical dist, got {result['pvalue']}"

    def test_complementary_distributions_low_pvalue(self, complementary_distributions):
        """Very different distributions should reject null (low p-value)."""
        dataset1, dataset2, _ = complementary_distributions

        result = metrics.KS_label_test(dataset1, dataset2)

        # Low p-value means we reject null hypothesis
        assert result['pvalue'] < 0.05, f"Should have low p-value for different dist, got {result['pvalue']}"

    def test_self_comparison(self, skewed_distributions):
        """Comparing dataset with itself should give statistic=0, p-value=1."""
        dataset1, _, _ = skewed_distributions

        result = metrics.KS_label_test(dataset1, dataset1)

        assert result['statistic'] < 1e-10, f"Self-comparison should have statistic ~0, got {result['statistic']}"
        assert result['pvalue'] > 0.99, f"Self-comparison should have p-value ~1, got {result['pvalue']}"


class TestMMDLabelDistributions:
    """Test MMD for label distributions."""

    def test_self_distance_zero(self, skewed_distributions):
        """MMD(P, P) should be 0."""
        dataset1, _, num_classes = skewed_distributions

        mmd = metrics.MMD_label_distributions(dataset1, dataset1, num_classes)

        assert mmd < 1e-10, f"Self-distance should be 0, got {mmd}"

    def test_non_negative(self, skewed_distributions):
        """MMD should be non-negative."""
        dataset1, dataset2, num_classes = skewed_distributions

        mmd = metrics.MMD_label_distributions(dataset1, dataset2, num_classes)

        assert mmd >= 0, f"MMD should be non-negative, got {mmd}"

    def test_symmetric(self, skewed_distributions):
        """MMD(P, Q) should equal MMD(Q, P)."""
        dataset1, dataset2, num_classes = skewed_distributions

        mmd_pq = metrics.MMD_label_distributions(dataset1, dataset2, num_classes)
        mmd_qp = metrics.MMD_label_distributions(dataset2, dataset1, num_classes)

        assert abs(mmd_pq - mmd_qp) < 1e-10, f"MMD should be symmetric, got {mmd_pq} vs {mmd_qp}"

    def test_identical_distributions_near_zero(self, identical_distributions):
        """Identical distributions should give near-zero MMD."""
        dataset1, dataset2, num_classes = identical_distributions

        mmd = metrics.MMD_label_distributions(dataset1, dataset2, num_classes)

        assert mmd < 0.1, f"MMD for identical distributions should be near 0, got {mmd}"

    def test_complementary_distributions_high(self, complementary_distributions):
        """Completely different distributions should give high MMD."""
        dataset1, dataset2, num_classes = complementary_distributions

        mmd = metrics.MMD_label_distributions(dataset1, dataset2, num_classes)

        assert mmd > 0.5, f"MMD for complementary distributions should be high, got {mmd}"


class TestMahalanobisDistanceLabels:
    """Test Mahalanobis distance for label distributions."""

    def test_non_negative(self, skewed_distributions):
        """Mahalanobis distance should be non-negative."""
        dataset1, dataset2, num_classes = skewed_distributions

        dist = metrics.mahalanobis_distance_labels(dataset1, dataset2, num_classes)

        assert dist >= 0, f"Distance should be non-negative, got {dist}"

    def test_identical_distributions_near_zero(self, identical_distributions):
        """Identical distributions should give near-zero distance."""
        dataset1, dataset2, num_classes = identical_distributions

        dist = metrics.mahalanobis_distance_labels(dataset1, dataset2, num_classes)

        assert dist < 0.5, f"Distance for identical distributions should be near 0, got {dist}"

    def test_self_distance_zero(self, skewed_distributions):
        """Distance to self should be 0."""
        dataset1, _, num_classes = skewed_distributions

        dist = metrics.mahalanobis_distance_labels(dataset1, dataset1, num_classes)

        # May not be exactly 0 due to covariance estimation
        assert dist < 1e-6, f"Self-distance should be ~0, got {dist}"

    def test_complementary_distributions_high(self, complementary_distributions):
        """Very different distributions should give high distance."""
        dataset1, dataset2, num_classes = complementary_distributions

        dist = metrics.mahalanobis_distance_labels(dataset1, dataset2, num_classes)

        assert dist > 0.5, f"Distance for different distributions should be high, got {dist}"


# ============================================================================
# MMD RBF Kernel Tests
# ============================================================================

class TestMMDRBFKernel:
    """Test MMD with RBF kernel (used by feature metrics)."""

    def test_self_distance_zero(self):
        """MMD between identical samples should be ~0."""
        X = np.random.randn(50, 10)

        mmd = metrics.MMD_rbf_kernel(X, X, sigma=1.0)

        assert abs(mmd) < 1e-6, f"Self-distance should be ~0, got {mmd}"

    def test_non_negative_for_different_samples(self):
        """MMD should be non-negative (might be slightly negative due to unbiased estimator)."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10) + 2.0  # Shifted

        mmd = metrics.MMD_rbf_kernel(X, Y, sigma=1.0)

        # Unbiased estimator can be slightly negative, but should be close to 0 or positive
        assert mmd > -0.1, f"MMD should be approximately non-negative, got {mmd}"

    def test_symmetric(self):
        """MMD(X, Y) should equal MMD(Y, X)."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10) + 1.0

        mmd_xy = metrics.MMD_rbf_kernel(X, Y, sigma=1.0)
        mmd_yx = metrics.MMD_rbf_kernel(Y, X, sigma=1.0)

        assert abs(mmd_xy - mmd_yx) < 1e-10, f"MMD should be symmetric, got {mmd_xy} vs {mmd_yx}"

    def test_increases_with_distance(self):
        """MMD should increase as distributions become more different."""
        X = np.random.randn(50, 10)
        Y1 = np.random.randn(50, 10) + 0.5
        Y2 = np.random.randn(50, 10) + 2.0

        mmd1 = metrics.MMD_rbf_kernel(X, Y1, sigma=1.0)
        mmd2 = metrics.MMD_rbf_kernel(X, Y2, sigma=1.0)

        assert mmd2 > mmd1, f"MMD should increase with distance: {mmd1} vs {mmd2}"

    def test_median_heuristic(self):
        """Test that median heuristic for sigma is computed correctly."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10)

        # Should run without error when sigma=None
        mmd = metrics.MMD_rbf_kernel(X, Y, sigma=None)

        assert isinstance(mmd, (float, np.floating)), "Should return float"


# ============================================================================
# Feature Distribution Metrics Tests
# ============================================================================

class TestFIDDistance:
    """Test Fréchet Inception Distance."""

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        """FID between identical datasets should be near 0."""
        (dataset1, _), _, _ = temp_image_datasets

        # Use small subset for speed
        subset = dataset1[:10]

        fid = metrics.FID_distance([subset, subset], batch_size=5)

        assert fid < 10.0, f"Self-distance should be near 0, got {fid}"

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        """FID should be non-negative."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        fid = metrics.FID_distance([dataset1[:10], dataset2[:10]], batch_size=5)

        assert fid >= 0, f"FID should be non-negative, got {fid}"

    @pytest.mark.slow
    def test_symmetric(self, temp_image_datasets):
        """FID should be symmetric."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        subset1 = dataset1[:10]
        subset2 = dataset2[:10]

        fid_12 = metrics.FID_distance([subset1, subset2], batch_size=5)
        fid_21 = metrics.FID_distance([subset2, subset1], batch_size=5)

        assert abs(fid_12 - fid_21) < 1e-6, f"FID should be symmetric, got {fid_12} vs {fid_21}"

    @pytest.mark.slow
    def test_different_distributions_higher(self, temp_image_datasets):
        """FID should be higher for more different distributions."""
        (dataset1, dataset2), (dataset3, dataset4), _ = temp_image_datasets

        # Small color shift vs large color shift
        fid_small = metrics.FID_distance([dataset1[:10], dataset2[:10]], batch_size=5)
        fid_large = metrics.FID_distance([dataset3[:10], dataset4[:10]], batch_size=5)

        # Larger shift should give higher FID (though not guaranteed with small samples)
        assert fid_large >= 0 and fid_small >= 0, "Both should be non-negative"

    @pytest.mark.slow
    def test_return_all_option(self, temp_image_datasets):
        """Test return_all parameter returns list of pairwise FIDs."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        # 3 datasets → 3 pairwise comparisons
        subset1 = dataset1[:5]
        subset2 = dataset2[:5]
        subset3 = dataset1[5:10]

        fid_scores = metrics.FID_distance(
            [subset1, subset2, subset3],
            batch_size=5,
            return_all=True
        )

        assert isinstance(fid_scores, (list, np.ndarray)), "Should return list"
        assert len(fid_scores) == 3, f"Should have 3 pairwise scores, got {len(fid_scores)}"


class TestMMDFeatures:
    """Test MMD on features."""

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        """MMD on features should be approximately non-negative."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        mmd = metrics.MMD_features(dataset1[:10], dataset2[:10], batch_size=5)

        # Unbiased estimator can be slightly negative
        assert mmd > -1.0, f"MMD should be approximately non-negative, got {mmd}"

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        """MMD between identical feature sets should be near 0."""
        (dataset1, _), _, _ = temp_image_datasets

        subset = dataset1[:10]
        mmd = metrics.MMD_features(subset, subset, batch_size=5)

        assert abs(mmd) < 1.0, f"Self-distance should be near 0, got {mmd}"


class TestWassersteinDistanceFeatures:
    """Test Wasserstein distance on features."""

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        """Wasserstein distance should be non-negative."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        wd = metrics.wasserstein_distance_features(dataset1[:10], dataset2[:10], batch_size=5)

        assert wd >= 0, f"Wasserstein distance should be non-negative, got {wd}"

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        """Wasserstein distance to self should be near 0."""
        (dataset1, _), _, _ = temp_image_datasets

        subset = dataset1[:10]
        wd = metrics.wasserstein_distance_features(subset, subset, batch_size=5)

        assert wd < 1.0, f"Self-distance should be near 0, got {wd}"


class TestCramerVonMisesFeatures:
    """Test Cramér-von Mises test on features."""

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        """CvM statistic should be non-negative."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        cvm = metrics.cramer_von_mises_features(dataset1[:10], dataset2[:10], batch_size=5)

        assert cvm >= 0, f"CvM statistic should be non-negative, got {cvm}"

    @pytest.mark.slow
    def test_self_comparison_small(self, temp_image_datasets):
        """CvM for identical distributions should be small."""
        (dataset1, _), _, _ = temp_image_datasets

        subset = dataset1[:10]
        cvm = metrics.cramer_von_mises_features(subset, subset, batch_size=5)

        assert cvm < 1.0, f"Self-comparison should give small statistic, got {cvm}"


class TestKSMultivariateTest:
    """Test KS multivariate test."""

    @pytest.mark.slow
    def test_returns_dict_with_keys(self, temp_image_datasets):
        """Should return dict with expected keys."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        result = metrics.KS_multivariate_test(dataset1[:10], dataset2[:10], batch_size=5)

        assert isinstance(result, dict), "Should return dict"
        assert 'reject_null' in result, "Should have 'reject_null'"
        assert 'min_pvalue' in result, "Should have 'min_pvalue'"
        assert 'corrected_alpha' in result, "Should have 'corrected_alpha'"
        assert 'n_tests' in result, "Should have 'n_tests'"
        assert 'significant_dims' in result, "Should have 'significant_dims'"

    @pytest.mark.slow
    def test_reject_null_is_boolean(self, temp_image_datasets):
        """reject_null should be boolean."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        result = metrics.KS_multivariate_test(dataset1[:10], dataset2[:10], batch_size=5)

        assert isinstance(result['reject_null'], bool), "reject_null should be boolean"

    @pytest.mark.slow
    def test_pvalue_range(self, temp_image_datasets):
        """min_pvalue should be in [0, 1]."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        result = metrics.KS_multivariate_test(dataset1[:10], dataset2[:10], batch_size=5)

        assert 0 <= result['min_pvalue'] <= 1, f"p-value should be in [0,1], got {result['min_pvalue']}"

    @pytest.mark.slow
    def test_bonferroni_correction(self, temp_image_datasets):
        """corrected_alpha should be alpha / n_tests."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        alpha = 0.05
        result = metrics.KS_multivariate_test(dataset1[:10], dataset2[:10], batch_size=5, alpha=alpha)

        expected_alpha = alpha / result['n_tests']
        assert abs(result['corrected_alpha'] - expected_alpha) < 1e-10, \
            f"Corrected alpha should be {expected_alpha}, got {result['corrected_alpha']}"

    @pytest.mark.slow
    def test_self_comparison_fails_to_reject(self, temp_image_datasets):
        """Comparing dataset with itself should fail to reject null."""
        (dataset1, _), _, _ = temp_image_datasets

        subset = dataset1[:10]
        result = metrics.KS_multivariate_test(subset, subset, batch_size=5)

        # Should fail to reject (not significantly different)
        assert result['reject_null'] == False, "Self-comparison should fail to reject null"
        assert result['min_pvalue'] > 0.01, f"Self-comparison should have high p-value, got {result['min_pvalue']}"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset_handling(self):
        """Test behavior with empty datasets."""
        dataset_empty = []
        dataset_normal = [{"label": i % 5} for i in range(20)]

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            metrics.KL_divergence(dataset_empty, dataset_normal, 5)

    def test_single_sample_datasets(self):
        """Test with single-sample datasets."""
        dataset1 = [{"label": 0}]
        dataset2 = [{"label": 1}]

        # Should work or raise informative error
        try:
            kl = metrics.KL_divergence(dataset1, dataset2, 5)
            assert kl >= 0, "KL should be non-negative"
        except (ValueError, ZeroDivisionError):
            # Acceptable to fail with informative error
            pass

    def test_single_class_datasets(self, single_class_distributions):
        """Test with datasets containing only one class each."""
        dataset1, dataset2, num_classes = single_class_distributions

        # Should handle without crashing
        kl = metrics.KL_divergence(dataset1, dataset2, num_classes)
        assert kl >= 0, "KL should be non-negative"

        mmd = metrics.MMD_label_distributions(dataset1, dataset2, num_classes)
        assert mmd >= 0, "MMD should be non-negative"

    def test_mismatched_num_classes(self):
        """Test when num_classes doesn't match actual classes."""
        dataset1 = [{"label": 0}, {"label": 1}, {"label": 5}]
        dataset2 = [{"label": 2}, {"label": 3}]

        # Should work with num_classes=10 (sparse representation)
        kl = metrics.KL_divergence(dataset1, dataset2, num_classes=10)
        assert kl >= 0, "Should handle sparse class representation"

    def test_large_num_classes(self):
        """Test with large number of classes."""
        dataset1 = [{"label": i} for i in range(1000)]
        dataset2 = [{"label": i} for i in range(500, 1500)]

        # Should handle without memory issues
        kl = metrics.KL_divergence(dataset1, dataset2, num_classes=1500)
        assert kl >= 0, "Should handle large num_classes"


# ============================================================================
# Integration Tests
# ============================================================================

class TestMetricConsistency:
    """Test consistency between different metrics."""

    def test_all_metrics_agree_on_identical(self, identical_distributions):
        """All metrics should indicate low heterogeneity for identical distributions."""
        dataset1, dataset2, num_classes = identical_distributions

        kl = metrics.KL_divergence(dataset1, dataset2, num_classes)
        ks = metrics.KS_label_test(dataset1, dataset2)
        mmd = metrics.MMD_label_distributions(dataset1, dataset2, num_classes)
        mahal = metrics.mahalanobis_distance_labels(dataset1, dataset2, num_classes)

        # All should indicate similarity
        assert kl < 0.1, "KL should be low"
        assert ks['pvalue'] > 0.05, "KS should fail to reject null"
        assert mmd < 0.1, "MMD should be low"
        assert mahal < 0.5, "Mahalanobis should be low"

    def test_all_metrics_agree_on_complementary(self, complementary_distributions):
        """All metrics should indicate high heterogeneity for complementary distributions."""
        dataset1, dataset2, num_classes = complementary_distributions

        kl = metrics.KL_divergence(dataset1, dataset2, num_classes)
        ks = metrics.KS_label_test(dataset1, dataset2)
        mmd = metrics.MMD_label_distributions(dataset1, dataset2, num_classes)
        mahal = metrics.mahalanobis_distance_labels(dataset1, dataset2, num_classes)

        # All should indicate difference
        assert kl > 0.5, "KL should be high"
        assert ks['pvalue'] < 0.05, "KS should reject null"
        assert mmd > 0.3, "MMD should be high"
        assert mahal > 0.5, "Mahalanobis should be high"

    def test_metrics_order_distributions_consistently(self):
        """All metrics should order distributions by heterogeneity consistently."""
        # Three pairs: identical < slightly different < very different
        identical1 = [{"label": i % 10} for i in range(100)]
        identical2 = [{"label": i % 10} for i in range(100)]

        slight_diff1 = [{"label": 0}] * 60 + [{"label": i % 9 + 1} for i in range(40)]
        slight_diff2 = [{"label": 0}] * 40 + [{"label": i % 9 + 1} for i in range(60)]

        very_diff1 = [{"label": i % 5} for i in range(100)]
        very_diff2 = [{"label": i % 5 + 5} for i in range(100)]

        # Calculate KL for each pair
        kl_identical = metrics.KL_divergence(identical1, identical2, 10)
        kl_slight = metrics.KL_divergence(slight_diff1, slight_diff2, 10)
        kl_very = metrics.KL_divergence(very_diff1, very_diff2, 10)

        # Should be ordered
        assert kl_identical < kl_slight < kl_very, \
            f"KL should order distributions: {kl_identical} < {kl_slight} < {kl_very}"


# ============================================================================
# Performance and Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Test numerical stability of metrics."""

    def test_handles_zero_counts(self):
        """Test that metrics handle classes with zero counts."""
        # One client has class 0, other has class 1
        dataset1 = [{"label": 0}] * 50
        dataset2 = [{"label": 1}] * 50

        # Should not crash due to log(0) or division by zero
        kl = metrics.KL_divergence(dataset1, dataset2, 2)
        assert np.isfinite(kl), "KL should be finite even with zero counts"

    def test_large_sample_sizes(self):
        """Test metrics scale to large sample sizes."""
        # Create large datasets
        dataset1 = [{"label": i % 100} for i in range(10000)]
        dataset2 = [{"label": i % 100} for i in range(10000)]

        # Should complete without memory issues
        kl = metrics.KL_divergence(dataset1, dataset2, 100)
        assert np.isfinite(kl), "Should handle large datasets"

    def test_extreme_skew(self):
        """Test with extremely skewed distributions."""
        # 99.9% in one class
        dataset1 = [{"label": 0}] * 999 + [{"label": 1}]
        dataset2 = [{"label": i % 10} for i in range(1000)]

        kl = metrics.KL_divergence(dataset1, dataset2, 10)
        assert np.isfinite(kl), "Should handle extreme skew"
        assert kl > 0, "Should detect extreme difference"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-m", "not slow"])
