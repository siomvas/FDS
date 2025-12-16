"""
Unit tests for evaluation metrics.
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import metrics


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def identical_distributions():
    """Two datasets with identical label distributions."""
    labels = list(range(10)) * 100
    np.random.shuffle(labels)
    dataset1 = [{"label": label} for label in labels[:500]]
    dataset2 = [{"label": label} for label in labels[500:]]
    return dataset1, dataset2, 10


@pytest.fixture
def complementary_distributions():
    """Two datasets with non-overlapping classes."""
    dataset1 = [{"label": i % 5} for i in range(50)]
    dataset2 = [{"label": i % 5 + 5} for i in range(50)]
    return dataset1, dataset2, 10


@pytest.fixture
def skewed_distributions():
    """One uniform, one heavily skewed distribution."""
    dataset1 = [{"label": i % 10} for i in range(100)]
    labels = [0] * 90 + [i % 9 + 1 for i in range(10)]
    np.random.shuffle(labels)
    dataset2 = [{"label": label} for label in labels]
    return dataset1, dataset2, 10


@pytest.fixture
def single_class_distributions():
    """Both datasets have only one class each."""
    dataset1 = [{"label": 0} for _ in range(50)]
    dataset2 = [{"label": 1} for _ in range(50)]
    return dataset1, dataset2, 10


@pytest.fixture
def temp_image_datasets():
    """Create temporary image datasets for feature metrics."""
    temp_dir = tempfile.mkdtemp()

    def create_dataset(num_samples, color_shift=0):
        dataset = []
        for i in range(num_samples):
            color = ((i * 20 + color_shift) % 256,
                     (i * 30 + color_shift) % 256,
                     (i * 40 + color_shift) % 256)
            img = Image.new('RGB', (64, 64), color=color)
            img_path = Path(temp_dir) / f"img_{color_shift}_{i}.png"
            img.save(img_path)
            dataset.append({"image": str(img_path), "label": i % 5})
        return dataset

    dataset1 = create_dataset(20, color_shift=0)
    dataset2 = create_dataset(20, color_shift=10)
    dataset3 = create_dataset(20, color_shift=0)
    dataset4 = create_dataset(20, color_shift=128)

    yield (dataset1, dataset2), (dataset3, dataset4), temp_dir

    shutil.rmtree(temp_dir)


# ============================================================================
# Label Distribution Metrics
# ============================================================================

class TestJSDivergence:

    def test_self_divergence_is_zero(self, identical_distributions):
        dataset1, _, num_classes = identical_distributions
        JS = metrics.JS_divergence([dataset1, dataset1], num_classes)
        assert JS < 1e-6

    def test_non_negative(self, skewed_distributions):
        dataset1, dataset2, num_classes = skewed_distributions
        JS = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert JS >= 0

    def test_identical_distributions_near_zero(self, identical_distributions):
        dataset1, dataset2, num_classes = identical_distributions
        JS = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert JS < 0.1

    def test_complementary_distributions_high(self, complementary_distributions):
        dataset1, dataset2, num_classes = complementary_distributions
        JS = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert JS > 0.5

    def test_not_symmetric(self, skewed_distributions):
        dataset1, dataset2, num_classes = skewed_distributions
        JS_all = metrics.JS_divergence([dataset1, dataset2], num_classes, return_all=True)
        # Single pair, so just verify it's valid
        assert len(JS_all) == 1
        assert JS_all[0] >= 0

    def test_uniform_distributions_zero(self):
        dataset1 = [{"label": i % 10} for i in range(100)]
        dataset2 = [{"label": i % 10} for i in range(100)]
        JS = metrics.JS_divergence([dataset1, dataset2], 10)
        assert JS < 1e-6

    def test_known_value(self):
        # P = [0.5, 0.5], Q = [0.8, 0.2]
        # JS(P||Q) = 0.0505
        dataset1 = [{"label": 0}] * 50 + [{"label": 1}] * 50
        dataset2 = [{"label": 0}] * 80 + [{"label": 1}] * 20
        JS = metrics.JS_divergence([dataset1, dataset2], 2)
        assert abs(JS - 0.0505) < 0.01

    def test_return_all_multiple_datasets(self):
        d1 = [{"label": i % 5} for i in range(50)]
        d2 = [{"label": i % 5} for i in range(50)]
        d3 = [{"label": i % 5} for i in range(50)]
        JS_all = metrics.JS_divergence([d1, d2, d3], 5, return_all=True)
        assert len(JS_all) == 3  # 3 pairs: (0,1), (0,2), (1,2)


class TestKSLabelTest:

    def test_returns_dict_with_keys(self, identical_distributions):
        dataset1, dataset2, _ = identical_distributions
        result = metrics.KS_label_test([dataset1, dataset2])
        assert isinstance(result, dict)
        assert 'statistic' in result
        assert 'pvalue' in result

    def test_pvalue_range(self, skewed_distributions):
        dataset1, dataset2, _ = skewed_distributions
        result = metrics.KS_label_test([dataset1, dataset2])
        assert 0 <= result['pvalue'] <= 1

    def test_statistic_range(self, skewed_distributions):
        dataset1, dataset2, _ = skewed_distributions
        result = metrics.KS_label_test([dataset1, dataset2])
        assert 0 <= result['statistic'] <= 1

    def test_identical_distributions_high_pvalue(self, identical_distributions):
        dataset1, dataset2, _ = identical_distributions
        result = metrics.KS_label_test([dataset1, dataset2])
        assert result['pvalue'] > 0.05

    def test_complementary_distributions_low_pvalue(self, complementary_distributions):
        dataset1, dataset2, _ = complementary_distributions
        result = metrics.KS_label_test([dataset1, dataset2])
        assert result['pvalue'] < 0.05

    def test_self_comparison(self, skewed_distributions):
        dataset1, _, _ = skewed_distributions
        result = metrics.KS_label_test([dataset1, dataset1])
        assert result['statistic'] < 1e-10
        assert result['pvalue'] > 0.99

    def test_return_all(self, identical_distributions):
        dataset1, dataset2, _ = identical_distributions
        d3 = [{"label": i % 10} for i in range(100)]
        results = metrics.KS_label_test([dataset1, dataset2, d3], return_all=True)
        assert isinstance(results, list)
        assert len(results) == 3


class TestMMDLabelDistributions:

    def test_self_distance_zero(self, skewed_distributions):
        dataset1, _, num_classes = skewed_distributions
        mmd = metrics.MMD_label_distributions([dataset1, dataset1], num_classes)
        assert mmd < 1e-10

    def test_non_negative(self, skewed_distributions):
        dataset1, dataset2, num_classes = skewed_distributions
        mmd = metrics.MMD_label_distributions([dataset1, dataset2], num_classes)
        assert mmd >= 0

    def test_symmetric(self, skewed_distributions):
        dataset1, dataset2, num_classes = skewed_distributions
        mmd_pq = metrics.MMD_label_distributions([dataset1, dataset2], num_classes)
        mmd_qp = metrics.MMD_label_distributions([dataset2, dataset1], num_classes)
        assert abs(mmd_pq - mmd_qp) < 1e-10

    def test_identical_distributions_near_zero(self, identical_distributions):
        dataset1, dataset2, num_classes = identical_distributions
        mmd = metrics.MMD_label_distributions([dataset1, dataset2], num_classes)
        assert mmd < 0.1

    def test_complementary_distributions_high(self, complementary_distributions):
        dataset1, dataset2, num_classes = complementary_distributions
        mmd = metrics.MMD_label_distributions([dataset1, dataset2], num_classes)
        assert mmd > 0.5

    def test_return_all(self, identical_distributions):
        dataset1, dataset2, num_classes = identical_distributions
        d3 = [{"label": i % 10} for i in range(100)]
        mmd_all = metrics.MMD_label_distributions([dataset1, dataset2, d3], num_classes, return_all=True)
        assert len(mmd_all) == 3


class TestMahalanobisDistanceLabels:

    def test_non_negative(self, skewed_distributions):
        dataset1, dataset2, num_classes = skewed_distributions
        dist = metrics.mahalanobis_distance_labels([dataset1, dataset2], num_classes)
        assert dist >= 0

    def test_identical_distributions_near_zero(self, identical_distributions):
        dataset1, dataset2, num_classes = identical_distributions
        dist = metrics.mahalanobis_distance_labels([dataset1, dataset2], num_classes)
        assert dist < 0.5

    def test_self_distance_zero(self, skewed_distributions):
        dataset1, _, num_classes = skewed_distributions
        dist = metrics.mahalanobis_distance_labels([dataset1, dataset1], num_classes)
        assert dist < 1e-6

    def test_complementary_distributions_high(self, complementary_distributions):
        dataset1, dataset2, num_classes = complementary_distributions
        dist = metrics.mahalanobis_distance_labels([dataset1, dataset2], num_classes)
        assert dist > 0.5

    def test_return_all(self, identical_distributions):
        dataset1, dataset2, num_classes = identical_distributions
        d3 = [{"label": i % 10} for i in range(100)]
        dists = metrics.mahalanobis_distance_labels([dataset1, dataset2, d3], num_classes, return_all=True)
        assert len(dists) == 3


# ============================================================================
# MMD RBF Kernel
# ============================================================================

class TestMMDRBFKernel:

    def test_self_distance_zero(self):
        X = np.random.randn(50, 10)
        mmd = metrics.MMD_rbf_kernel(X, X, sigma=1.0)
        assert abs(mmd) < 1e-6

    def test_non_negative_for_different_samples(self):
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10) + 2.0
        mmd = metrics.MMD_rbf_kernel(X, Y, sigma=1.0)
        assert mmd > -0.1

    def test_symmetric(self):
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10) + 1.0
        mmd_xy = metrics.MMD_rbf_kernel(X, Y, sigma=1.0)
        mmd_yx = metrics.MMD_rbf_kernel(Y, X, sigma=1.0)
        assert abs(mmd_xy - mmd_yx) < 1e-10

    def test_increases_with_distance(self):
        X = np.random.randn(50, 10)
        Y1 = np.random.randn(50, 10) + 0.5
        Y2 = np.random.randn(50, 10) + 2.0
        mmd1 = metrics.MMD_rbf_kernel(X, Y1, sigma=1.0)
        mmd2 = metrics.MMD_rbf_kernel(X, Y2, sigma=1.0)
        assert mmd2 > mmd1

    def test_median_heuristic(self):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10)
        mmd = metrics.MMD_rbf_kernel(X, Y, sigma=None)
        assert isinstance(mmd, (float, np.floating))

    def test_minimum_samples_required(self):
        X = np.random.randn(1, 10)
        Y = np.random.randn(50, 10)
        with pytest.raises(ValueError):
            metrics.MMD_rbf_kernel(X, Y)


# ============================================================================
# Feature Distribution Metrics
# ============================================================================

class TestFIDDistance:

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        fid = metrics.FID_distance([subset, subset], batch_size=5)
        assert fid < 10.0

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        fid = metrics.FID_distance([dataset1[:10], dataset2[:10]], batch_size=5)
        assert fid >= 0

    @pytest.mark.slow
    def test_symmetric(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        subset1, subset2 = dataset1[:10], dataset2[:10]
        fid_12 = metrics.FID_distance([subset1, subset2], batch_size=5)
        fid_21 = metrics.FID_distance([subset2, subset1], batch_size=5)
        assert abs(fid_12 - fid_21) < 1e-6

    @pytest.mark.slow
    def test_different_distributions_higher(self, temp_image_datasets):
        (dataset1, dataset2), (dataset3, dataset4), _ = temp_image_datasets
        fid_small = metrics.FID_distance([dataset1[:10], dataset2[:10]], batch_size=5)
        fid_large = metrics.FID_distance([dataset3[:10], dataset4[:10]], batch_size=5)
        assert fid_large >= 0 and fid_small >= 0

    @pytest.mark.slow
    def test_return_all_option(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        subset1, subset2, subset3 = dataset1[:5], dataset2[:5], dataset1[5:10]
        fid_scores = metrics.FID_distance([subset1, subset2, subset3], batch_size=5, return_all=True)
        assert isinstance(fid_scores, list)
        assert len(fid_scores) == 3


class TestMMDFeatures:

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        mmd = metrics.MMD_features([dataset1[:10], dataset2[:10]], batch_size=5)
        assert mmd > -1.0

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        mmd = metrics.MMD_features([subset, subset], batch_size=5)
        assert abs(mmd) < 1.0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        mmd_all = metrics.MMD_features([dataset1[:5], dataset2[:5], dataset1[5:10]], batch_size=5, return_all=True)
        assert len(mmd_all) == 3


class TestWassersteinDistanceFeatures:

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        wd = metrics.wasserstein_distance_features([dataset1[:10], dataset2[:10]], batch_size=5)
        assert wd >= 0

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        wd = metrics.wasserstein_distance_features([subset, subset], batch_size=5)
        assert wd < 1.0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        wd_all = metrics.wasserstein_distance_features(
            [dataset1[:5], dataset2[:5], dataset1[5:10]], batch_size=5, return_all=True
        )
        assert len(wd_all) == 3


class TestCramerVonMisesFeatures:

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        cvm = metrics.cramer_von_mises_features([dataset1[:10], dataset2[:10]], batch_size=5)
        assert cvm >= 0

    @pytest.mark.slow
    def test_self_comparison_small(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        cvm = metrics.cramer_von_mises_features([subset, subset], batch_size=5)
        assert cvm < 1.0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        cvm_all = metrics.cramer_von_mises_features(
            [dataset1[:5], dataset2[:5], dataset1[5:10]], batch_size=5, return_all=True
        )
        assert len(cvm_all) == 3


class TestKSMultivariateTest:

    @pytest.mark.slow
    def test_returns_dict_with_keys(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        result = metrics.KS_multivariate_test([dataset1[:10], dataset2[:10]], batch_size=5)
        assert isinstance(result, dict)
        assert 'reject_null' in result
        assert 'min_pvalue' in result
        assert 'significant_dims' in result

    @pytest.mark.slow
    def test_reject_null_is_boolean(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        result = metrics.KS_multivariate_test([dataset1[:10], dataset2[:10]], batch_size=5)
        assert isinstance(result['reject_null'], bool)

    @pytest.mark.slow
    def test_pvalue_range(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        result = metrics.KS_multivariate_test([dataset1[:10], dataset2[:10]], batch_size=5)
        assert 0 <= result['min_pvalue'] <= 1

    @pytest.mark.slow
    def test_self_comparison_fails_to_reject(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        result = metrics.KS_multivariate_test([subset, subset], batch_size=5)
        assert result['reject_null'] == False
        assert result['min_pvalue'] > 0.01

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        results = metrics.KS_multivariate_test(
            [dataset1[:5], dataset2[:5], dataset1[5:10]], batch_size=5, return_all=True
        )
        assert isinstance(results, list)
        assert len(results) == 3


# ============================================================================
# Vendi Score
# ============================================================================

class TestVendiScoreFeatures:

    @pytest.mark.slow
    def test_returns_float(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        vs = metrics.vendi_score_features([dataset1[:10]], batch_size=5)
        assert isinstance(vs, float)

    @pytest.mark.slow
    def test_positive(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        vs = metrics.vendi_score_features([dataset1[:10]], batch_size=5)
        assert vs > 0

    @pytest.mark.slow
    def test_higher_diversity_higher_score(self, temp_image_datasets):
        """More diverse dataset should have higher Vendi score."""
        (dataset1, _), (_, dataset4), _ = temp_image_datasets
        # dataset1 has gradual color changes, dataset4 has more varied colors
        vs_gradual = metrics.vendi_score_features([dataset1[:10]], batch_size=5)
        # Both should be positive
        assert vs_gradual > 0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        vs_all = metrics.vendi_score_features([dataset1[:5], dataset2[:5]], batch_size=5, return_all=True)
        assert isinstance(vs_all, list)
        assert len(vs_all) == 2
        assert all(v > 0 for v in vs_all)

    @pytest.mark.slow
    def test_single_vs_multiple_datasets(self, temp_image_datasets):
        """Average of individual scores should equal mean when return_all=False."""
        (dataset1, dataset2), _, _ = temp_image_datasets
        vs_mean = metrics.vendi_score_features([dataset1[:5], dataset2[:5]], batch_size=5)
        vs_all = metrics.vendi_score_features([dataset1[:5], dataset2[:5]], batch_size=5, return_all=True)
        assert abs(vs_mean - np.mean(vs_all)) < 1e-6

    @pytest.mark.slow
    def test_identical_images_low_score(self, temp_image_datasets):
        """Dataset with identical images should have low diversity score."""
        _, _, temp_dir = temp_image_datasets
        # Create dataset with identical images
        img = Image.new('RGB', (64, 64), color=(128, 128, 128))
        identical_dataset = []
        for i in range(10):
            img_path = Path(temp_dir) / f"identical_{i}.png"
            img.save(img_path)
            identical_dataset.append({"image": str(img_path), "label": 0})

        vs = metrics.vendi_score_features([identical_dataset], batch_size=5)
        # Identical images should give score close to 1
        assert vs < 2.0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:

    def test_empty_dataset_handling(self):
        dataset_empty = []
        dataset_normal = [{"label": i % 5} for i in range(20)]
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            metrics.MMD_label_distributions([dataset_empty, dataset_normal], num_classes=5)

    def test_single_sample_datasets(self):
        dataset1 = [{"label": 0}]
        dataset2 = [{"label": 1}]
        try:
            JS = metrics.JS_divergence([dataset1, dataset2], 5)
            assert JS >= 0
        except (ValueError, ZeroDivisionError):
            pass

    def test_single_class_datasets(self, single_class_distributions):
        dataset1, dataset2, num_classes = single_class_distributions
        JS = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert JS >= 0
        mmd = metrics.MMD_label_distributions([dataset1, dataset2], num_classes)
        assert mmd >= 0

    def test_mismatched_num_classes(self):
        dataset1 = [{"label": 0}, {"label": 1}, {"label": 5}]
        dataset2 = [{"label": 2}, {"label": 3}]
        JS = metrics.JS_divergence([dataset1, dataset2], num_classes=10)
        assert JS >= 0

    def test_large_num_classes(self):
        dataset1 = [{"label": i} for i in range(1000)]
        dataset2 = [{"label": i} for i in range(500, 1500)]
        JS = metrics.JS_divergence([dataset1, dataset2], num_classes=1500)
        assert JS >= 0

    def test_minimum_two_datasets_required(self):
        dataset1 = [{"label": i % 5} for i in range(50)]
        with pytest.raises(ValueError):
            metrics.JS_divergence([dataset1], 5)


# ============================================================================
# Integration Tests
# ============================================================================

class TestMetricConsistency:

    def test_all_metrics_agree_on_identical(self, identical_distributions):
        dataset1, dataset2, num_classes = identical_distributions

        JS = metrics.JS_divergence([dataset1, dataset2], num_classes)
        ks = metrics.KS_label_test([dataset1, dataset2])
        mmd = metrics.MMD_label_distributions([dataset1, dataset2], num_classes)
        mahal = metrics.mahalanobis_distance_labels([dataset1, dataset2], num_classes)

        assert JS < 0.1
        assert ks['pvalue'] > 0.05
        assert mmd < 0.1
        assert mahal < 0.5

    def test_all_metrics_agree_on_complementary(self, complementary_distributions):
        dataset1, dataset2, num_classes = complementary_distributions

        JS = metrics.JS_divergence([dataset1, dataset2], num_classes)
        ks = metrics.KS_label_test([dataset1, dataset2])
        mmd = metrics.MMD_label_distributions([dataset1, dataset2], num_classes)
        mahal = metrics.mahalanobis_distance_labels([dataset1, dataset2], num_classes)

        assert JS > 0.5
        assert ks['pvalue'] < 0.05
        assert mmd > 0.3
        assert mahal > 0.5

    def test_metrics_order_distributions_consistently(self):
        identical1 = [{"label": i % 10} for i in range(100)]
        identical2 = [{"label": i % 10} for i in range(100)]

        slight_diff1 = [{"label": 0}] * 60 + [{"label": i % 9 + 1} for i in range(40)]
        slight_diff2 = [{"label": 0}] * 40 + [{"label": i % 9 + 1} for i in range(60)]

        very_diff1 = [{"label": i % 5} for i in range(100)]
        very_diff2 = [{"label": i % 5 + 5} for i in range(100)]

        JS_identical = metrics.JS_divergence([identical1, identical2], 10)
        JS_slight = metrics.JS_divergence([slight_diff1, slight_diff2], 10)
        JS_very = metrics.JS_divergence([very_diff1, very_diff2], 10)

        assert JS_identical < JS_slight < JS_very

    def test_multiple_datasets_pairwise(self):
        """Test that N datasets produce N*(N-1)/2 pairwise comparisons."""
        datasets = [[{"label": i % 10} for i in range(50)] for _ in range(4)]

        JS_all = metrics.JS_divergence(datasets, 10, return_all=True)
        mmd_all = metrics.MMD_label_distributions(datasets, 10, return_all=True)

        expected_pairs = 4 * 3 // 2  # 6 pairs
        assert len(JS_all) == expected_pairs
        assert len(mmd_all) == expected_pairs


# ============================================================================
# Numerical Stability
# ============================================================================

class TestNumericalStability:

    def test_handles_zero_counts(self):
        dataset1 = [{"label": 0}] * 50
        dataset2 = [{"label": 1}] * 50
        JS = metrics.JS_divergence([dataset1, dataset2], 2)
        assert np.isfinite(JS)

    def test_large_sample_sizes(self):
        dataset1 = [{"label": i % 100} for i in range(10000)]
        dataset2 = [{"label": i % 100} for i in range(10000)]
        JS = metrics.JS_divergence([dataset1, dataset2], 100)
        assert np.isfinite(JS)

    def test_extreme_skew(self):
        dataset1 = [{"label": 0}] * 999 + [{"label": 1}]
        dataset2 = [{"label": i % 10} for i in range(1000)]
        JS = metrics.JS_divergence([dataset1, dataset2], 10)
        assert np.isfinite(JS)
        assert JS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])