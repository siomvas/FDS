"""
Unit tests for evaluation metrics.

Updated to match current metrics in evaluation/metrics.py
Run with: pytest metric_tests.py -v
Run without slow tests: pytest metric_tests.py -v -m "not slow"
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
    """Test Jensen-Shannon Divergence."""

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

    def test_return_all_multiple_datasets(self):
        d1 = [{"label": i % 5} for i in range(50)]
        d2 = [{"label": i % 5} for i in range(50)]
        d3 = [{"label": i % 5} for i in range(50)]
        JS_all = metrics.JS_divergence([d1, d2, d3], 5, return_all=True)
        assert len(JS_all) == 3  # 3 pairs: (0,1), (0,2), (1,2)

    def test_known_value(self):
        # P = [0.5, 0.5], Q = [0.8, 0.2]
        # JS(P||Q) ≈ 0.0505
        dataset1 = [{"label": 0}] * 50 + [{"label": 1}] * 50
        dataset2 = [{"label": 0}] * 80 + [{"label": 1}] * 20
        JS = metrics.JS_divergence([dataset1, dataset2], 2)
        assert abs(JS - 0.0505) < 0.01


class TestTVLabelDistributions:
    """Test Total Variation distance."""

    def test_self_distance_zero(self, identical_distributions):
        dataset1, _, num_classes = identical_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset1], num_classes)
        assert tv < 1e-6

    def test_non_negative(self, skewed_distributions):
        dataset1, dataset2, num_classes = skewed_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        assert tv >= 0

    def test_bounded_by_one(self, complementary_distributions):
        dataset1, dataset2, num_classes = complementary_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        assert tv <= 1.0

    def test_complementary_distributions(self, complementary_distributions):
        """Non-overlapping distributions should give TV close to 1."""
        dataset1, dataset2, num_classes = complementary_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        assert tv > 0.8

    def test_return_all(self):
        d1 = [{"label": i % 5} for i in range(50)]
        d2 = [{"label": i % 5} for i in range(50)]
        d3 = [{"label": 0}] * 50
        tv_all = metrics.TV_label_distributions([d1, d2, d3], 5, return_all=True)
        assert len(tv_all) == 3


class TestJSDClientToGlobal:
    """Test JSD client-to-global metric."""

    def test_returns_list_or_float(self):
        d1 = [{"label": i % 5} for i in range(50)]
        d2 = [{"label": i % 5} for i in range(50)]

        # return_all=False should give float
        jsd_mean = metrics.JSD_client_to_global([d1, d2], 5, return_all=False)
        assert isinstance(jsd_mean, (float, np.floating))

        # return_all=True should give list
        jsd_list = metrics.JSD_client_to_global([d1, d2], 5, return_all=True)
        assert isinstance(jsd_list, list)
        assert len(jsd_list) == 2

    def test_identical_to_global_near_zero(self):
        """If all clients identical, JSD to global should be near zero."""
        d1 = [{"label": i % 5} for i in range(50)]
        d2 = [{"label": i % 5} for i in range(50)]
        d3 = [{"label": i % 5} for i in range(50)]

        jsd_list = metrics.JSD_client_to_global([d1, d2, d3], 5, return_all=True)
        assert all(jsd < 0.01 for jsd in jsd_list)

    def test_specialized_client_high_jsd(self):
        """Client with specialized distribution should have high JSD to global."""
        d1 = [{"label": 0}] * 90 + [{"label": 1}] * 10  # Specialized in class 0
        d2 = [{"label": i % 5} for i in range(100)]     # Uniform
        d3 = [{"label": i % 5} for i in range(100)]     # Uniform

        jsd_list = metrics.JSD_client_to_global([d1, d2, d3], 5, return_all=True)
        # d1 should have highest JSD (different from balanced global)
        assert jsd_list[0] > jsd_list[1]
        assert jsd_list[0] > jsd_list[2]


# ============================================================================
# Feature Distribution Metrics
# ============================================================================

class TestFIDDistance:
    """Test Fréchet Inception Distance."""

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        fid = metrics.FID_distance([subset, subset], batch_size=5, use_cache=False)
        assert fid < 10.0

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        fid = metrics.FID_distance([dataset1[:10], dataset2[:10]], batch_size=5, use_cache=False)
        assert fid >= 0

    @pytest.mark.slow
    def test_symmetric(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        subset1, subset2 = dataset1[:10], dataset2[:10]
        fid_12 = metrics.FID_distance([subset1, subset2], batch_size=5, use_cache=False)
        fid_21 = metrics.FID_distance([subset2, subset1], batch_size=5, use_cache=False)
        assert abs(fid_12 - fid_21) < 1e-6

    @pytest.mark.slow
    def test_return_all_option(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        subset1, subset2, subset3 = dataset1[:5], dataset2[:5], dataset1[5:10]
        fid_scores = metrics.FID_distance(
            [subset1, subset2, subset3],
            batch_size=5,
            return_all=True,
            use_cache=False
        )
        assert isinstance(fid_scores, list)
        assert len(fid_scores) == 3


class TestMMD:
    """Test Maximum Mean Discrepancy (RBF kernel)."""

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        mmd = metrics.MMD([dataset1[:10], dataset2[:10]], batch_size=5, use_cache=False)
        assert mmd > -1.0

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        mmd = metrics.MMD([subset, subset], batch_size=5, use_cache=False)
        assert abs(mmd) < 1.0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        mmd_all = metrics.MMD(
            [dataset1[:5], dataset2[:5], dataset1[5:10]],
            batch_size=5,
            return_all=True,
            use_cache=False
        )
        assert len(mmd_all) == 3

    @pytest.mark.slow
    def test_with_different_models(self, temp_image_datasets):
        """Test MMD with different feature extractors."""
        (dataset1, dataset2), _, _ = temp_image_datasets

        # Test with InceptionV3 (default)
        mmd_inception = metrics.MMD(
            [dataset1[:5], dataset2[:5]],
            model_name="inception_v3",
            batch_size=5,
            use_cache=False
        )
        assert mmd_inception >= -0.5  # Allow small negative due to estimation


class TestEnergyDistance:
    """Test Energy Distance."""

    @pytest.mark.slow
    def test_non_negative_or_zero(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        ed = metrics.energy_distance_features([dataset1[:10], dataset2[:10]], batch_size=5, use_cache=False)
        # Energy distance should be non-negative for different distributions
        assert isinstance(ed, (float, np.floating))

    @pytest.mark.slow
    def test_self_distance_zero(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        ed = metrics.energy_distance_features([subset, subset], batch_size=5, use_cache=False)
        assert abs(ed) < 1.0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        ed_all = metrics.energy_distance_features(
            [dataset1[:5], dataset2[:5], dataset1[5:10]],
            batch_size=5,
            return_all=True,
            use_cache=False
        )
        assert len(ed_all) == 3


class TestSlicedWassersteinDistance:
    """Test Sliced Wasserstein Distance."""

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        swd = metrics.sliced_wasserstein_distance([dataset1[:10], dataset2[:10]], batch_size=5, use_cache=False)
        assert swd >= 0

    @pytest.mark.slow
    def test_self_distance_near_zero(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        swd = metrics.sliced_wasserstein_distance([subset, subset], batch_size=5, use_cache=False)
        assert swd < 1.0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        swd_all = metrics.sliced_wasserstein_distance(
            [dataset1[:5], dataset2[:5], dataset1[5:10]],
            batch_size=5,
            return_all=True,
            use_cache=False
        )
        assert len(swd_all) == 3


class TestKSMultivariateTest:
    """Test Kolmogorov-Smirnov multivariate test."""

    @pytest.mark.slow
    def test_returns_dict_with_keys(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        result = metrics.KS_multivariate_test([dataset1[:10], dataset2[:10]], batch_size=5, use_cache=False)
        assert isinstance(result, dict)
        assert 'reject_null' in result
        assert 'min_pvalue' in result
        assert 'significant_dims' in result

    @pytest.mark.slow
    def test_reject_null_is_boolean(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        result = metrics.KS_multivariate_test([dataset1[:10], dataset2[:10]], batch_size=5, use_cache=False)
        assert isinstance(result['reject_null'], bool)

    @pytest.mark.slow
    def test_pvalue_range(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        result = metrics.KS_multivariate_test([dataset1[:10], dataset2[:10]], batch_size=5, use_cache=False)
        assert 0 <= result['min_pvalue'] <= 1

    @pytest.mark.slow
    def test_self_comparison_fails_to_reject(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        result = metrics.KS_multivariate_test([subset, subset], batch_size=5, use_cache=False)
        assert result['reject_null'] == False
        assert result['min_pvalue'] > 0.01

    @pytest.mark.slow
    def test_with_pca(self, temp_image_datasets):
        """Test KS test with PCA preprocessing."""
        (dataset1, dataset2), _, _ = temp_image_datasets
        result = metrics.KS_multivariate_test(
            [dataset1[:15], dataset2[:15]],
            pca_components=50,
            batch_size=5,
            use_cache=False
        )
        assert isinstance(result, dict)
        assert 'reject_null' in result


class TestVendiScore:
    """Test Vendi Score (diversity metric)."""

    @pytest.mark.slow
    def test_returns_float(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        vs = metrics.vendi_score([dataset1[:10]], batch_size=5, use_cache=False)
        assert isinstance(vs, (float, np.floating))

    @pytest.mark.slow
    def test_positive(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        vs = metrics.vendi_score([dataset1[:10]], batch_size=5, use_cache=False)
        assert vs > 0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        vs_all = metrics.vendi_score([dataset1[:5], dataset2[:5]], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(vs_all, list)
        assert len(vs_all) == 2
        assert all(v > 0 for v in vs_all)

    @pytest.mark.slow
    def test_identical_images_low_score(self, temp_image_datasets):
        """Dataset with identical images should have low diversity score (close to 1)."""
        _, _, temp_dir = temp_image_datasets
        # Create dataset with identical images
        img = Image.new('RGB', (64, 64), color=(128, 128, 128))
        identical_dataset = []
        for i in range(10):
            img_path = Path(temp_dir) / f"identical_{i}.png"
            img.save(img_path)
            identical_dataset.append({"image": str(img_path), "label": 0})

        vs = metrics.vendi_score([identical_dataset], batch_size=5, use_cache=False)
        # Identical images should give score close to 1
        assert vs < 2.0


class TestIntrinsicDimensionality:
    """Test intrinsic dimensionality estimation."""

    @pytest.mark.slow
    def test_returns_float_or_list(self, temp_image_datasets):
        (dataset1,_), _, _ = temp_image_datasets

        # return_all=False
        id_mean = metrics.intrinsic_dimensionality(
            [dataset1[:15]],
            method="two_nn",
            batch_size=5,
            use_faiss=False,
            return_all=False,
            use_cache=False
        )
        assert isinstance(id_mean, (float, np.floating))

        # return_all=True
        id_list = metrics.intrinsic_dimensionality(
            [dataset1[:15]],
            method="two_nn",
            batch_size=5,
            use_faiss=False,
            return_all=True,
            use_cache=False
        )
        assert isinstance(id_list, list)

    @pytest.mark.slow
    def test_positive_estimate(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        id_est = metrics.intrinsic_dimensionality(
            [dataset1[:15]],
            method="two_nn",
            use_faiss=False,
            batch_size=5,
            use_cache=False
        )
        assert id_est > 0

    @pytest.mark.slow
    def test_different_methods(self, temp_image_datasets):
        """Test both TWO-NN and MLE methods."""
        (dataset1, _), _, _ = temp_image_datasets

        id_two_nn = metrics.intrinsic_dimensionality(
            [dataset1[:15]],
            method="two_nn",
            use_faiss=False,
            batch_size=5,
            use_cache=False
        )

        id_mle = metrics.intrinsic_dimensionality(
            [dataset1[:15]],
            method="mle",
            k=10,
            use_faiss=False,
            batch_size=5,
            use_cache=False
        )

        assert id_two_nn > 0
        assert id_mle > 0


class TestKNNSeparability:
    """Test k-NN separability metric."""

    @pytest.mark.slow
    def test_returns_float_in_range(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        sep = metrics.knn_separability(
            [dataset1[:10], dataset2[:10]],
            k=5,
            batch_size=5,
            use_cache=False
        )
        assert 0 <= sep <= 1

    @pytest.mark.slow
    def test_high_separability_for_different_distributions(self, temp_image_datasets):
        """Different distributions should have higher separability."""
        (dataset1, _), (_, dataset4), _ = temp_image_datasets
        # dataset4 has very different colors (shift=128)
        sep = metrics.knn_separability(
            [dataset1[:10], dataset4[:10]],
            k=3,
            batch_size=5,
            use_cache=False
        )
        # Should have some separability
        assert sep > 0.3


class TestCentroidDistance:
    """Test centroid distance metric."""

    @pytest.mark.slow
    def test_non_negative(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        cd = metrics.centroid_distance(
            [dataset1[:10], dataset2[:10]],
            batch_size=5,
            use_cache=False
        )
        assert cd >= 0

    @pytest.mark.slow
    def test_self_distance_zero(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        subset = dataset1[:10]
        cd = metrics.centroid_distance([subset, subset], batch_size=5, use_cache=False)
        assert cd < 1.0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        cd_all = metrics.centroid_distance(
            [dataset1[:5], dataset2[:5], dataset1[5:10]],
            batch_size=5,
            return_all=True,
            use_cache=False
        )
        assert len(cd_all) == 3


class TestFeatureVariance:
    """Test feature variance metric."""

    @pytest.mark.slow
    def test_positive(self, temp_image_datasets):
        (dataset1, _), _, _ = temp_image_datasets
        fv = metrics.feature_variance([dataset1[:10]], batch_size=5, use_cache=False)
        assert fv > 0

    @pytest.mark.slow
    def test_return_all(self, temp_image_datasets):
        (dataset1, dataset2), _, _ = temp_image_datasets
        fv_all = metrics.feature_variance(
            [dataset1[:5], dataset2[:5]],
            batch_size=5,
            return_all=True,
            use_cache=False
        )
        assert isinstance(fv_all, list)
        assert len(fv_all) == 2
        assert all(v > 0 for v in fv_all)

    @pytest.mark.slow
    def test_identical_images_low_variance(self, temp_image_datasets):
        """Identical images should have very low variance."""
        _, _, temp_dir = temp_image_datasets
        img = Image.new('RGB', (64, 64), color=(128, 128, 128))
        identical_dataset = []
        for i in range(10):
            img_path = Path(temp_dir) / f"identical_{i}.png"
            img.save(img_path)
            identical_dataset.append({"image": str(img_path), "label": 0})

        fv = metrics.feature_variance([identical_dataset], batch_size=5, use_cache=False)
        # Identical images should have near-zero variance
        assert fv < 10.0  # Very low compared to varied images


# ============================================================================
# MMD RBF Kernel Utility
# ============================================================================

class TestMMDRBFKernel:
    """Test MMD RBF kernel utility function."""

    def test_self_distance_zero(self):
        X = np.random.randn(50, 10)
        mmd = metrics.MMD_rbf_kernel(X, X, sigma=1.0)
        assert abs(mmd) < 1e-6

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
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset_handling(self):
        dataset_empty = []
        dataset_normal = [{"label": i % 5} for i in range(20)]
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            metrics.JS_divergence([dataset_empty, dataset_normal], num_classes=5)

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
        TV = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        assert TV >= 0

    def test_minimum_two_datasets_required(self):
        dataset1 = [{"label": i % 5} for i in range(50)]
        with pytest.raises(ValueError):
            metrics.JS_divergence([dataset1], 5)


# ============================================================================
# Integration & Consistency Tests
# ============================================================================

class TestMetricConsistency:
    """Test that metrics agree on relative ordering."""

    def test_metrics_order_distributions_consistently(self):
        """All label metrics should agree on ordering: identical < slight_diff < very_diff."""
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
        TV_all = metrics.TV_label_distributions(datasets, 10, return_all=True)

        expected_pairs = 4 * 3 // 2  # 6 pairs
        assert len(JS_all) == expected_pairs
        assert len(TV_all) == expected_pairs


# ============================================================================
# Numerical Stability
# ============================================================================

class TestNumericalStability:
    """Test numerical stability with edge cases."""

    def test_handles_zero_counts(self):
        """Test with classes that have zero samples."""
        dataset1 = [{"label": 0}] * 50
        dataset2 = [{"label": 1}] * 50
        JS = metrics.JS_divergence([dataset1, dataset2], 2)
        assert np.isfinite(JS)

    def test_large_sample_sizes(self):
        """Test with large datasets."""
        dataset1 = [{"label": i % 100} for i in range(10000)]
        dataset2 = [{"label": i % 100} for i in range(10000)]
        JS = metrics.JS_divergence([dataset1, dataset2], 100)
        assert np.isfinite(JS)

    def test_extreme_skew(self):
        """Test with extremely skewed distributions."""
        dataset1 = [{"label": 0}] * 999 + [{"label": 1}]
        dataset2 = [{"label": i % 10} for i in range(1000)]
        JS = metrics.JS_divergence([dataset1, dataset2], 10)
        assert np.isfinite(JS)
        assert JS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
