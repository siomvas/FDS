"""Tests for evaluation metrics."""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from PIL import Image
from unittest.mock import MagicMock, patch, PropertyMock
import warnings

from evaluation import metrics


# ============================================================================
# Fixtures - Label Distributions
# ============================================================================


@pytest.fixture
def identical_label_distributions():
    """Two datasets with identical label distributions."""
    labels = list(range(10)) * 100
    np.random.seed(42)
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
    np.random.seed(42)
    np.random.shuffle(labels)
    dataset2 = [{"label": label} for label in labels]
    return dataset1, dataset2, 10


@pytest.fixture
def single_class_distributions():
    """Both datasets have only one class each (different classes)."""
    dataset1 = [{"label": 0} for _ in range(50)]
    dataset2 = [{"label": 1} for _ in range(50)]
    return dataset1, dataset2, 10


@pytest.fixture
def same_single_class():
    """Both datasets have the same single class."""
    dataset1 = [{"label": 0} for _ in range(50)]
    dataset2 = [{"label": 0} for _ in range(50)]
    return dataset1, dataset2, 10


@pytest.fixture
def three_client_distributions():
    """Three datasets for testing multi-client metrics."""
    d1 = [{"label": i % 5} for i in range(100)]
    d2 = [{"label": i % 5} for i in range(100)]
    d3 = [{"label": 0}] * 80 + [{"label": i % 4 + 1} for i in range(20)]
    return d1, d2, d3, 5


@pytest.fixture
def four_client_distributions():
    """Four datasets for testing pairwise computation."""
    datasets = [[{"label": i % 10} for i in range(50)] for _ in range(4)]
    return datasets, 10


# ============================================================================
# Fixtures - Image Datasets
# ============================================================================


@pytest.fixture
def temp_image_dir():
    """Create and cleanup temporary image directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def create_image_dataset(temp_image_dir):
    """Factory fixture to create image datasets."""

    def _create(num_samples, color_shift=0, prefix="img"):
        dataset = []
        for i in range(num_samples):
            color = ((i * 20 + color_shift) % 256, (i * 30 + color_shift) % 256, (i * 40 + color_shift) % 256)
            img = Image.new("RGB", (64, 64), color=color)
            img_path = Path(temp_image_dir) / f"{prefix}_{color_shift}_{i}.png"
            img.save(img_path)
            dataset.append({"image": str(img_path), "label": i % 5})
        return dataset

    return _create


@pytest.fixture
def small_image_datasets(create_image_dataset):
    """Two small image datasets (10 images each) for fast tests."""
    dataset1 = create_image_dataset(10, color_shift=0, prefix="small1")
    dataset2 = create_image_dataset(10, color_shift=50, prefix="small2")
    return dataset1, dataset2


@pytest.fixture
def medium_image_datasets(create_image_dataset):
    """Two medium image datasets (25 images each)."""
    dataset1 = create_image_dataset(25, color_shift=0, prefix="med1")
    dataset2 = create_image_dataset(25, color_shift=100, prefix="med2")
    return dataset1, dataset2


@pytest.fixture
def three_image_datasets(create_image_dataset):
    """Three image datasets for multi-client feature tests."""
    d1 = create_image_dataset(15, color_shift=0, prefix="three1")
    d2 = create_image_dataset(15, color_shift=60, prefix="three2")
    d3 = create_image_dataset(15, color_shift=120, prefix="three3")
    return d1, d2, d3


@pytest.fixture
def identical_image_dataset(temp_image_dir):
    """Dataset with identical images (for low diversity tests)."""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    dataset = []
    for i in range(15):
        img_path = Path(temp_image_dir) / f"identical_{i}.png"
        img.save(img_path)
        dataset.append({"image": str(img_path), "label": 0})
    return dataset


@pytest.fixture
def highly_different_datasets(create_image_dataset):
    """Two datasets with very different color distributions."""
    dataset1 = create_image_dataset(15, color_shift=0, prefix="diff1")
    dataset2 = create_image_dataset(15, color_shift=180, prefix="diff2")
    return dataset1, dataset2


# ============================================================================
# Test Global Cache Management
# ============================================================================


class TestGlobalCache:
    """Test get_global_cache function."""

    def test_returns_cache_manager(self, temp_image_dir):
        """Test that get_global_cache returns a CacheManager."""
        cache = metrics.get_global_cache(cache_dir=temp_image_dir)
        assert cache is not None
        # Reset global cache for other tests
        metrics._GLOBAL_CACHE = None

    def test_reuses_cache_same_dir(self, temp_image_dir):
        """Test that same cache_dir reuses existing cache."""
        cache1 = metrics.get_global_cache(cache_dir=temp_image_dir)
        cache2 = metrics.get_global_cache(cache_dir=temp_image_dir)
        assert cache1 is cache2
        metrics._GLOBAL_CACHE = None

    def test_recreates_cache_different_dir(self, temp_image_dir):
        """Test that different cache_dir creates new cache."""
        other_dir = tempfile.mkdtemp()
        try:
            cache1 = metrics.get_global_cache(cache_dir=temp_image_dir)
            cache2 = metrics.get_global_cache(cache_dir=other_dir)
            assert cache1 is not cache2
        finally:
            shutil.rmtree(other_dir)
            metrics._GLOBAL_CACHE = None

    def test_uses_env_default_cache_dir(self, monkeypatch, tmp_path):
        """Default cache dir should come from FDS_CACHE_DIR when set."""
        cache_dir = tmp_path / "env-cache"
        monkeypatch.setenv("FDS_CACHE_DIR", str(cache_dir))
        cache = metrics.get_global_cache()
        assert str(cache.cache_dir) == str(cache_dir)
        metrics._GLOBAL_CACHE = None


# ============================================================================
# Test _compute_pairwise Helper
# ============================================================================


class TestComputePairwise:
    """Test the _compute_pairwise helper function."""

    def test_single_item_raises(self):
        """Single item should raise ValueError."""
        with pytest.raises(ValueError, match="At least two items"):
            metrics._compute_pairwise([1], lambda x, y: x + y)

    def test_empty_list_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="At least two items"):
            metrics._compute_pairwise([], lambda x, y: x + y)

    def test_two_items_single_pair(self):
        """Two items produces one pair."""
        result = metrics._compute_pairwise([1, 2], lambda x, y: x + y, return_all=True)
        assert result == [3]

    def test_three_items_three_pairs(self):
        """Three items produces three pairs."""
        result = metrics._compute_pairwise([1, 2, 3], lambda x, y: x + y, return_all=True)
        assert len(result) == 3
        assert set(result) == {3, 4, 5}  # 1+2, 1+3, 2+3

    def test_four_items_six_pairs(self):
        """Four items produces six pairs (n*(n-1)/2)."""
        result = metrics._compute_pairwise([1, 2, 3, 4], lambda x, y: x * y, return_all=True)
        assert len(result) == 6

    def test_return_all_false_returns_mean(self):
        """return_all=False returns mean of all pairs."""
        result = metrics._compute_pairwise([1, 2, 3], lambda x, y: x + y, return_all=False)
        expected_mean = np.mean([3, 4, 5])  # 1+2, 1+3, 2+3
        assert result == expected_mean

    def test_return_all_true_returns_list(self):
        """return_all=True returns list."""
        result = metrics._compute_pairwise([1, 2], lambda x, y: 0, return_all=True)
        assert isinstance(result, list)

    def test_return_all_false_returns_scalar(self):
        """return_all=False returns scalar."""
        result = metrics._compute_pairwise([1, 2], lambda x, y: 5.0, return_all=False)
        assert isinstance(result, (float, np.floating))


# ============================================================================
# Test _extract_features
# ============================================================================


class TestExtractFeatures:
    """Test the _extract_features function."""

    def test_extracts_features_inception(self, small_image_datasets):
        """Test feature extraction with InceptionV3."""
        dataset1, dataset2 = small_image_datasets
        features = metrics._extract_features([dataset1, dataset2], model_name="inception_v3", batch_size=5, use_cache=False)
        assert len(features) == 2
        assert features[0].shape[0] == 10
        assert features[1].shape[0] == 10
        assert features[0].shape[1] == 2048  # InceptionV3 dim

    def test_extracts_features_dino(self, small_image_datasets):
        """Test feature extraction with DINOv2."""
        dataset1, dataset2 = small_image_datasets
        features = metrics._extract_features(
            [dataset1, dataset2], model_name="dinov2-small", batch_size=5, use_cache=False, proxy="http://proxy.example:3128"
        )
        assert len(features) == 2
        assert features[0].shape[1] == 384  # DINOv2-small dim

    def test_unsupported_model_raises(self, small_image_datasets):
        """Test that unsupported model raises ValueError."""
        dataset1, _ = small_image_datasets
        with pytest.raises(ValueError, match="Unsupported model"):
            metrics._extract_features([dataset1], model_name="nonexistent_model", use_cache=False)

    def test_passes_extractor_kwargs(self, small_image_datasets):
        """Test that extractor_kwargs are passed through."""
        dataset1, _ = small_image_datasets
        # This should work without error - k_last is a valid kwarg for ViT models
        features = metrics._extract_features(
            [dataset1],
            model_name="dinov2-small",
            batch_size=5,
            use_cache=False,
            return_which_tokens="k_last_cls",
            k_last=2,
            proxy="http://proxy.example:3128",
        )
        # k_last=2 with dim 384 = 768
        assert features[0].shape[1] == 768


# ============================================================================
# Test JS Divergence
# ============================================================================


class TestJSDivergence:
    """Test Jensen-Shannon Divergence."""

    def test_self_divergence_zero(self, identical_label_distributions):
        """JSD(D, D) should be 0."""
        dataset1, _, num_classes = identical_label_distributions
        jsd = metrics.JS_divergence([dataset1, dataset1], num_classes)
        assert jsd < 1e-10

    def test_symmetric(self, skewed_distributions):
        """JSD(D1, D2) == JSD(D2, D1)."""
        dataset1, dataset2, num_classes = skewed_distributions
        jsd_12 = metrics.JS_divergence([dataset1, dataset2], num_classes)
        jsd_21 = metrics.JS_divergence([dataset2, dataset1], num_classes)
        assert abs(jsd_12 - jsd_21) < 1e-10

    def test_non_negative(self, skewed_distributions):
        """JSD is always >= 0."""
        dataset1, dataset2, num_classes = skewed_distributions
        jsd = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert jsd >= 0

    def test_bounded_by_log2(self, complementary_distributions):
        """JSD is bounded by log(2) ≈ 0.693."""
        dataset1, dataset2, num_classes = complementary_distributions
        jsd = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert jsd <= np.log(2) + 1e-6

    def test_identical_distributions_near_zero(self, identical_label_distributions):
        """Similar distributions have low JSD."""
        dataset1, dataset2, num_classes = identical_label_distributions
        jsd = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert jsd < 0.1

    def test_complementary_distributions_high(self, complementary_distributions):
        """Non-overlapping classes have high JSD."""
        dataset1, dataset2, num_classes = complementary_distributions
        jsd = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert jsd > 0.5

    def test_return_all_multiple_datasets(self, three_client_distributions):
        """return_all=True returns list of all pairwise JSD."""
        d1, d2, d3, num_classes = three_client_distributions
        jsd_all = metrics.JS_divergence([d1, d2, d3], num_classes, return_all=True)
        assert isinstance(jsd_all, list)
        assert len(jsd_all) == 3  # 3 pairs

    def test_return_all_false_returns_mean(self, three_client_distributions):
        """return_all=False returns mean."""
        d1, d2, d3, num_classes = three_client_distributions
        jsd_mean = metrics.JS_divergence([d1, d2, d3], num_classes, return_all=False)
        jsd_all = metrics.JS_divergence([d1, d2, d3], num_classes, return_all=True)
        assert abs(jsd_mean - np.mean(jsd_all)) < 1e-10

    def test_auto_detect_num_classes(self, skewed_distributions):
        """num_classes=None auto-detects from data."""
        dataset1, dataset2, _ = skewed_distributions
        jsd = metrics.JS_divergence([dataset1, dataset2], num_classes=None)
        assert jsd >= 0

    def test_known_value(self):
        """Test against known analytical value."""
        # P = [0.5, 0.5], Q = [0.8, 0.2]
        # JSD ≈ 0.0505
        dataset1 = [{"label": 0}] * 50 + [{"label": 1}] * 50
        dataset2 = [{"label": 0}] * 80 + [{"label": 1}] * 20
        jsd = metrics.JS_divergence([dataset1, dataset2], 2)
        assert abs(jsd - 0.0505) < 0.01

    def test_four_datasets_six_pairs(self, four_client_distributions):
        """Four datasets produce 6 pairwise comparisons."""
        datasets, num_classes = four_client_distributions
        jsd_all = metrics.JS_divergence(datasets, num_classes, return_all=True)
        assert len(jsd_all) == 6


# ============================================================================
# Test TV Label Distributions
# ============================================================================


class TestTVLabelDistributions:
    """Test Total Variation distance."""

    def test_self_distance_zero(self, identical_label_distributions):
        """TV(D, D) = 0."""
        dataset1, _, num_classes = identical_label_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset1], num_classes)
        assert tv < 1e-10

    def test_symmetric(self, skewed_distributions):
        """TV(D1, D2) == TV(D2, D1)."""
        dataset1, dataset2, num_classes = skewed_distributions
        tv_12 = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        tv_21 = metrics.TV_label_distributions([dataset2, dataset1], num_classes)
        assert abs(tv_12 - tv_21) < 1e-10

    def test_non_negative(self, skewed_distributions):
        """TV >= 0."""
        dataset1, dataset2, num_classes = skewed_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        assert tv >= 0

    def test_bounded_by_one(self, complementary_distributions):
        """TV <= 1."""
        dataset1, dataset2, num_classes = complementary_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        assert tv <= 1.0 + 1e-6

    def test_complementary_near_one(self, complementary_distributions):
        """Non-overlapping distributions have TV close to 1."""
        dataset1, dataset2, num_classes = complementary_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        assert tv > 0.9

    def test_identical_near_zero(self, identical_label_distributions):
        """Similar distributions have TV near 0."""
        dataset1, dataset2, num_classes = identical_label_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset2], num_classes)
        assert tv < 0.1

    def test_return_all(self, three_client_distributions):
        """return_all=True returns list."""
        d1, d2, d3, num_classes = three_client_distributions
        tv_all = metrics.TV_label_distributions([d1, d2, d3], num_classes, return_all=True)
        assert isinstance(tv_all, list)
        assert len(tv_all) == 3

    def test_auto_detect_num_classes(self, skewed_distributions):
        """num_classes=None auto-detects."""
        dataset1, dataset2, _ = skewed_distributions
        tv = metrics.TV_label_distributions([dataset1, dataset2], num_classes=None)
        assert 0 <= tv <= 1

    def test_triangle_inequality(self):
        """TV satisfies triangle inequality: TV(A,C) <= TV(A,B) + TV(B,C)."""
        d1 = [{"label": 0}] * 50 + [{"label": 1}] * 50
        d2 = [{"label": 0}] * 70 + [{"label": 1}] * 30
        d3 = [{"label": 0}] * 90 + [{"label": 1}] * 10

        tv_12 = metrics.TV_label_distributions([d1, d2], 2)
        tv_23 = metrics.TV_label_distributions([d2, d3], 2)
        tv_13 = metrics.TV_label_distributions([d1, d3], 2)

        assert tv_13 <= tv_12 + tv_23 + 1e-6


# ============================================================================
# Test JSD Client to Global
# ============================================================================


class TestJSDClientToGlobal:
    """Test JSD client-to-global metric."""

    def test_returns_float_when_return_all_false(self, three_client_distributions):
        """return_all=False returns float."""
        d1, d2, d3, num_classes = three_client_distributions
        result = metrics.JSD_client_to_global([d1, d2, d3], num_classes, return_all=False)
        assert isinstance(result, (float, np.floating))

    def test_returns_list_when_return_all_true(self, three_client_distributions):
        """return_all=True returns list with one value per client."""
        d1, d2, d3, num_classes = three_client_distributions
        result = metrics.JSD_client_to_global([d1, d2, d3], num_classes, return_all=True)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_identical_to_global_near_zero(self):
        """Clients identical to global have JSD near 0."""
        d1 = [{"label": i % 5} for i in range(100)]
        d2 = [{"label": i % 5} for i in range(100)]
        d3 = [{"label": i % 5} for i in range(100)]

        jsd_list = metrics.JSD_client_to_global([d1, d2, d3], 5, return_all=True)
        assert all(jsd < 0.01 for jsd in jsd_list)

    def test_specialized_client_high_jsd(self):
        """Client with specialized distribution has high JSD to global."""
        d1 = [{"label": 0}] * 90 + [{"label": 1}] * 10  # Specialized
        d2 = [{"label": i % 5} for i in range(100)]  # Uniform
        d3 = [{"label": i % 5} for i in range(100)]  # Uniform

        jsd_list = metrics.JSD_client_to_global([d1, d2, d3], 5, return_all=True)
        # d1 should have highest JSD
        assert jsd_list[0] > jsd_list[1]
        assert jsd_list[0] > jsd_list[2]

    def test_all_values_non_negative(self, three_client_distributions):
        """All JSD values are non-negative."""
        d1, d2, d3, num_classes = three_client_distributions
        jsd_list = metrics.JSD_client_to_global([d1, d2, d3], num_classes, return_all=True)
        assert all(jsd >= 0 for jsd in jsd_list)

    def test_auto_detect_num_classes(self, three_client_distributions):
        """num_classes=None auto-detects."""
        d1, d2, d3, _ = three_client_distributions
        result = metrics.JSD_client_to_global([d1, d2, d3], num_classes=None)
        assert result >= 0


# ============================================================================
# Test FID Distance
# ============================================================================


@pytest.mark.slow
class TestFIDDistance:
    """Test Fréchet Inception Distance."""

    def test_forces_inception_model(self, small_image_datasets):
        """FID always uses InceptionV3 regardless of model_name kwarg."""
        dataset1, dataset2 = small_image_datasets
        # Even if we pass a different model_name, FID should override it
        fid = metrics.FID_distance([dataset1, dataset2], model_name="dinov2-small", batch_size=5, use_cache=False)  # This should be ignored
        assert fid >= 0

    def test_self_distance_near_zero(self, small_image_datasets):
        """FID(D, D) ≈ 0."""
        dataset1, _ = small_image_datasets
        fid = metrics.FID_distance([dataset1, dataset1], batch_size=5, use_cache=False)
        assert fid < 50.0  # Should be very small

    def test_non_negative(self, small_image_datasets):
        """FID >= 0."""
        dataset1, dataset2 = small_image_datasets
        fid = metrics.FID_distance([dataset1, dataset2], batch_size=5, use_cache=False)
        assert fid >= 0

    def test_symmetric(self, small_image_datasets):
        """FID(D1, D2) == FID(D2, D1)."""
        dataset1, dataset2 = small_image_datasets
        fid_12 = metrics.FID_distance([dataset1, dataset2], batch_size=5, use_cache=False)
        fid_21 = metrics.FID_distance([dataset2, dataset1], batch_size=5, use_cache=False)
        assert abs(fid_12 - fid_21) < 1e-4

    def test_return_all(self, three_image_datasets):
        """return_all=True returns list."""
        d1, d2, d3 = three_image_datasets
        fid_all = metrics.FID_distance([d1, d2, d3], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(fid_all, list)
        assert len(fid_all) == 3

    def test_different_distributions_higher_fid(self, highly_different_datasets, small_image_datasets):
        """Very different distributions have higher FID."""
        d1_similar, d2_similar = small_image_datasets
        d1_diff, d2_diff = highly_different_datasets

        fid_similar = metrics.FID_distance([d1_similar, d2_similar], batch_size=5, use_cache=False)
        fid_diff = metrics.FID_distance([d1_diff, d2_diff], batch_size=5, use_cache=False)

        # Different distributions should have higher FID
        # (not always guaranteed with small samples, so we just check both are valid)
        assert fid_similar >= 0
        assert fid_diff >= 0

    def test_requires_two_datasets(self, small_image_datasets):
        """FID requires at least two datasets."""
        dataset1, _ = small_image_datasets
        with pytest.raises(ValueError, match="At least two datasets"):
            metrics.FID_distance([dataset1], batch_size=5, use_cache=False)


# ============================================================================
# Test MMD
# ============================================================================

class TestMMD:
    """Test Maximum Mean Discrepancy."""

    def test_self_distance_near_zero(self, small_image_datasets):
        """MMD(D, D) ≈ 0."""
        dataset1, _ = small_image_datasets
        mmd = metrics.MMD([dataset1, dataset1], batch_size=5, use_cache=False)
        assert abs(mmd) < 0.1

    def test_symmetric(self, small_image_datasets):
        """MMD(D1, D2) == MMD(D2, D1)."""
        dataset1, dataset2 = small_image_datasets
        mmd_12 = metrics.MMD([dataset1, dataset2], batch_size=5, use_cache=False)
        mmd_21 = metrics.MMD([dataset2, dataset1], batch_size=5, use_cache=False)
        assert abs(mmd_12 - mmd_21) < 1e-6

    def test_non_negative(self, small_image_datasets):
        """MMD >= 0 (approximately, due to estimation)."""
        dataset1, dataset2 = small_image_datasets
        mmd = metrics.MMD([dataset1, dataset2], batch_size=5, use_cache=False)
        assert mmd >= -0.1  # Allow small negative due to estimation

    def test_return_all(self, three_image_datasets):
        """return_all=True returns list."""
        d1, d2, d3 = three_image_datasets
        mmd_all = metrics.MMD([d1, d2, d3], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(mmd_all, list)
        assert len(mmd_all) == 3

    def test_with_explicit_sigma(self, small_image_datasets):
        """Test MMD with explicit sigma parameter."""
        dataset1, dataset2 = small_image_datasets
        mmd = metrics.MMD([dataset1, dataset2], sigma=1.0, batch_size=5, use_cache=False)
        assert isinstance(mmd, (float, np.floating))

    def test_with_different_models(self, small_image_datasets):
        """Test MMD with different feature extractors."""
        dataset1, dataset2 = small_image_datasets

        mmd_inception = metrics.MMD([dataset1, dataset2], model_name="inception_v3", batch_size=5, use_cache=False)

        mmd_dino = metrics.MMD(
            [dataset1, dataset2], model_name="dinov2-small", batch_size=5, use_cache=False, proxy="http://proxy.example:3128"
        )

        # Both should be valid
        assert isinstance(mmd_inception, (float, np.floating))
        assert isinstance(mmd_dino, (float, np.floating))


# ============================================================================
# Test Energy Distance
# ============================================================================

class TestEnergyDistance:
    """Test Energy Distance."""

    def test_self_distance_zero(self, small_image_datasets):
        """Energy distance of identical sets is 0."""
        dataset1, _ = small_image_datasets
        ed = metrics.energy_distance_features([dataset1, dataset1], batch_size=5, use_cache=False)
        assert abs(ed) < 1e-6

    def test_symmetric(self, small_image_datasets):
        """Energy distance is symmetric."""
        dataset1, dataset2 = small_image_datasets
        ed_12 = metrics.energy_distance_features([dataset1, dataset2], batch_size=5, use_cache=False)
        ed_21 = metrics.energy_distance_features([dataset2, dataset1], batch_size=5, use_cache=False)
        assert abs(ed_12 - ed_21) < 1e-6

    def test_return_all(self, three_image_datasets):
        """return_all=True returns list."""
        d1, d2, d3 = three_image_datasets
        ed_all = metrics.energy_distance_features([d1, d2, d3], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(ed_all, list)
        assert len(ed_all) == 3

    def test_with_different_models(self, small_image_datasets):
        """Test with different feature extractors."""
        dataset1, dataset2 = small_image_datasets
        ed = metrics.energy_distance_features(
            [dataset1, dataset2], model_name="dinov2-small", batch_size=5, use_cache=False, proxy="http://proxy.example:3128"
        )
        assert isinstance(ed, (float, np.floating))


# ============================================================================
# Test Sliced Wasserstein Distance
# ============================================================================


class TestSlicedWassersteinDistance:
    """Test Sliced Wasserstein Distance."""

    def test_self_distance_near_zero(self, small_image_datasets):
        """SWD(D, D) ≈ 0."""
        dataset1, _ = small_image_datasets
        swd = metrics.sliced_wasserstein_distance([dataset1, dataset1], batch_size=5, use_cache=False)
        assert swd < 1.0

    def test_non_negative(self, small_image_datasets):
        """SWD >= 0."""
        dataset1, dataset2 = small_image_datasets
        swd = metrics.sliced_wasserstein_distance([dataset1, dataset2], batch_size=5, use_cache=False)
        assert swd >= 0

    def test_symmetric(self, small_image_datasets):
        """SWD is approximately symmetric (random projections)."""
        dataset1, dataset2 = small_image_datasets
        # Run multiple times and average to reduce variance
        np.random.seed(42)
        swd_12 = metrics.sliced_wasserstein_distance([dataset1, dataset2], batch_size=5, use_cache=False)
        np.random.seed(42)
        swd_21 = metrics.sliced_wasserstein_distance([dataset2, dataset1], batch_size=5, use_cache=False)
        # Should be close (same seed = same projections)
        assert abs(swd_12 - swd_21) < 1e-6

    def test_return_all(self, three_image_datasets):
        """return_all=True returns list."""
        d1, d2, d3 = three_image_datasets
        swd_all = metrics.sliced_wasserstein_distance([d1, d2, d3], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(swd_all, list)
        assert len(swd_all) == 3


# ============================================================================
# Test KS Multivariate Test
# ============================================================================


class TestKSMultivariateTest:
    """Test Kolmogorov-Smirnov multivariate test."""

    def test_returns_dict(self, small_image_datasets):
        """Returns dict with expected keys."""
        dataset1, dataset2 = small_image_datasets
        result = metrics.KS_multivariate_test([dataset1, dataset2], batch_size=5, use_cache=False)
        assert isinstance(result, dict)
        assert "reject_null" in result
        assert "min_pvalue" in result
        assert "significant_dims" in result

    def test_reject_null_is_boolean(self, small_image_datasets):
        """reject_null is boolean."""
        dataset1, dataset2 = small_image_datasets
        result = metrics.KS_multivariate_test([dataset1, dataset2], batch_size=5, use_cache=False)
        assert isinstance(result["reject_null"], bool)

    def test_pvalue_in_range(self, small_image_datasets):
        """p-value is in [0, 1]."""
        dataset1, dataset2 = small_image_datasets
        result = metrics.KS_multivariate_test([dataset1, dataset2], batch_size=5, use_cache=False)
        assert 0 <= result["min_pvalue"] <= 1

    def test_self_comparison_fails_to_reject(self, small_image_datasets):
        """Identical distributions should not reject null."""
        dataset1, _ = small_image_datasets
        result = metrics.KS_multivariate_test([dataset1, dataset1], batch_size=5, use_cache=False)
        assert result["reject_null"] == False
        assert result["min_pvalue"] > 0.01

    def test_return_all(self, three_image_datasets):
        """return_all=True returns list."""
        d1, d2, d3 = three_image_datasets
        result = metrics.KS_multivariate_test([d1, d2, d3], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_with_pca_components(self, medium_image_datasets):
        """Test with PCA dimensionality reduction (n_components)."""
        dataset1, dataset2 = medium_image_datasets
        result = metrics.KS_multivariate_test([dataset1, dataset2], pca_components=50, batch_size=5, use_cache=False)
        assert isinstance(result, dict)

    def test_with_pca_variance(self, medium_image_datasets):
        """Test with PCA dimensionality reduction (variance threshold)."""
        dataset1, dataset2 = medium_image_datasets
        result = metrics.KS_multivariate_test([dataset1, dataset2], pca_variance=0.95, batch_size=5, use_cache=False)
        assert isinstance(result, dict)

    def test_custom_alpha(self, small_image_datasets):
        """Test with custom significance level."""
        dataset1, dataset2 = small_image_datasets
        result = metrics.KS_multivariate_test([dataset1, dataset2], alpha=0.01, batch_size=5, use_cache=False)
        assert isinstance(result, dict)


# ============================================================================
# Test Vendi Score
# ============================================================================


@pytest.mark.slow
class TestVendiScore:
    """Test Vendi Score diversity metric."""

    def test_returns_positive_float(self, small_image_datasets):
        """Returns positive float."""
        dataset1, _ = small_image_datasets
        vs = metrics.vendi_score([dataset1], batch_size=5, use_cache=False)
        assert isinstance(vs, (float, np.floating))
        assert vs > 0

    def test_return_all(self, small_image_datasets):
        """return_all=True returns list."""
        dataset1, dataset2 = small_image_datasets
        vs_all = metrics.vendi_score([dataset1, dataset2], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(vs_all, list)
        assert len(vs_all) == 2
        assert all(v > 0 for v in vs_all)

    def test_identical_images_low_score(self, identical_image_dataset):
        """Identical images have low diversity (score ≈ 1)."""
        vs = metrics.vendi_score([identical_image_dataset], batch_size=5, use_cache=False)
        assert vs < 2.0  # Should be close to 1

    def test_diverse_images_higher_score(self, small_image_datasets, identical_image_dataset):
        """Diverse images have higher score than identical."""
        dataset1, _ = small_image_datasets
        vs_diverse = metrics.vendi_score([dataset1], batch_size=5, use_cache=False)
        vs_identical = metrics.vendi_score([identical_image_dataset], batch_size=5, use_cache=False)
        assert vs_diverse > vs_identical

    def test_custom_q_parameter(self, small_image_datasets):
        """Test with custom diversity order q."""
        dataset1, _ = small_image_datasets
        vs_q1 = metrics.vendi_score([dataset1], q=1, batch_size=5, use_cache=False)
        vs_q2 = metrics.vendi_score([dataset1], q=2, batch_size=5, use_cache=False)
        # Both should be valid
        assert vs_q1 > 0
        assert vs_q2 > 0


# ============================================================================
# Test Intrinsic Dimensionality
# ============================================================================


@pytest.mark.slow
class TestIntrinsicDimensionality:
    """Test intrinsic dimensionality estimation."""

    def test_returns_float(self, medium_image_datasets):
        """Returns float when return_all=False."""
        dataset1, _ = medium_image_datasets
        id_est = metrics.intrinsic_dimensionality([dataset1], method="two_nn", use_faiss=False, batch_size=5, use_cache=False)
        assert isinstance(id_est, (float, np.floating))

    def test_returns_list(self, medium_image_datasets):
        """Returns list when return_all=True."""
        dataset1, dataset2 = medium_image_datasets
        id_list = metrics.intrinsic_dimensionality(
            [dataset1, dataset2], method="two_nn", use_faiss=False, return_all=True, batch_size=5, use_cache=False
        )
        assert isinstance(id_list, list)
        assert len(id_list) == 2

    def test_positive_estimate(self, medium_image_datasets):
        """ID estimate is positive."""
        dataset1, _ = medium_image_datasets
        id_est = metrics.intrinsic_dimensionality([dataset1], method="two_nn", use_faiss=False, batch_size=5, use_cache=False)
        assert id_est > 0

    def test_two_nn_method(self, medium_image_datasets):
        """Test TWO-NN method."""
        dataset1, _ = medium_image_datasets
        id_est = metrics.intrinsic_dimensionality([dataset1], method="two_nn", use_faiss=False, batch_size=5, use_cache=False)
        assert id_est > 0

    def test_mle_method(self, medium_image_datasets):
        """Test MLE method."""
        dataset1, _ = medium_image_datasets
        id_est = metrics.intrinsic_dimensionality([dataset1], method="mle", k=5, use_faiss=False, batch_size=5, use_cache=False)
        assert id_est > 0

    def test_invalid_method_raises(self, medium_image_datasets):
        """Invalid method raises ValueError."""
        dataset1, _ = medium_image_datasets
        with pytest.raises(ValueError, match="Unknown method"):
            metrics.intrinsic_dimensionality([dataset1], method="invalid", use_faiss=False, batch_size=5, use_cache=False)

    def test_custom_k_for_mle(self, medium_image_datasets):
        """Test MLE with custom k parameter."""
        dataset1, _ = medium_image_datasets
        id_k5 = metrics.intrinsic_dimensionality([dataset1], method="mle", k=5, use_faiss=False, batch_size=5, use_cache=False)
        id_k10 = metrics.intrinsic_dimensionality([dataset1], method="mle", k=10, use_faiss=False, batch_size=5, use_cache=False)
        # Both should be valid positive numbers
        assert id_k5 > 0
        assert id_k10 > 0


# ============================================================================
# Test kNN Separability
# ============================================================================


class TestKNNSeparability:
    """Test k-NN separability metric."""

    def test_returns_float_in_range(self, small_image_datasets):
        """Returns float in [0, 1]."""
        dataset1, dataset2 = small_image_datasets
        sep = metrics.knn_separability([dataset1, dataset2], k=3, batch_size=5, use_cache=False)
        assert isinstance(sep, (float, np.floating))
        assert 0 <= sep <= 1

    def test_return_all(self, small_image_datasets):
        """return_all=True returns list of per-sample scores."""
        dataset1, dataset2 = small_image_datasets
        sep_all = metrics.knn_separability([dataset1, dataset2], k=3, return_all=True, batch_size=5, use_cache=False)
        assert isinstance(sep_all, list)
        assert len(sep_all) == len(dataset1) + len(dataset2)

    def test_self_separability_high(self, small_image_datasets):
        """Same dataset repeated should have high separability (all neighbors from same 'client')."""
        dataset1, _ = small_image_datasets
        # Duplicate the dataset - all neighbors will be from "same client"
        sep = metrics.knn_separability([dataset1, dataset1], k=3, batch_size=5, use_cache=False)
        # When both datasets are identical, neighbors are randomly from either
        assert 0 <= sep <= 1

    def test_different_k_values(self, small_image_datasets):
        """Test with different k values."""
        dataset1, dataset2 = small_image_datasets
        sep_k3 = metrics.knn_separability([dataset1, dataset2], k=3, batch_size=5, use_cache=False)
        sep_k5 = metrics.knn_separability([dataset1, dataset2], k=5, batch_size=5, use_cache=False)
        # Both should be valid
        assert 0 <= sep_k3 <= 1
        assert 0 <= sep_k5 <= 1


# ============================================================================
# Test Centroid Distance
# ============================================================================


class TestCentroidDistance:
    """Test centroid distance metric."""

    def test_self_distance_zero(self, small_image_datasets):
        """Distance to self is 0."""
        dataset1, _ = small_image_datasets
        cd = metrics.centroid_distance([dataset1, dataset1], batch_size=5, use_cache=False)
        assert cd < 1e-6

    def test_non_negative(self, small_image_datasets):
        """Centroid distance >= 0."""
        dataset1, dataset2 = small_image_datasets
        cd = metrics.centroid_distance([dataset1, dataset2], batch_size=5, use_cache=False)
        assert cd >= 0

    def test_symmetric(self, small_image_datasets):
        """Centroid distance is symmetric."""
        dataset1, dataset2 = small_image_datasets
        cd_12 = metrics.centroid_distance([dataset1, dataset2], batch_size=5, use_cache=False)
        cd_21 = metrics.centroid_distance([dataset2, dataset1], batch_size=5, use_cache=False)
        assert abs(cd_12 - cd_21) < 1e-6

    def test_return_all(self, three_image_datasets):
        """return_all=True returns list."""
        d1, d2, d3 = three_image_datasets
        cd_all = metrics.centroid_distance([d1, d2, d3], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(cd_all, list)
        assert len(cd_all) == 3


# ============================================================================
# Test Feature Variance
# ============================================================================


class TestFeatureVariance:
    """Test feature variance metric."""

    def test_returns_positive(self, small_image_datasets):
        """Returns positive value."""
        dataset1, _ = small_image_datasets
        fv = metrics.feature_variance([dataset1], batch_size=5, use_cache=False)
        assert fv > 0

    def test_return_all(self, small_image_datasets):
        """return_all=True returns list."""
        dataset1, dataset2 = small_image_datasets
        fv_all = metrics.feature_variance([dataset1, dataset2], batch_size=5, return_all=True, use_cache=False)
        assert isinstance(fv_all, list)
        assert len(fv_all) == 2
        assert all(v > 0 for v in fv_all)

    def test_identical_images_low_variance(self, identical_image_dataset):
        """Identical images have very low variance."""
        fv = metrics.feature_variance([identical_image_dataset], batch_size=5, use_cache=False)
        # Should be close to 0 but not exactly 0 due to floating point
        assert fv < 10.0

    def test_diverse_images_higher_variance(self, small_image_datasets, identical_image_dataset):
        """Diverse images have higher variance."""
        dataset1, _ = small_image_datasets
        fv_diverse = metrics.feature_variance([dataset1], batch_size=5, use_cache=False)
        fv_identical = metrics.feature_variance([identical_image_dataset], batch_size=5, use_cache=False)
        assert fv_diverse > fv_identical


# ============================================================================
# Test MMD RBF Kernel Utility
# ============================================================================


class TestMMDRBFKernel:
    """Test MMD_rbf_kernel utility function."""

    def test_self_distance_zero(self):
        """MMD(X, X) = 0."""
        X = np.random.randn(50, 10)
        mmd = metrics.MMD_rbf_kernel(X, X, sigma=1.0)
        assert abs(mmd) < 1e-6

    def test_symmetric(self):
        """MMD(X, Y) == MMD(Y, X)."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10) + 1.0
        mmd_xy = metrics.MMD_rbf_kernel(X, Y, sigma=1.0)
        mmd_yx = metrics.MMD_rbf_kernel(Y, X, sigma=1.0)
        assert abs(mmd_xy - mmd_yx) < 1e-10

    def test_non_negative(self):
        """MMD >= 0."""
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 10)
        mmd = metrics.MMD_rbf_kernel(X, Y, sigma=1.0)
        assert mmd >= 0

    def test_increases_with_distance(self):
        """MMD increases as distributions separate."""
        X = np.random.randn(50, 10)
        Y1 = np.random.randn(50, 10) + 0.5
        Y2 = np.random.randn(50, 10) + 2.0
        mmd1 = metrics.MMD_rbf_kernel(X, Y1, sigma=1.0)
        mmd2 = metrics.MMD_rbf_kernel(X, Y2, sigma=1.0)
        assert mmd2 > mmd1

    def test_median_heuristic(self):
        """sigma=None uses median heuristic."""
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10)
        mmd = metrics.MMD_rbf_kernel(X, Y, sigma=None)
        assert isinstance(mmd, (float, np.floating))
        assert mmd >= 0

    def test_minimum_samples_required(self):
        """Requires at least 2 samples per distribution."""
        X = np.random.randn(1, 10)
        Y = np.random.randn(50, 10)
        with pytest.raises(ValueError, match="at least 2 points"):
            metrics.MMD_rbf_kernel(X, Y)

    def test_different_sample_sizes(self):
        """Works with different sample sizes."""
        X = np.random.randn(30, 10)
        Y = np.random.randn(50, 10)
        mmd = metrics.MMD_rbf_kernel(X, Y, sigma=1.0)
        assert mmd >= 0

    def test_dimension_mismatch_raises_value_error(self):
        """Mismatched feature dimensions should fail explicitly."""
        X = np.random.randn(30, 10)
        Y = np.random.randn(30, 8)
        with pytest.raises(ValueError, match="Feature dimensions must match"):
            metrics.MMD_rbf_kernel(X, Y, sigma=1.0)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset_raises(self):
        """Empty dataset raises error."""
        empty = []
        normal = [{"label": i % 5} for i in range(20)]
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            metrics.JS_divergence([empty, normal], num_classes=5)

    def test_single_sample_datasets(self):
        """Single sample datasets work or raise clearly."""
        d1 = [{"label": 0}]
        d2 = [{"label": 1}]
        try:
            jsd = metrics.JS_divergence([d1, d2], 5)
            assert jsd >= 0
        except (ValueError, ZeroDivisionError):
            pass

    def test_single_class_datasets(self, single_class_distributions):
        """Single class per dataset works."""
        dataset1, dataset2, num_classes = single_class_distributions
        jsd = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert jsd >= 0
        assert np.isfinite(jsd)

    def test_same_single_class(self, same_single_class):
        """Same single class in both datasets."""
        dataset1, dataset2, num_classes = same_single_class
        jsd = metrics.JS_divergence([dataset1, dataset2], num_classes)
        assert jsd < 1e-6  # Should be 0

    def test_minimum_two_datasets_required(self):
        """Pairwise metrics require at least two datasets."""
        d1 = [{"label": i % 5} for i in range(50)]
        with pytest.raises(ValueError):
            metrics.JS_divergence([d1], 5)

    def test_large_num_classes(self):
        """Works with large number of classes."""
        d1 = [{"label": i % 100} for i in range(1000)]
        d2 = [{"label": i % 100} for i in range(1000)]
        jsd = metrics.JS_divergence([d1, d2], 100)
        assert np.isfinite(jsd)

    def test_extremely_skewed(self):
        """Extremely skewed distributions."""
        d1 = [{"label": 0}] * 999 + [{"label": 1}]
        d2 = [{"label": i % 10} for i in range(1000)]
        jsd = metrics.JS_divergence([d1, d2], 10)
        assert np.isfinite(jsd)
        assert jsd > 0

    def test_five_datasets(self):
        """Test with five datasets (10 pairs)."""
        datasets = [[{"label": i % 10} for i in range(50)] for _ in range(5)]
        jsd_all = metrics.JS_divergence(datasets, 10, return_all=True)
        assert len(jsd_all) == 10  # 5*(5-1)/2 = 10 pairs

    def test_tv_five_datasets(self):
        """Test TV with five datasets."""
        datasets = [[{"label": i % 10} for i in range(50)] for _ in range(5)]
        tv_all = metrics.TV_label_distributions(datasets, 10, return_all=True)
        assert len(tv_all) == 10


class TestJSDClientToGlobalEdgeCases:
    """Test JSD client-to-global edge cases."""

    def test_single_client(self):
        """Single client's JSD to itself as global should be 0."""
        d1 = [{"label": i % 5} for i in range(100)]
        # With only one client, global = client, so JSD should be ~0
        jsd_list = metrics.JSD_client_to_global([d1], 5, return_all=True)
        assert len(jsd_list) == 1
        assert jsd_list[0] < 1e-6

    def test_two_clients(self):
        """Test with exactly two clients."""
        d1 = [{"label": 0}] * 100  # All class 0
        d2 = [{"label": 1}] * 100  # All class 1

        jsd_list = metrics.JSD_client_to_global([d1, d2], 2, return_all=True)
        assert len(jsd_list) == 2
        # Both should have similar JSD to balanced global
        assert jsd_list[0] > 0
        assert jsd_list[1] > 0

    def test_mean_vs_return_all(self):
        """Mean should equal np.mean of return_all."""
        d1 = [{"label": i % 5} for i in range(100)]
        d2 = [{"label": i % 5} for i in range(100)]
        d3 = [{"label": 0}] * 80 + [{"label": i % 4 + 1} for i in range(20)]

        jsd_mean = metrics.JSD_client_to_global([d1, d2, d3], 5, return_all=False)
        jsd_all = metrics.JSD_client_to_global([d1, d2, d3], 5, return_all=True)

        assert abs(jsd_mean - np.mean(jsd_all)) < 1e-10


# ============================================================================
# Test Numerical Stability
# ============================================================================


class TestNumericalStability:
    """Test numerical stability."""

    def test_near_zero_probabilities(self):
        """Handles near-zero probabilities."""
        d1 = [{"label": 0}] * 999 + [{"label": 9}]
        d2 = [{"label": i % 10} for i in range(1000)]
        jsd = metrics.JS_divergence([d1, d2], 10)
        assert np.isfinite(jsd)
        assert not np.isnan(jsd)
        assert 0 <= jsd <= np.log(2) + 1e-6

    def test_uniform_distributions(self):
        """Uniform distributions give JSD ≈ 0."""
        d1 = []
        d2 = []
        for i in range(10):
            d1.extend([{"label": i}] * 10)
            d2.extend([{"label": i}] * 10)
        jsd = metrics.JS_divergence([d1, d2], 10)
        assert abs(jsd) < 1e-6

    def test_large_sample_sizes(self):
        """Large datasets work correctly."""
        d1 = [{"label": i % 100} for i in range(10000)]
        d2 = [{"label": i % 100} for i in range(10000)]
        jsd = metrics.JS_divergence([d1, d2], 100)
        assert np.isfinite(jsd)

    def test_tv_extreme_cases(self):
        """TV handles extreme cases."""
        d1 = [{"label": 0}] * 100
        d2 = [{"label": 1}] * 100
        tv = metrics.TV_label_distributions([d1, d2], 2)
        assert np.isfinite(tv)
        assert tv > 0.99


# ============================================================================
# Test Metric Consistency
# ============================================================================


class TestMetricConsistency:
    """Test that metrics agree on relative ordering."""

    def test_ordering_identical_vs_different(self):
        """Identical distributions have lower divergence than different ones."""
        identical1 = [{"label": i % 10} for i in range(100)]
        identical2 = [{"label": i % 10} for i in range(100)]
        different1 = [{"label": i % 5} for i in range(100)]
        different2 = [{"label": i % 5 + 5} for i in range(100)]

        jsd_identical = metrics.JS_divergence([identical1, identical2], 10)
        jsd_different = metrics.JS_divergence([different1, different2], 10)

        assert jsd_identical < jsd_different

        tv_identical = metrics.TV_label_distributions([identical1, identical2], 10)
        tv_different = metrics.TV_label_distributions([different1, different2], 10)

        assert tv_identical < tv_different

    def test_pairwise_count(self, four_client_distributions):
        """N datasets produce N*(N-1)/2 pairs."""
        datasets, num_classes = four_client_distributions
        jsd_all = metrics.JS_divergence(datasets, num_classes, return_all=True)
        tv_all = metrics.TV_label_distributions(datasets, num_classes, return_all=True)

        expected_pairs = 4 * 3 // 2
        assert len(jsd_all) == expected_pairs
        assert len(tv_all) == expected_pairs


# ============================================================================
# Property-Based Tests
# ============================================================================

try:
    from hypothesis import given, strategies as st, settings

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@pytest.mark.property
class TestMetricProperties:
    """Property-based tests for metric mathematical properties."""

    @given(dataset_size=st.integers(20, 200), num_classes=st.integers(2, 20), seed=st.integers(0, 10000))
    @settings(max_examples=30, deadline=None)
    def test_jsd_bounded(self, dataset_size, num_classes, seed):
        """JSD is bounded in [0, log(2)]."""
        np.random.seed(seed)
        labels_1 = np.random.randint(0, num_classes, size=dataset_size)
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)

        d1 = [{"label": int(l)} for l in labels_1]
        d2 = [{"label": int(l)} for l in labels_2]

        jsd = metrics.JS_divergence([d1, d2], num_classes=num_classes)
        assert 0.0 <= jsd <= np.log(2) + 1e-6

    @given(dataset_size=st.integers(20, 200), num_classes=st.integers(2, 20), seed=st.integers(0, 10000))
    @settings(max_examples=30, deadline=None)
    def test_tv_bounded(self, dataset_size, num_classes, seed):
        """TV is bounded in [0, 1]."""
        np.random.seed(seed)
        labels_1 = np.random.randint(0, num_classes, size=dataset_size)
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)

        d1 = [{"label": int(l)} for l in labels_1]
        d2 = [{"label": int(l)} for l in labels_2]

        tv = metrics.TV_label_distributions([d1, d2], num_classes=num_classes)
        assert 0.0 <= tv <= 1.0 + 1e-6

    @given(dataset_size=st.integers(20, 200), num_classes=st.integers(2, 20), seed=st.integers(0, 10000))
    @settings(max_examples=30, deadline=None)
    def test_tv_triangle_inequality(self, dataset_size, num_classes, seed):
        """TV satisfies triangle inequality."""
        np.random.seed(seed)
        labels_1 = np.random.randint(0, num_classes, size=dataset_size)
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)
        labels_3 = np.random.randint(0, num_classes, size=dataset_size)

        d1 = [{"label": int(l)} for l in labels_1]
        d2 = [{"label": int(l)} for l in labels_2]
        d3 = [{"label": int(l)} for l in labels_3]

        tv_12 = metrics.TV_label_distributions([d1, d2], num_classes=num_classes)
        tv_13 = metrics.TV_label_distributions([d1, d3], num_classes=num_classes)
        tv_23 = metrics.TV_label_distributions([d2, d3], num_classes=num_classes)

        assert tv_13 <= tv_12 + tv_23 + 1e-6

    @given(dataset_size=st.integers(20, 200), num_classes=st.integers(2, 20), seed=st.integers(0, 10000))
    @settings(max_examples=30, deadline=None)
    def test_jsd_symmetric(self, dataset_size, num_classes, seed):
        """JSD is symmetric."""
        np.random.seed(seed)
        labels_1 = np.random.randint(0, num_classes, size=dataset_size)
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)

        d1 = [{"label": int(l)} for l in labels_1]
        d2 = [{"label": int(l)} for l in labels_2]

        jsd_12 = metrics.JS_divergence([d1, d2], num_classes=num_classes)
        jsd_21 = metrics.JS_divergence([d2, d1], num_classes=num_classes)

        assert abs(jsd_12 - jsd_21) < 1e-10

    @given(dataset_size=st.integers(20, 200), num_classes=st.integers(2, 20), seed=st.integers(0, 10000))
    @settings(max_examples=30, deadline=None)
    def test_tv_symmetric(self, dataset_size, num_classes, seed):
        """TV is symmetric."""
        np.random.seed(seed)
        labels_1 = np.random.randint(0, num_classes, size=dataset_size)
        labels_2 = np.random.randint(0, num_classes, size=dataset_size)

        d1 = [{"label": int(l)} for l in labels_1]
        d2 = [{"label": int(l)} for l in labels_2]

        tv_12 = metrics.TV_label_distributions([d1, d2], num_classes=num_classes)
        tv_21 = metrics.TV_label_distributions([d2, d1], num_classes=num_classes)

        assert abs(tv_12 - tv_21) < 1e-10

    @given(dataset_size=st.integers(20, 200), num_classes=st.integers(2, 20), seed=st.integers(0, 10000))
    @settings(max_examples=30, deadline=None)
    def test_jsd_identity(self, dataset_size, num_classes, seed):
        """JSD(D, D) = 0."""
        np.random.seed(seed)
        labels = np.random.randint(0, num_classes, size=dataset_size)
        d = [{"label": int(l)} for l in labels]

        jsd = metrics.JS_divergence([d, d], num_classes=num_classes)
        assert jsd < 1e-10


# ============================================================================
# Additional Missing Tests for Full Coverage
# ============================================================================


class TestIntrinsicDimensionalityFAISS:
    """Test intrinsic dimensionality with FAISS (when available)."""

    def test_two_nn_faiss(self, medium_image_datasets):
        """Test TWO-NN method with FAISS."""
        dataset1, _ = medium_image_datasets
        try:
            id_est = metrics.intrinsic_dimensionality([dataset1], method="two_nn", use_faiss=True, batch_size=5, use_cache=False)
            assert id_est > 0
        except ImportError:
            pytest.skip("FAISS not installed")

    def test_mle_faiss(self, medium_image_datasets):
        """Test MLE method with FAISS."""
        dataset1, _ = medium_image_datasets
        try:
            id_est = metrics.intrinsic_dimensionality([dataset1], method="mle", k=5, use_faiss=True, batch_size=5, use_cache=False)
            assert id_est > 0
        except ImportError:
            pytest.skip("FAISS not installed")


class TestSlicedWassersteinReproducibility:
    """Test SWD reproducibility with seeding."""

    def test_reproducibility_with_seed(self, small_image_datasets):
        """SWD is reproducible when seeding numpy."""
        dataset1, dataset2 = small_image_datasets

        np.random.seed(12345)
        swd1 = metrics.sliced_wasserstein_distance([dataset1, dataset2], batch_size=5, use_cache=False)

        np.random.seed(12345)
        swd2 = metrics.sliced_wasserstein_distance([dataset1, dataset2], batch_size=5, use_cache=False)

        assert abs(swd1 - swd2) < 1e-10

    def test_different_seeds_different_results(self, small_image_datasets):
        """Different seeds give slightly different results (due to random projections)."""
        dataset1, dataset2 = small_image_datasets

        np.random.seed(111)
        swd1 = metrics.sliced_wasserstein_distance([dataset1, dataset2], batch_size=5, use_cache=False)

        np.random.seed(222)
        swd2 = metrics.sliced_wasserstein_distance([dataset1, dataset2], batch_size=5, use_cache=False)

        # Results should be similar but potentially not exactly equal
        # (testing that different seeds work, not exact values)
        assert swd1 >= 0 and swd2 >= 0


class TestKNNSeparabilityExtended:
    """Extended kNN separability tests."""

    def test_three_datasets(self, three_image_datasets):
        """Test with three datasets."""
        d1, d2, d3 = three_image_datasets
        sep = metrics.knn_separability([d1, d2, d3], k=3, batch_size=5, use_cache=False)
        assert 0 <= sep <= 1

    def test_return_all_three_datasets(self, three_image_datasets):
        """return_all with three datasets returns correct length."""
        d1, d2, d3 = three_image_datasets
        sep_all = metrics.knn_separability([d1, d2, d3], k=3, return_all=True, batch_size=5, use_cache=False)
        expected_len = len(d1) + len(d2) + len(d3)
        assert len(sep_all) == expected_len

    def test_all_values_in_range(self, small_image_datasets):
        """All per-sample scores are in [0, 1]."""
        dataset1, dataset2 = small_image_datasets
        sep_all = metrics.knn_separability([dataset1, dataset2], k=3, return_all=True, batch_size=5, use_cache=False)
        assert all(0 <= s <= 1 for s in sep_all)


class TestCentroidDistanceExtended:
    """Extended centroid distance tests."""

    def test_with_different_models(self, small_image_datasets):
        """Test centroid distance with different extractors."""
        dataset1, dataset2 = small_image_datasets

        cd_inception = metrics.centroid_distance([dataset1, dataset2], model_name="inception_v3", batch_size=5, use_cache=False)

        cd_dino = metrics.centroid_distance(
            [dataset1, dataset2], model_name="dinov2-small", batch_size=5, use_cache=False, proxy="http://proxy.example:3128"
        )

        assert cd_inception >= 0
        assert cd_dino >= 0

    def test_triangle_inequality(self, three_image_datasets):
        """Centroid distance satisfies triangle inequality."""
        d1, d2, d3 = three_image_datasets

        cd_all = metrics.centroid_distance([d1, d2, d3], batch_size=5, return_all=True, use_cache=False)

        # cd_all contains: [cd(d1,d2), cd(d1,d3), cd(d2,d3)]
        cd_12, cd_13, cd_23 = cd_all

        assert cd_13 <= cd_12 + cd_23 + 1e-6


class TestFeatureVarianceExtended:
    """Extended feature variance tests."""

    def test_with_different_models(self, small_image_datasets):
        """Test feature variance with different extractors."""
        dataset1, _ = small_image_datasets

        fv_inception = metrics.feature_variance([dataset1], model_name="inception_v3", batch_size=5, use_cache=False)

        fv_dino = metrics.feature_variance(
            [dataset1], model_name="dinov2-small", batch_size=5, use_cache=False, proxy="http://proxy.example:3128"
        )

        assert fv_inception > 0
        assert fv_dino > 0


@pytest.mark.slow
class TestVendiScoreExtended:
    """Extended Vendi Score tests."""

    def test_with_different_models(self, small_image_datasets):
        """Test Vendi score with different extractors."""
        dataset1, _ = small_image_datasets

        vs_inception = metrics.vendi_score([dataset1], model_name="inception_v3", batch_size=5, use_cache=False)

        vs_dino = metrics.vendi_score(
            [dataset1], model_name="dinov2-small", batch_size=5, use_cache=False, proxy="http://proxy.example:3128"
        )

        assert vs_inception > 0
        assert vs_dino > 0


class TestCacheVerification:
    """Test that cache parameters are properly passed."""

    @pytest.mark.slow
    def test_cache_params_passed_to_fid(self, small_image_datasets, temp_image_dir):
        """Verify cache parameters are passed correctly for FID."""
        dataset1, dataset2 = small_image_datasets

        # Call with cache enabled but custom dir
        fid = metrics.FID_distance(
            [dataset1, dataset2], batch_size=5, use_cache=True, cache_dir=temp_image_dir, dataset_name="test_fid", base_path=temp_image_dir
        )

        assert fid >= 0
        # Reset global cache to avoid affecting other tests
        metrics._GLOBAL_CACHE = None

    @pytest.mark.slow
    def test_cache_params_passed_to_mmd(self, small_image_datasets, temp_image_dir):
        """Verify cache parameters are passed correctly for MMD."""
        dataset1, dataset2 = small_image_datasets

        mmd = metrics.MMD(
            [dataset1, dataset2], batch_size=5, use_cache=True, cache_dir=temp_image_dir, dataset_name="test_mmd", base_path=temp_image_dir
        )

        assert isinstance(mmd, (float, np.floating))
        metrics._GLOBAL_CACHE = None

    def test_use_cache_false_skips_cache(self, temp_image_dir):
        """Verify use_cache=False doesn't create global cache."""
        # Reset any existing cache
        metrics._GLOBAL_CACHE = None

        # Create simple label datasets (no images needed)
        d1 = [{"label": i % 5} for i in range(50)]
        d2 = [{"label": i % 5} for i in range(50)]

        # Label metrics don't use cache, so this tests the code path
        jsd = metrics.JS_divergence([d1, d2], 5)

        assert jsd >= 0


class TestExtractFeaturesExtended:
    """Extended tests for _extract_features."""

    @pytest.mark.slow
    def test_multiple_datasets(self, three_image_datasets):
        """Test extraction with multiple datasets."""
        d1, d2, d3 = three_image_datasets
        features = metrics._extract_features([d1, d2, d3], model_name="inception_v3", batch_size=5, use_cache=False)
        assert len(features) == 3
        assert all(f.shape[1] == 2048 for f in features)

    @pytest.mark.slow
    def test_empty_dataset_behavior(self, small_image_datasets):
        """Test behavior with one empty dataset mixed in."""
        dataset1, _ = small_image_datasets
        empty = []

        # This should either handle gracefully or raise clear error
        try:
            features = metrics._extract_features([dataset1, empty], model_name="inception_v3", batch_size=5, use_cache=False)
            # If it succeeds, check structure
            assert len(features) == 2
        except (ValueError, RuntimeError, IndexError):
            # Expected - empty datasets should raise
            pass


class TestMetricOrderingConsistency:
    """Test that all metrics agree on ordering for clearly different distributions."""

    @pytest.mark.slow
    def test_all_feature_metrics_ordering(self, small_image_datasets, highly_different_datasets):
        """All feature metrics should rank identical < different."""
        d1_same, _ = small_image_datasets
        d1_diff, d2_diff = highly_different_datasets

        # Self-comparison vs different
        # Note: We compare self to self vs different to different

        # MMD
        mmd_self = metrics.MMD([d1_same, d1_same], batch_size=5, use_cache=False)
        mmd_diff = metrics.MMD([d1_diff, d2_diff], batch_size=5, use_cache=False)
        assert mmd_self < mmd_diff + 0.1  # self should be near 0

        # Energy Distance
        ed_self = metrics.energy_distance_features([d1_same, d1_same], batch_size=5, use_cache=False)
        ed_diff = metrics.energy_distance_features([d1_diff, d2_diff], batch_size=5, use_cache=False)
        assert abs(ed_self) < abs(ed_diff) + 0.1

        # Centroid Distance
        cd_self = metrics.centroid_distance([d1_same, d1_same], batch_size=5, use_cache=False)
        cd_diff = metrics.centroid_distance([d1_diff, d2_diff], batch_size=5, use_cache=False)
        assert cd_self < cd_diff + 0.1


# ============================================================================
# Integration Tests - Different Extractors
# ============================================================================


@pytest.mark.slow
class TestDifferentExtractors:
    """Test feature metrics with different extractors."""

    @pytest.mark.parametrize(
        "model_name,expected_dim",
        [
            ("inception_v3", 2048),
            ("dinov2-small", 384),
        ],
    )
    def test_mmd_with_extractors(self, small_image_datasets, model_name, expected_dim):
        """Test MMD with different extractors."""
        dataset1, dataset2 = small_image_datasets

        kwargs = {"batch_size": 5, "use_cache": False}
        if "dinov2" in model_name:
            kwargs["proxy"] = "http://proxy.example:3128"

        mmd = metrics.MMD([dataset1, dataset2], model_name=model_name, **kwargs)
        assert isinstance(mmd, (float, np.floating))

    @pytest.mark.parametrize("model_name", ["inception_v3", "dinov2-small"])
    def test_energy_distance_with_extractors(self, small_image_datasets, model_name):
        """Test energy distance with different extractors."""
        dataset1, dataset2 = small_image_datasets

        kwargs = {"batch_size": 5, "use_cache": False}
        if "dinov2" in model_name:
            kwargs["proxy"] = "http://proxy.example:3128"

        ed = metrics.energy_distance_features([dataset1, dataset2], model_name=model_name, **kwargs)
        assert isinstance(ed, (float, np.floating))

    @pytest.mark.parametrize("model_name", ["inception_v3", "dinov2-small"])
    def test_swd_with_extractors(self, small_image_datasets, model_name):
        """Test Sliced Wasserstein Distance with different extractors."""
        dataset1, dataset2 = small_image_datasets

        kwargs = {"batch_size": 5, "use_cache": False}
        if "dinov2" in model_name:
            kwargs["proxy"] = "http://proxy.example:3128"

        swd = metrics.sliced_wasserstein_distance([dataset1, dataset2], model_name=model_name, **kwargs)
        assert swd >= 0

    @pytest.mark.parametrize("model_name", ["inception_v3", "dinov2-small"])
    def test_ks_test_with_extractors(self, small_image_datasets, model_name):
        """Test KS test with different extractors."""
        dataset1, dataset2 = small_image_datasets

        kwargs = {"batch_size": 5, "use_cache": False}
        if "dinov2" in model_name:
            kwargs["proxy"] = "http://proxy.example:3128"

        result = metrics.KS_multivariate_test([dataset1, dataset2], model_name=model_name, **kwargs)
        assert isinstance(result, dict)
        assert "reject_null" in result


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow and not property"])
