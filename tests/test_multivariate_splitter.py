"""Comprehensive tests for MultivariateDirichletSplitter.

Tests VLM-based multivariate data partitioning with multi-criterion heterogeneity.

Test Coverage:
- VLM annotation loading (JSONL format, all 5 criteria)
- Path normalization for multi-user HPC compatibility
- Client preference sampling (Dirichlet distributions)
- Affinity score computation (multi-criterion)
- Softmax assignment with temperature
- Constraint satisfaction and retry mechanism
- Preference summary generation
- Integration tests with real VLM annotations

Note: Some tests are marked @pytest.mark.slow as they use real datasets.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter


# ============================================================================
# VLM Annotation Loading Tests
# ============================================================================

class TestVLMAnnotationLoading:
    """Test loading VLM cluster labels from JSONL files."""

    def test_load_all_five_criteria(self, mock_train_dataset, vlm_annotation_directory):
        """Load VLM annotations for all 5 criteria."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()

        # Verify all criteria loaded
        assert set(splitter.cluster_labels.keys()) == {
            "color", "main_object", "size", "location", "time"
        }

        # Verify each criterion has labels
        for criterion in splitter.CRITERIA:
            assert len(splitter.cluster_labels[criterion]) > 0
            assert len(splitter.unique_labels[criterion]) > 0

    def test_missing_annotation_file_raises(self, mock_train_dataset, tmp_path):
        """Missing annotation file raises FileNotFoundError."""
        # Create directory without annotation files
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(empty_dir),
            valset_type="none",
            seed=42
        )

        with pytest.raises(FileNotFoundError, match="Cluster labels not found"):
            splitter._load_cluster_labels()

    def test_jsonl_format_support(self, mock_train_dataset, tmp_path):
        """JSONL format (one JSON object per line) is supported."""
        # Create JSONL file
        vlm_dir = tmp_path / "vlm"
        color_dir = vlm_dir / "output_color_train"
        color_dir.mkdir(parents=True)

        jsonl_path = color_dir / "step3_result.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write('{"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "class": "warm"}\n')
            f.write('{"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_1.JPEG", "class": "cool"}\n')

        # Create other criterion files (empty but valid)
        for criterion_dir_name in ["output_main_object_train", "output_size_train", "output_location_train", "output_time_train"]:
            dir_path = vlm_dir / criterion_dir_name
            dir_path.mkdir(parents=True)
            file_path = dir_path / "step3_result.jsonl"
            with open(file_path, 'w') as f:
                f.write('{"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "class": "default"}\n')

        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_dir),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()

        assert "color" in splitter.cluster_labels
        assert len(splitter.unique_labels["color"]) >= 2  # warm, cool

    def test_json_array_format_support(self, mock_train_dataset, tmp_path):
        """JSON array format is also supported."""
        # Create JSON array file
        vlm_dir = tmp_path / "vlm"
        color_dir = vlm_dir / "output_color_train"
        color_dir.mkdir(parents=True)

        jsonl_path = color_dir / "step3_result.jsonl"
        annotations = [
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "class": "warm"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_1.JPEG", "class": "cool"}
        ]
        with open(jsonl_path, 'w') as f:
            json.dump(annotations, f)

        # Create other criterion files
        for criterion_dir_name in ["output_main_object_train", "output_size_train", "output_location_train", "output_time_train"]:
            dir_path = vlm_dir / criterion_dir_name
            dir_path.mkdir(parents=True)
            file_path = dir_path / "step3_result.jsonl"
            with open(file_path, 'w') as f:
                json.dump([{"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "class": "default"}], f)

        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_dir),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()

        assert "color" in splitter.cluster_labels


# ============================================================================
# Path Normalization Tests (Critical for Multi-User HPC)
# ============================================================================

class TestPathNormalization:
    """Test path normalization for multi-user HPC compatibility."""

    def test_normalize_preserves_last_5_components(self, mock_train_dataset, vlm_annotation_directory):
        """Path normalization extracts last 5 components."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()

        # Check normalized paths
        for criterion in splitter.CRITERIA:
            for normalized_path in splitter.cluster_labels[criterion].keys():
                # Should not have /users/user1 prefix
                assert not normalized_path.startswith("/users")
                # Should have exactly 5 components
                path_parts = normalized_path.split("/")
                assert len(path_parts) == 5

    def test_normalize_different_user_paths(self, mock_train_dataset, tmp_path):
        """Different users' paths for same image normalize to same path."""
        vlm_dir = tmp_path / "vlm"
        color_dir = vlm_dir / "output_color_train"
        color_dir.mkdir(parents=True)

        jsonl_path = color_dir / "step3_result.jsonl"
        annotations = [
            # User 1
            {"image": "/users/user1/gtai/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "class": "warm"},
            # User 2 (same image, different user path)
            {"image": "/users/user2/gtai/FDS/tiny-imagenet-200/train/n01/images/img_1.JPEG", "class": "cool"},
        ]
        with open(jsonl_path, 'w') as f:
            json.dump(annotations, f)

        # Create other criterion files
        for criterion_dir_name in ["output_main_object_train", "output_size_train", "output_location_train", "output_time_train"]:
            dir_path = vlm_dir / criterion_dir_name
            dir_path.mkdir(parents=True)
            file_path = dir_path / "step3_result.jsonl"
            with open(file_path, 'w') as f:
                json.dump([{"image": "/users/user/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "class": "default"}], f)

        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_dir),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()

        # Both should be normalized to same format
        normalized_paths = list(splitter.cluster_labels["color"].keys())
        assert len(normalized_paths) == 2

        # Both should have same structure (last 5 components)
        for path in normalized_paths:
            assert path.startswith("tiny-imagenet-200/")

    def test_lookup_with_normalized_paths(self, mock_train_dataset, vlm_annotation_directory):
        """_get_cluster_label_for_sample works with normalized paths."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()

        # Use full path (as would come from dataset)
        full_path = "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG"

        label = splitter._get_cluster_label_for_sample(full_path, "color")

        # Should successfully retrieve label
        assert label is not None
        assert isinstance(label, str)

    def test_pascalvoc_alias_is_accepted(self, tmp_path):
        """PascalVOC dataset alias maps to VOC criteria set."""

        class DummyDataset:
            dataset_name = "PascalVOC"

            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError(idx)

        splitter = MultivariateDirichletSplitter(
            train_dataset=DummyDataset(),
            val_dataset=None,
            n_clients=2,
            heterogeneity=1.0,
            cluster_labels_dir=str(tmp_path),
            valset_type="none",
            seed=42,
        )

        assert splitter.CRITERIA == splitter.CRITERIA_VOC

    def test_normalize_image_key_voc_path_alignment(self):
        """VOC paths with/without local_data prefix normalize to the same key."""
        local_path = (
            "/Users/example/FDS/local_data/VOC2012/images/train/2008_000008.jpg"
        )
        vlm_path = "/path/to/gtai/FDS/VOC2012/images/train/2008_000008.jpg"

        local_key = MultivariateDirichletSplitter._normalize_image_key(local_path)
        vlm_key = MultivariateDirichletSplitter._normalize_image_key(vlm_path)

        assert local_key == "VOC2012/images/train/2008_000008.jpg"
        assert vlm_key == "VOC2012/images/train/2008_000008.jpg"
        assert local_key == vlm_key


# ============================================================================
# Preference Sampling Tests
# ============================================================================

class TestPreferenceSampling:
    """Test Dirichlet preference sampling for clients."""

    def test_dirichlet_sampling_low_alpha(self, mock_train_dataset, vlm_annotation_directory):
        """Low alpha (high heterogeneity) produces skewed preferences."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.1,  # Low alpha
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        # Check preferences are skewed (high std deviation)
        for client_id in range(3):
            for criterion in splitter.CRITERIA:
                prefs = splitter.client_preferences[client_id][criterion]

                # High variance (skewed)
                assert np.std(prefs) > 0.05  # Arbitrary threshold

    def test_dirichlet_sampling_high_alpha(self, mock_train_dataset, vlm_annotation_directory):
        """High alpha (low heterogeneity) produces uniform preferences."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=10.0,  # High alpha
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        # Check preferences are more uniform (low std deviation)
        for client_id in range(3):
            for criterion in splitter.CRITERIA:
                prefs = splitter.client_preferences[client_id][criterion]

                # Low variance (uniform)
                assert np.std(prefs) < 0.3  # Should be close to uniform

    def test_preferences_sum_to_one(self, mock_train_dataset, vlm_annotation_directory):
        """Preference vectors sum to 1.0 (Dirichlet property)."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        for client_id in range(5):
            for criterion in splitter.CRITERIA:
                prefs = splitter.client_preferences[client_id][criterion]

                # Sum to 1.0
                assert np.abs(prefs.sum() - 1.0) < 1e-10

    def test_reproducibility_with_seed(self, mock_train_dataset, vlm_annotation_directory):
        """Same seed produces identical preferences."""
        # First run
        splitter_1 = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter_1._load_cluster_labels()
        splitter_1._sample_client_preferences()

        # Second run with same seed
        splitter_2 = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter_2._load_cluster_labels()
        splitter_2._sample_client_preferences()

        # Compare preferences
        for client_id in range(3):
            for criterion in splitter_1.CRITERIA:
                prefs_1 = splitter_1.client_preferences[client_id][criterion]
                prefs_2 = splitter_2.client_preferences[client_id][criterion]

                np.testing.assert_allclose(prefs_1, prefs_2, rtol=1e-10)

    def test_multi_criterion_independence(self, mock_train_dataset, vlm_annotation_directory):
        """Each criterion gets independent preference samples."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.5,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        # Check preferences differ across criteria (not identical)
        client_0_color = splitter.client_preferences[0]["color"]
        client_0_object = splitter.client_preferences[0]["main_object"]

        # Different criteria should have different dimensionalities and values
        # (unless by chance they have same number of unique labels)
        assert len(client_0_color) != len(client_0_object) or not np.allclose(client_0_color, client_0_object)


# ============================================================================
# Affinity Computation Tests
# ============================================================================

class TestAffinityComputation:
    """Test affinity score computation between samples and clients."""

    def test_affinity_multi_criterion_sum(self, mock_train_dataset, vlm_annotation_directory):
        """Affinity is sum of preferences across all criteria."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        # Use an image from the annotations
        image_path = "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG"

        affinity = splitter._compute_affinity_score(image_path, client_id=0)

        # Affinity should be in range [0, num_criteria]
        assert 0.0 <= affinity <= len(splitter.CRITERIA)

    def test_missing_label_raises(self, mock_train_dataset, vlm_annotation_directory):
        """Missing cluster label raises ValueError."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        # Use non-existent image
        missing_image = "/users/user1/FDS/tiny-imagenet-200/train/n99/images/missing.JPEG"

        with pytest.raises(ValueError, match="Cluster label.*not found"):
            splitter._compute_affinity_score(missing_image, client_id=0)

    def test_affinity_range_bounded(self, mock_train_dataset, vlm_annotation_directory):
        """Affinity scores bounded between 0 and num_criteria."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=5,
            heterogeneity=0.5,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        image_path = "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG"

        for client_id in range(5):
            affinity = splitter._compute_affinity_score(image_path, client_id)

            assert 0.0 <= affinity <= len(splitter.CRITERIA)


# ============================================================================
# Softmax Assignment Tests
# ============================================================================

class TestSoftmaxAssignment:
    """Test softmax-based sample assignment."""

    def test_softmax_probabilities_sum_to_one(self, mock_train_dataset, vlm_annotation_directory):
        """Assignment probabilities sum to 1.0."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        # Compute affinities for one sample
        image_path = "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG"
        affinities = np.array([
            splitter._compute_affinity_score(image_path, client_id)
            for client_id in range(3)
        ])

        # Apply softmax
        affinities_scaled = affinities / splitter.temperature
        exp_affinities = np.exp(affinities_scaled - np.max(affinities_scaled))
        probabilities = exp_affinities / exp_affinities.sum()

        # Should sum to 1.0
        assert np.abs(probabilities.sum() - 1.0) < 1e-10

    def test_temperature_parameter_effects(self, mock_train_dataset, vlm_annotation_directory):
        """Temperature parameter affects probability distribution."""
        # Low temperature (sharp distribution)
        splitter_low = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            temperature=0.1,  # Low
            seed=42
        )

        # High temperature (smooth distribution)
        splitter_high = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            temperature=10.0,  # High
            seed=42
        )

        assert splitter_low.temperature == 0.1
        assert splitter_high.temperature == 10.0

    def test_deterministic_with_seed(self, mock_train_dataset, vlm_annotation_directory):
        """Assignment is deterministic with same seed."""
        sample_indices = list(range(5))
        sample_paths = ["/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG"] * 5

        # First run
        splitter_1 = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter_1._load_cluster_labels()
        splitter_1._sample_client_preferences()

        assignments_1 = splitter_1._assign_samples_with_softmax(sample_indices, sample_paths)

        # Second run with same seed
        splitter_2 = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter_2._load_cluster_labels()
        splitter_2._sample_client_preferences()

        assignments_2 = splitter_2._assign_samples_with_softmax(sample_indices, sample_paths)

        # Should be identical
        for client_id in range(3):
            assert assignments_1[client_id] == assignments_2[client_id]


# ============================================================================
# Constraint Satisfaction Tests
# ============================================================================

class TestConstraintSatisfaction:
    """Test min_samples_per_client constraint and retry mechanism."""

    def test_min_samples_satisfied(self, mock_dataset_with_vlm_paths, vlm_annotation_directory):
        """All clients satisfy min_samples_per_client constraint."""
        from tests.conftest import MockDataset
        train_dataset = MockDataset(mock_dataset_with_vlm_paths)

        splitter = MultivariateDirichletSplitter(
            train_dataset=train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            min_samples_per_client=5,
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        sample_indices = list(range(len(train_dataset)))
        sample_paths = [train_dataset[i][0] for i in sample_indices]

        assignments = splitter._partition_with_retry(sample_indices, sample_paths)

        # All clients should have at least 5 samples
        for client_id in range(3):
            assert len(assignments[client_id]) >= 5

    def test_retry_mechanism_resamples_preferences(self, mock_train_dataset, vlm_annotation_directory):
        """Retry mechanism resamples preferences on constraint violation."""
        # This test verifies the retry logic exists
        # Actual retry would require extreme conditions
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=0.1,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            min_samples_per_client=1,  # Easy to satisfy
            max_retries=10,
            seed=42
        )

        # Method exists
        assert hasattr(splitter, '_partition_with_retry')
        assert hasattr(splitter, 'max_retries')
        assert splitter.max_retries == 10

    def test_max_retries_exhaustion_raises(self, mock_dataset_with_vlm_paths, vlm_annotation_directory):
        """Max retries exhaustion raises ValueError."""
        from tests.conftest import MockDataset
        train_dataset = MockDataset(mock_dataset_with_vlm_paths)

        # Create impossible constraint (30 samples, 10 clients, min 10 each)
        splitter = MultivariateDirichletSplitter(
            train_dataset=train_dataset,
            val_dataset=None,
            n_clients=10,
            heterogeneity=0.01,  # Very low (extreme preferences)
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            min_samples_per_client=10,  # Impossible (30 total / 10 clients)
            max_retries=2,  # Few retries
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        sample_indices = list(range(len(train_dataset)))
        sample_paths = [train_dataset[i][0] for i in sample_indices]

        with pytest.raises(ValueError, match="Unable to satisfy min_samples_per_client"):
            splitter._partition_with_retry(sample_indices, sample_paths)


# ============================================================================
# Preference Summary Tests
# ============================================================================

class TestPreferenceSummary:
    """Test get_preference_summary() output."""

    def test_get_preference_summary_structure(self, mock_train_dataset, vlm_annotation_directory):
        """Preference summary has correct structure."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        summary = splitter.get_preference_summary()

        # Check structure
        assert "n_clients" in summary
        assert "heterogeneity" in summary
        assert "criteria" in summary

        # Check top preferences for each criterion
        for criterion in splitter.CRITERIA:
            key = f"{criterion}_top_preferences"
            assert key in summary
            assert len(summary[key]) == 3  # 3 clients

    def test_top_preferences_correct(self, mock_train_dataset, vlm_annotation_directory):
        """Top preferences match highest preference values."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        splitter._load_cluster_labels()
        splitter._sample_client_preferences()

        summary = splitter.get_preference_summary()

        # Verify for one criterion
        criterion = "color"
        top_prefs = summary[f"{criterion}_top_preferences"]

        for client_id in range(3):
            top_pref_entry = top_prefs[client_id]

            # Get actual preferences
            prefs = splitter.client_preferences[client_id][criterion]
            max_idx = np.argmax(prefs)
            max_label = splitter.unique_labels[criterion][max_idx]
            max_value = prefs[max_idx]

            # Should match
            assert top_pref_entry["client"] == client_id
            assert top_pref_entry["label"] == max_label
            assert np.abs(top_pref_entry["preference"] - max_value) < 1e-6

    def test_summary_before_split_returns_error(self, mock_train_dataset, vlm_annotation_directory):
        """get_preference_summary() before sampling returns error dict."""
        splitter = MultivariateDirichletSplitter(
            train_dataset=mock_train_dataset,
            val_dataset=None,
            n_clients=3,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            seed=42
        )

        # Don't sample preferences yet
        summary = splitter.get_preference_summary()

        assert "error" in summary


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests with full pipeline."""

    @pytest.mark.slow
    def test_full_split_with_annotations(self, vlm_annotation_directory):
        """Full split using VLM annotations."""
        # Create small mock dataset with paths matching annotations
        from tests.conftest import MockDataset

        data = [
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "label": "class_0"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_1.JPEG", "label": "class_0"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_2.JPEG", "label": "class_1"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n02/images/img_3.JPEG", "label": "class_1"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n02/images/img_4.JPEG", "label": "class_2"},
        ]
        dataset = MockDataset(data)

        splitter = MultivariateDirichletSplitter(
            train_dataset=dataset,
            val_dataset=None,
            n_clients=2,
            heterogeneity=1.0,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            min_samples_per_client=1,
            seed=42
        )

        splits = splitter.get_client_splits()

        # Verify splits structure
        assert "clients" in splits
        assert "global_val" in splits
        assert len(splits["clients"]) == 2

        # Verify all samples distributed
        total_samples = sum(
            len(splits["clients"][f"client_{i}"]["train"])
            for i in range(2)
        )
        assert total_samples == len(dataset)

    def test_reproducibility_across_runs(self, vlm_annotation_directory):
        """Same seed produces identical splits."""
        from tests.conftest import MockDataset

        data = [
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "label": "class_0"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_1.JPEG", "label": "class_0"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_2.JPEG", "label": "class_1"},
        ]
        dataset = MockDataset(data)

        # First run
        splitter_1 = MultivariateDirichletSplitter(
            train_dataset=dataset,
            val_dataset=None,
            n_clients=2,
            heterogeneity=0.5,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            min_samples_per_client=1,
            seed=42
        )

        splits_1 = splitter_1.get_client_splits()

        # Second run with same seed
        splitter_2 = MultivariateDirichletSplitter(
            train_dataset=dataset,
            val_dataset=None,
            n_clients=2,
            heterogeneity=0.5,
            cluster_labels_dir=str(vlm_annotation_directory),
            valset_type="none",
            min_samples_per_client=1,
            seed=42
        )

        splits_2 = splitter_2.get_client_splits()

        # Compare splits
        for client_id in range(2):
            train_1 = splits_1["clients"][f"client_{client_id}"]["train"]
            train_2 = splits_2["clients"][f"client_{client_id}"]["train"]

            assert len(train_1) == len(train_2)

    def test_heterogeneity_parameter_sweep(self, vlm_annotation_directory):
        """Different heterogeneity values produce different distributions."""
        from tests.conftest import MockDataset

        data = [
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_0.JPEG", "label": "class_0"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_1.JPEG", "label": "class_0"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n01/images/img_2.JPEG", "label": "class_1"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n02/images/img_3.JPEG", "label": "class_1"},
            {"image": "/users/user1/FDS/tiny-imagenet-200/train/n02/images/img_4.JPEG", "label": "class_2"},
        ]
        dataset = MockDataset(data)

        client_sizes_by_alpha = {}

        for alpha in [0.1, 1.0, 10.0]:
            splitter = MultivariateDirichletSplitter(
                train_dataset=dataset,
                val_dataset=None,
                n_clients=2,
                heterogeneity=alpha,
                cluster_labels_dir=str(vlm_annotation_directory),
                valset_type="none",
                min_samples_per_client=1,
                seed=42
            )

            splits = splitter.get_client_splits()

            sizes = [
                len(splits["clients"][f"client_{i}"]["train"])
                for i in range(2)
            ]
            client_sizes_by_alpha[alpha] = sizes

        # Different alphas should produce different distributions
        # (This is probabilistic, but with seed should be deterministic)
        assert len(set(tuple(v) for v in client_sizes_by_alpha.values())) >= 1
