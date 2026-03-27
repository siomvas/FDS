from __future__ import annotations

import numpy as np

from partitioning.correlated_multivariate_dirichlet_splitter import CorrelatedMultivariateDirichletSplitter
from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter


class _DatasetWithName:
    dataset_name = "tiny_imagenet"

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return row["image"], row["label"]


def test_correlated_splitter_resolves_rules_and_adds_bonus(mock_dataset_with_vlm_paths, vlm_annotation_directory):
    dataset = _DatasetWithName(mock_dataset_with_vlm_paths)
    base_splitter = MultivariateDirichletSplitter(
        train_dataset=dataset,
        val_dataset=None,
        n_clients=2,
        heterogeneity=1.0,
        cluster_labels_dir=str(vlm_annotation_directory),
        valset_type="none",
        seed=1,
    )
    corr_splitter = CorrelatedMultivariateDirichletSplitter(
        train_dataset=dataset,
        val_dataset=None,
        n_clients=2,
        heterogeneity=1.0,
        cluster_labels_dir=str(vlm_annotation_directory),
        valset_type="none",
        seed=1,
        correlation_strength=2.0,
    )

    base_splitter._load_cluster_labels()
    corr_splitter._load_cluster_labels()

    client_prefs = {
        0: {
            "color": np.array([0.3, 0.3, 0.4]),
            "main_object": np.array([0.25, 0.25, 0.25, 0.25]),
            "size": np.array([0.3, 0.4, 0.3]),
            "location": np.array([0.8, 0.1, 0.05, 0.05]),
            "time": np.array([0.85, 0.05, 0.1]),
        },
        1: {
            "color": np.array([0.3, 0.3, 0.4]),
            "main_object": np.array([0.25, 0.25, 0.25, 0.25]),
            "size": np.array([0.3, 0.4, 0.3]),
            "location": np.array([0.2, 0.2, 0.3, 0.3]),
            "time": np.array([0.2, 0.3, 0.5]),
        },
    }
    base_splitter.client_preferences = client_prefs
    corr_splitter.client_preferences = client_prefs

    image_path = next(
        row["image"]
        for row in mock_dataset_with_vlm_paths
        if corr_splitter._sample_labels_for_image(row["image"]).get("location") in {"outdoor", "natural", "urban"}
        and corr_splitter._sample_labels_for_image(row["image"]).get("time") in {"day", "dusk"}
    )
    assert corr_splitter._resolved_rules
    assert corr_splitter._compute_affinity_score(image_path, 0) > base_splitter._compute_affinity_score(image_path, 0)
