from __future__ import annotations

from .optuna_proxy import EmbeddingStudyPreset

FULL_DATA_FIXED_EXTRA_ARGS = {
    "stage_train_fractions": [1.0],
    "stage_metric_subsamples": [0],
}


PRESETS: dict[str, EmbeddingStudyPreset] = {
    "tiny_imagenet_embedding_fixed": EmbeddingStudyPreset(
        key="tiny_imagenet_embedding_fixed",
        dataset_name="tiny_imagenet",
        study_mode="fixed",
        default_target_n_clients=10,
        extra_args=FULL_DATA_FIXED_EXTRA_ARGS,
    ),
    "tiny_imagenet_embedding_discovery": EmbeddingStudyPreset(
        key="tiny_imagenet_embedding_discovery",
        dataset_name="tiny_imagenet",
        study_mode="discovery",
    ),
    "food101_embedding_fixed": EmbeddingStudyPreset(
        key="food101_embedding_fixed",
        dataset_name="food101",
        study_mode="fixed",
        default_target_n_clients=10,
        extra_args=FULL_DATA_FIXED_EXTRA_ARGS,
    ),
    "food101_embedding_discovery": EmbeddingStudyPreset(
        key="food101_embedding_discovery",
        dataset_name="food101",
        study_mode="discovery",
    ),
    "ade20k_embedding_fixed": EmbeddingStudyPreset(
        key="ade20k_embedding_fixed",
        dataset_name="ade20k",
        study_mode="fixed",
        default_target_n_clients=10,
        extra_args=FULL_DATA_FIXED_EXTRA_ARGS,
    ),
    "ade20k_embedding_discovery": EmbeddingStudyPreset(
        key="ade20k_embedding_discovery",
        dataset_name="ade20k",
        study_mode="discovery",
    ),
    "PascalVOC_embedding_fixed": EmbeddingStudyPreset(
        key="PascalVOC_embedding_fixed",
        dataset_name="PascalVOC",
        study_mode="fixed",
        default_target_n_clients=10,
        extra_args=FULL_DATA_FIXED_EXTRA_ARGS,
    ),
    "PascalVOC_embedding_discovery": EmbeddingStudyPreset(
        key="PascalVOC_embedding_discovery",
        dataset_name="PascalVOC",
        study_mode="discovery",
    ),
}
