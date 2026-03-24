from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from data_loading.dataset import BaseDataset
from evaluation import metrics as feature_metrics
from evaluation.evaluate_partitions import compute_balance_metrics, subsample_datasets
from partitioning.embedding_splitter import EmbeddingSplitter
from utils.cli_common import add_standard_cli_args

log = logging.getLogger(__name__)

DEFAULT_EXTRACTORS = [
    "dinov2-base",
    "clip-base-patch16",
    "siglip-base-patch16-224",
    "mae-base",
    "inception_v3",
    "raw_pixel_resizer-64",
    "sparse_random_projection_resizer",
    "UAE",
]
DEFAULT_DISCOVERY_EXTRACTORS = [
    "dinov2-base",
    "clip-base-patch16",
    "mae-base",
]
DEFAULT_FIXED_CLUSTERING = ["kmeans"]
DEFAULT_DISCOVERY_CLUSTERING = ["hdbscan"]
DEFAULT_TOKEN_CHOICES = ["cls", "all_last", "k_last_cls"]
DEFAULT_DISCOVERY_TOKEN_CHOICES = ["cls", "all_last"]
DEFAULT_POOLING_CHOICES = ["mean", "max", "gem"]
DEFAULT_DISCOVERY_POOLING_CHOICES = ["mean", "max"]
DEFAULT_LAYER_AGGREGATIONS = ["mean", "concat", "weighted"]
DEFAULT_LASTK_CHOICES = [2, 4]
DEFAULT_LASTK_POOLING = ["mean", "max", "gem", "cls"]
DEFAULT_STAGE_FRACTIONS = [1.0]
DEFAULT_STAGE_SUBSAMPLES = [0]
DEFAULT_DR_DIMS = [32, 128]
DEFAULT_UMAP_NEIGHBORS = [15]
DEFAULT_UMAP_MIN_DISTS = [0.1]
DEFAULT_GEM_P = [2.0]
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_MULTIPLIERS = [1, 2, 4]
DEFAULT_HDBSCAN_MIN_SAMPLES = [1, 5, 10]
EXTRACTOR_FAMILY_ALIASES = {
    "dino": "dinov2",
}
TOKEN_AWARE_EXTRACTOR_FAMILIES = frozenset(
    {
        "clip",
        "dinov2",
        "mae",
        "siglip",
    }
)
STUDY_POOLING_CHOICES = frozenset(DEFAULT_POOLING_CHOICES)
LASTK_TOKEN_AGGREGATION_CHOICES = frozenset(DEFAULT_LASTK_POOLING)
STUDY_CLUSTERING_CHOICES = frozenset({"kmeans", "hdbscan"})


@dataclass(frozen=True)
class StageConfig:
    fraction: float
    metric_subsample: int


@dataclass
class TrialArtifacts:
    partition_path: Path
    metrics_path: Path
    params_path: Path


class TrialRejected(RuntimeError):
    pass


class TrialPruneSignal(RuntimeError):
    pass


class SubsetDataset:
    """Minimal dataset wrapper used for deterministic staged subsets."""

    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.dataset_name = getattr(dataset, "dataset_name", None)
        self.requested_dataset_name = getattr(dataset, "requested_dataset_name", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Optuna is required for embedding partition tuning. Install the project with the FL extras or add 'optuna'."
        ) from exc
    return optuna


def _parse_csv_items(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or list(default)


def _parse_csv_ints(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return list(default)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(item) for item in items] if items else list(default)


def _parse_csv_floats(raw: str | None, default: list[float | None]) -> list[float | None]:
    if raw is None:
        return list(default)
    out: list[float | None] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if item.lower() == "none":
            out.append(None)
        else:
            out.append(float(item))
    return out or list(default)


def _parse_restricted_csv_items(
    raw: str | None,
    default: list[str],
    *,
    allowed: set[str] | frozenset[str],
    arg_name: str,
) -> list[str]:
    items = _parse_csv_items(raw, default)
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise ValueError(
            f"{arg_name} contains unsupported values {invalid}. Allowed values: {sorted(allowed)}."
        )
    return items


def _resolve_hdbscan_min_cluster_size_choices(args, dataset_len: int) -> list[int]:
    multipliers = _parse_csv_ints(
        getattr(args, "hdbscan_min_cluster_size_multipliers", None),
        DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_MULTIPLIERS,
    )
    min_client_size = int(args.min_client_size)
    raw_choices = sorted({max(min_client_size, min_client_size * multiplier) for multiplier in multipliers})
    valid_choices = [value for value in raw_choices if value <= max(1, dataset_len)]
    if valid_choices:
        return valid_choices
    return [max(2, min(min_client_size, max(2, dataset_len)))]


def _resolve_hdbscan_min_samples_choices(args, min_cluster_size_choices: list[int]) -> list[int]:
    base_choices = _parse_csv_ints(
        getattr(args, "hdbscan_min_samples_choices", None),
        DEFAULT_HDBSCAN_MIN_SAMPLES,
    )
    max_choice = max(min_cluster_size_choices)
    valid = sorted({value for value in base_choices if 1 <= value <= max_choice})
    if valid:
        return valid
    return [1]


def _parse_stage_configs(args) -> list[StageConfig]:
    fractions = [float(value) for value in args.stage_train_fractions]
    metric_subsamples = [int(value) for value in args.stage_metric_subsamples]
    if len(fractions) != len(metric_subsamples):
        raise ValueError("--stage_train_fractions and --stage_metric_subsamples must have the same length.")
    stages = [StageConfig(fraction=fraction, metric_subsample=subsample) for fraction, subsample in zip(fractions, metric_subsamples)]
    if not stages:
        raise ValueError("At least one stage must be configured.")
    if stages[-1].fraction != 1.0:
        raise ValueError("The final stage must use fraction=1.0 so completed trials are evaluated on the full requested train set.")
    for stage in stages:
        if not (0.0 < stage.fraction <= 1.0):
            raise ValueError(f"Invalid stage fraction {stage.fraction}; expected 0 < fraction <= 1.")
        if stage.metric_subsample < 0:
            raise ValueError("Stage metric subsample must be >= 0.")
    return stages


def build_arg_parser() -> argparse.ArgumentParser:
    default_cache_dir = Path.cwd() / "embedding_cache"
    parser = argparse.ArgumentParser(description="Optuna tuning for embedding-based partitioning.")
    add_standard_cli_args(
        parser,
        include_dataset=True,
        dataset_required=True,
        include_base_path=True,
        base_path_required=True,
        include_seed=True,
        seed_default=42,
        include_dry_run=True,
    )
    parser.add_argument("--study_dir", required=True, type=str, help="Directory for Optuna DB and trial artifacts.")
    parser.add_argument("--study_name", default=None, type=str, help="Study name override.")
    parser.add_argument("--n_trials", default=100, type=int, help="Optuna trial budget.")
    parser.add_argument("--timeout", default=None, type=int, help="Optional walltime limit in seconds.")
    parser.add_argument(
        "--sampler",
        choices=["tpe", "brute_force"],
        default="tpe",
        help="Optuna sampler to use.",
    )
    parser.add_argument(
        "--bruteforce_avoid_premature_stop",
        action="store_true",
        help="Use BruteForceSampler's strict exhaustive-search stopping rule.",
    )
    parser.add_argument("--study_mode", choices=["fixed", "discovery"], default="fixed", help="Fixed client-count search or HDBSCAN-style discovery search.")
    parser.add_argument("--objective_metric", choices=["mmd"], default="mmd", help="Optimization objective. Only MMD is supported in the first pass.")
    parser.add_argument("--scorer_model", default="inception_v3", type=str, help="Fixed feature extractor used for evaluation metrics.")
    parser.add_argument(
        "--extractors",
        default=None,
        type=str,
        help=(
            "Comma-separated embedding extractors to tune. "
            f"Default fixed mode: {','.join(DEFAULT_EXTRACTORS)}. "
            f"Default discovery mode: {','.join(DEFAULT_DISCOVERY_EXTRACTORS)}."
        ),
    )
    parser.add_argument("--fixed_clustering_choices", default=",".join(DEFAULT_FIXED_CLUSTERING), type=str, help="Comma-separated clustering methods for fixed mode.")
    parser.add_argument("--discovery_clustering_choices", default=",".join(DEFAULT_DISCOVERY_CLUSTERING), type=str, help="Comma-separated clustering methods for discovery mode.")
    parser.add_argument("--target_n_clients", default=None, type=int, help="Required in fixed mode. Target number of clients/clusters.")
    parser.add_argument("--min_discovered_clients", default=2, type=int, help="Minimum acceptable discovered client count in discovery mode.")
    parser.add_argument("--max_discovered_clients", default=50, type=int, help="Maximum acceptable discovered client count in discovery mode.")
    parser.add_argument("--min_client_size", default=20, type=int, help="Reject trials that create smaller clients than this.")
    parser.add_argument("--embed_batch_size", default=100, type=int, help="Batch size for embedding extraction during partition generation.")
    parser.add_argument("--metric_batch_size", default=128, type=int, help="Batch size for metric feature extraction.")
    parser.add_argument("--num_workers", default=0, type=int, help="Worker count for embedding extraction and metrics.")
    parser.add_argument("--device", default=None, type=str, help="Device for embedding extraction and metrics.")
    parser.add_argument("--cache_dir", default=str(default_cache_dir), type=str, help="Shared cache directory for embeddings and feature metrics.")
    parser.add_argument("--proxy", default=None, type=str, help="Optional proxy passed to extractors.")
    parser.add_argument("--no_cache", action="store_true", help="Disable embedding and metric caching.")
    parser.add_argument(
        "--token_choices",
        default=None,
        type=str,
        help=(
            "Comma-separated token extraction strategies. "
            f"Default fixed mode: {','.join(DEFAULT_TOKEN_CHOICES)}. "
            f"Default discovery mode: {','.join(DEFAULT_DISCOVERY_TOKEN_CHOICES)}."
        ),
    )
    parser.add_argument(
        "--pooling_choices",
        default=None,
        type=str,
        help=(
            "Comma-separated pooling strategies for non-CLS token extraction. "
            f"Default fixed mode: {','.join(DEFAULT_POOLING_CHOICES)}. "
            f"Default discovery mode: {','.join(DEFAULT_DISCOVERY_POOLING_CHOICES)}."
        ),
    )
    parser.add_argument("--layer_aggregation_choices", default=",".join(DEFAULT_LAYER_AGGREGATIONS), type=str, help="Comma-separated layer aggregation choices for last-k token extraction.")
    parser.add_argument("--lastk_choices", default=",".join(str(v) for v in DEFAULT_LASTK_CHOICES), type=str, help="Comma-separated K values for last-k token extraction.")
    parser.add_argument("--lastk_pooling_choices", default=",".join(DEFAULT_LASTK_POOLING), type=str, help="Comma-separated token aggregation choices for last-k pooling.")
    parser.add_argument("--normalize_choices", default="false,true", type=str, help="Comma-separated boolean choices for normalization.")
    parser.add_argument("--dim_reduction_choices", default="none,umap", type=str, help="Comma-separated dimensionality reduction choices.")
    parser.add_argument("--dr_dims_choices", default=",".join(str(v) for v in DEFAULT_DR_DIMS), type=str, help="Comma-separated UMAP output dimensions.")
    parser.add_argument("--umap_neighbors_choices", default=",".join(str(v) for v in DEFAULT_UMAP_NEIGHBORS), type=str, help="Comma-separated UMAP neighbor counts.")
    parser.add_argument("--umap_min_dist_choices", default=",".join(str(v) for v in DEFAULT_UMAP_MIN_DISTS), type=str, help="Comma-separated UMAP min-dist values.")
    parser.add_argument("--gem_p_choices", default=",".join(str(v) for v in DEFAULT_GEM_P), type=str, help="Comma-separated GeM exponents.")
    parser.add_argument("--hdbscan_metrics", default="euclidean,cosine", type=str, help="Comma-separated HDBSCAN metrics.")
    parser.add_argument(
        "--hdbscan_min_cluster_size_multipliers",
        default=",".join(str(v) for v in DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_MULTIPLIERS),
        type=str,
        help="Comma-separated multipliers applied to --min_client_size to define HDBSCAN min_cluster_size choices.",
    )
    parser.add_argument(
        "--hdbscan_min_samples_choices",
        default=",".join(str(v) for v in DEFAULT_HDBSCAN_MIN_SAMPLES),
        type=str,
        help="Comma-separated HDBSCAN min_samples choices.",
    )
    parser.add_argument("--stage_train_fractions", nargs="+", type=float, default=DEFAULT_STAGE_FRACTIONS, help="Progressive train fractions for pruning stages.")
    parser.add_argument("--stage_metric_subsamples", nargs="+", type=int, default=DEFAULT_STAGE_SUBSAMPLES, help="Per-stage max samples per client for metric evaluation; 0 means all.")
    parser.add_argument("--artifact_top_k", default=10, type=int, help="How many top completed trials to keep as explicit artifact directories.")
    return parser


def _resolve_study_name(args) -> str:
    return args.study_name or f"{args.dataset}_{args.study_mode}_embedding_optuna"


def _bool_choices(raw: str) -> list[bool]:
    values = []
    for item in [token.strip().lower() for token in raw.split(",") if token.strip()]:
        if item not in {"true", "false"}:
            raise ValueError(f"Invalid boolean choice '{item}'. Use true/false.")
        values.append(item == "true")
    return values or [False, True]


def _build_progressive_indices(dataset_len: int, stages: list[StageConfig], seed: int) -> list[list[int]]:
    if dataset_len <= 0:
        return [[] for _ in stages]
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(dataset_len)
    outputs: list[list[int]] = []
    for stage in stages:
        count = max(1, int(round(stage.fraction * dataset_len))) if stage.fraction < 1.0 else dataset_len
        outputs.append(sorted(permutation[:count].tolist()))
    return outputs


def _write_stage_manifest(*, output_path: Path, dataset_name: str, stages: list[StageConfig], indices_per_stage: list[list[int]]) -> None:
    payload = {
        "dataset": dataset_name,
        "train_total": max((len(indices) for indices in indices_per_stage), default=0),
        "stages": [
            {
                "stage": stage_idx,
                "fraction": stage.fraction,
                "metric_subsample": stage.metric_subsample,
                "train_selected": len(indices),
                "indices": indices,
            }
            for stage_idx, (stage, indices) in enumerate(zip(stages, indices_per_stage), start=1)
        ],
    }
    _write_json(output_path, payload)


def _sorted_client_ids(splits: dict[str, Any]) -> list[str]:
    return sorted(splits["clients"].keys(), key=lambda value: int(value.split("_")[-1]))


def _client_datasets_from_splits(splits: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return [splits["clients"][client_id]["train"] for client_id in _sorted_client_ids(splits)]


def _trial_invalid_reason(n_clients: int, client_sizes: list[int], args) -> str | None:
    if n_clients < 2:
        return "partition_has_fewer_than_two_clients"
    if any(size < int(args.min_client_size) for size in client_sizes):
        return "partition_has_client_below_min_size"
    if args.study_mode == "fixed" and int(args.target_n_clients) != n_clients:
        return "partition_client_count_mismatch"
    if args.study_mode == "discovery":
        if n_clients < int(args.min_discovered_clients) or n_clients > int(args.max_discovered_clients):
            return "partition_client_count_out_of_bounds"
    return None


def _extractor_family(extractor_name: str) -> str:
    family = extractor_name.split("-", 1)[0].strip().lower()
    return EXTRACTOR_FAMILY_ALIASES.get(family, family)


def _extractor_supports_token_search(extractor_name: str) -> bool:
    return _extractor_family(extractor_name) in TOKEN_AWARE_EXTRACTOR_FAMILIES


def _default_layer_aggregation_choice(args) -> str:
    return _parse_restricted_csv_items(
        args.layer_aggregation_choices,
        DEFAULT_LAYER_AGGREGATIONS,
        allowed=set(DEFAULT_LAYER_AGGREGATIONS),
        arg_name="--layer_aggregation_choices",
    )[0]


def _study_extractor_choices(args) -> list[str]:
    if args.extractors is None:
        if args.study_mode == "discovery":
            return list(DEFAULT_DISCOVERY_EXTRACTORS)
        return list(DEFAULT_EXTRACTORS)
    return _parse_csv_items(args.extractors, DEFAULT_EXTRACTORS)


def _study_token_choices(args) -> list[str]:
    if args.token_choices is None:
        if args.study_mode == "discovery":
            return list(DEFAULT_DISCOVERY_TOKEN_CHOICES)
        return list(DEFAULT_TOKEN_CHOICES)
    return _parse_restricted_csv_items(
        args.token_choices,
        DEFAULT_TOKEN_CHOICES,
        allowed=set(DEFAULT_TOKEN_CHOICES),
        arg_name="--token_choices",
    )


def _study_pooling_choices(args) -> list[str]:
    if args.pooling_choices is None:
        if args.study_mode == "discovery":
            return list(DEFAULT_DISCOVERY_POOLING_CHOICES)
        return list(DEFAULT_POOLING_CHOICES)
    return _parse_restricted_csv_items(
        args.pooling_choices,
        DEFAULT_POOLING_CHOICES,
        allowed=STUDY_POOLING_CHOICES,
        arg_name="--pooling_choices",
    )


def _suggest_non_cls_pooling_params(trial, args) -> tuple[str, dict[str, Any]]:
    pooling_strategy = trial.suggest_categorical(
        "pooling",
        _study_pooling_choices(args),
    )
    pooling_params: dict[str, Any] = {}
    if pooling_strategy == "gem":
        pooling_params["p"] = float(
            trial.suggest_categorical("gem_p", _parse_csv_floats(args.gem_p_choices, DEFAULT_GEM_P))
        )
    return pooling_strategy, pooling_params


def _build_token_and_pooling_params(
    trial,
    args,
    *,
    extractor_name: str,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    # Non-transformer extractors already emit one sample-level vector, so token/pooling
    # search only creates invalid 3D-token pooling paths.
    if not _extractor_supports_token_search(extractor_name):
        return "cls", {}, "cls", {}

    token_choice = trial.suggest_categorical(
        "tokens",
        _study_token_choices(args),
    )
    token_params: dict[str, Any] = {}

    if token_choice == "cls":
        return token_choice, token_params, "cls", {}

    if token_choice in {"k_last_cls", "k_last_all"}:
        token_params["lastk_k"] = int(
            trial.suggest_categorical("lastk_k", _parse_csv_ints(args.lastk_choices, DEFAULT_LASTK_CHOICES))
        )
        # EmbeddingExtractor already materializes k-last outputs as concatenated 2D/3D
        # tensors, so the dedicated "lastk" pooler in embedding_utils is not end-to-end
        # compatible with the current splitter path. Keep the search on combinations that
        # produce valid arrays for clustering.
        token_params["layer_aggregation"] = _default_layer_aggregation_choice(args)
        if token_choice == "k_last_cls":
            return token_choice, token_params, "cls", {}
        pooling_strategy, pooling_params = _suggest_non_cls_pooling_params(trial, args)
        return token_choice, token_params, pooling_strategy, pooling_params

    pooling_strategy, pooling_params = _suggest_non_cls_pooling_params(trial, args)
    return token_choice, token_params, pooling_strategy, pooling_params


def _build_preprocessing_params(trial, args) -> dict[str, Any]:
    normalize = trial.suggest_categorical("normalize", _bool_choices(args.normalize_choices))
    dim_reduction = trial.suggest_categorical("dim_reduction", _parse_csv_items(args.dim_reduction_choices, ["none", "umap"]))
    params: dict[str, Any] = {}
    if normalize:
        params["normalization"] = "l2"
    if dim_reduction == "umap":
        params["dim_reduction"] = "umap"
        params["dr_dims"] = int(trial.suggest_categorical("dr_dims", _parse_csv_ints(args.dr_dims_choices, DEFAULT_DR_DIMS)))
        params["umap_neighbors"] = int(trial.suggest_categorical("umap_neighbors", _parse_csv_ints(args.umap_neighbors_choices, DEFAULT_UMAP_NEIGHBORS)))
        params["umap_min_dist"] = float(trial.suggest_categorical("umap_min_dist", _parse_csv_floats(args.umap_min_dist_choices, DEFAULT_UMAP_MIN_DISTS)))
        params["umap_metric"] = "euclidean"
    return params


def _build_clustering_config(
    trial,
    args,
    *,
    preprocessing_params: dict[str, Any],
    dataset_len: int,
) -> tuple[str, dict[str, Any]]:
    if args.study_mode == "fixed":
        clustering_method = trial.suggest_categorical(
            "clustering",
            _parse_csv_items(args.fixed_clustering_choices, DEFAULT_FIXED_CLUSTERING),
        )
    else:
        clustering_method = trial.suggest_categorical(
            "clustering",
            _parse_csv_items(args.discovery_clustering_choices, DEFAULT_DISCOVERY_CLUSTERING),
        )

    params: dict[str, Any] = {}
    if clustering_method == "kmeans":
        params["n_clusters"] = int(args.target_n_clients)
        params["random_state"] = int(args.seed)
    elif clustering_method == "hdbscan":
        min_cluster_size_choices = _resolve_hdbscan_min_cluster_size_choices(args, dataset_len)
        min_samples_choices = _resolve_hdbscan_min_samples_choices(args, min_cluster_size_choices)
        params["min_cluster_size"] = int(
            trial.suggest_categorical("min_cluster_size", min_cluster_size_choices)
        )
        params["min_samples"] = int(
            trial.suggest_categorical("min_samples", min_samples_choices)
        )
        if preprocessing_params.get("normalization") == "l2":
            params["hdbscan_metric"] = "euclidean"
        else:
            params["hdbscan_metric"] = trial.suggest_categorical(
                "hdbscan_metric",
                _parse_csv_items(args.hdbscan_metrics, ["euclidean", "cosine"]),
            )
        params["hdbscan_method"] = trial.suggest_categorical("hdbscan_method", ["eom", "leaf"])
    else:
        raise ValueError(f"Unsupported clustering method '{clustering_method}'.")
    return clustering_method, params


def _build_splitter(trial, args, train_dataset):
    extractor_name = trial.suggest_categorical("extractor", _study_extractor_choices(args))
    token_choice, token_params, pooling_strategy, pooling_params = _build_token_and_pooling_params(
        trial,
        args,
        extractor_name=extractor_name,
    )
    preprocessing_params = _build_preprocessing_params(trial, args)
    clustering_method, clustering_params = _build_clustering_config(
        trial,
        args,
        preprocessing_params=preprocessing_params,
        dataset_len=len(train_dataset),
    )

    return EmbeddingSplitter(
        train_dataset=train_dataset,
        val_dataset=None,
        extractor_name=extractor_name,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        token_extraction_strategy=token_choice,
        token_params=token_params,
        pooling_strategy=pooling_strategy,
        pooling_params=pooling_params,
        preprocessing_params=preprocessing_params,
        valset_type="none",
        seed=int(args.seed),
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        dataset_name=str(args.dataset),
        base_path=str(Path(args.base_path).expanduser().resolve()),
        device=args.device,
        batch_size=int(args.embed_batch_size),
        use_cache=not args.no_cache,
        proxy=args.proxy,
        num_workers=int(args.num_workers),
    )


def _evaluate_mmd(client_datasets: list[list[dict[str, Any]]], *, args, max_samples_per_client: int) -> float:
    if max_samples_per_client > 0:
        client_datasets = subsample_datasets(client_datasets, max_samples_per_client, seed=int(args.seed))
    values = feature_metrics.MMD(
        client_datasets,
        model_name=str(args.scorer_model),
        batch_size=int(args.metric_batch_size),
        use_cache=not args.no_cache,
        cache_dir=str(Path(args.cache_dir).expanduser().resolve()),
        dataset_name=str(args.dataset),
        base_path=str(Path(args.base_path).expanduser().resolve()),
        proxy=args.proxy,
        return_all=True,
    )
    return float(np.mean(values))


def _evaluate_final_metrics(client_datasets: list[list[dict[str, Any]]], *, args) -> dict[str, Any]:
    balance = compute_balance_metrics(client_datasets)
    scorer_kwargs = {
        "model_name": str(args.scorer_model),
        "batch_size": int(args.metric_batch_size),
        "use_cache": not args.no_cache,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "dataset_name": str(args.dataset),
        "base_path": str(Path(args.base_path).expanduser().resolve()),
        "proxy": args.proxy,
    }
    results = dict(balance)
    mmd_all = feature_metrics.MMD(client_datasets, return_all=True, **scorer_kwargs)
    fid_all = feature_metrics.FID_distance(client_datasets, return_all=True, **scorer_kwargs)
    centroid_all = feature_metrics.centroid_distance(client_datasets, return_all=True, **scorer_kwargs)
    variance_all = feature_metrics.feature_variance(client_datasets, return_all=True, **scorer_kwargs)
    results.update(
        {
            "mmd_features_mean": float(np.mean(mmd_all)),
            "mmd_features_std": float(np.std(mmd_all)),
            "fid_mean": float(np.mean(fid_all)),
            "fid_std": float(np.std(fid_all)),
            "centroid_dist_mean": float(np.mean(centroid_all)),
            "centroid_dist_std": float(np.std(centroid_all)),
            "feature_variance_mean": float(np.mean(variance_all)),
            "feature_variance_std": float(np.std(variance_all)),
        }
    )
    return results


def _trial_artifact_paths(study_root: Path, trial_number: int) -> TrialArtifacts:
    trial_dir = study_root / "trials" / f"trial_{trial_number:04d}"
    return TrialArtifacts(
        partition_path=trial_dir / "partition.json",
        metrics_path=trial_dir / "metrics.json",
        params_path=trial_dir / "params.json",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _run_trial(trial, *, args, study_root: Path, stage_datasets: list[Any], stages: list[StageConfig]) -> float:
    splitter = None
    client_datasets: list[list[dict[str, Any]]] | None = None
    final_metrics: dict[str, Any] = {}
    artifact_paths = _trial_artifact_paths(study_root, trial.number)

    for stage_index, (stage_cfg, stage_dataset) in enumerate(zip(stages, stage_datasets), start=1):
        splitter = _build_splitter(trial, args, stage_dataset)
        splits = splitter.get_client_splits()
        client_datasets = _client_datasets_from_splits(splits)
        client_sizes = [len(dataset) for dataset in client_datasets]
        invalid_reason = _trial_invalid_reason(splitter.n_clients, client_sizes, args)
        if invalid_reason is not None:
            trial.set_user_attr("invalid_reason", invalid_reason)
            raise TrialRejected(invalid_reason)

        stage_mmd = _evaluate_mmd(client_datasets, args=args, max_samples_per_client=int(stage_cfg.metric_subsample))
        trial.report(stage_mmd, step=stage_index)
        trial.set_user_attr(f"stage_{stage_index}_mmd", stage_mmd)
        trial.set_user_attr(f"stage_{stage_index}_n_clients", int(splitter.n_clients))
        if stage_index < len(stages) and trial.should_prune():
            raise TrialPruneSignal(f"pruned_after_stage_{stage_index}")

    if splitter is None or client_datasets is None:
        raise RuntimeError("Trial produced no splitter state.")

    final_metrics = _evaluate_final_metrics(client_datasets, args=args)
    final_metrics["n_clients"] = int(splitter.n_clients)
    final_metrics["client_sizes"] = [len(dataset) for dataset in client_datasets]
    final_metrics["objective"] = float(final_metrics["mmd_features_mean"])

    splitter.store_client_splits(str(artifact_paths.partition_path), format="json")
    _write_json(artifact_paths.metrics_path, final_metrics)
    _write_json(
        artifact_paths.params_path,
        {
            "trial_number": trial.number,
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
        },
    )

    for key, value in final_metrics.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr("partition_path", str(artifact_paths.partition_path))
    return float(final_metrics["mmd_features_mean"])


def _cleanup_top_k_artifacts(study, study_root: Path, top_k: int) -> None:
    completed = [trial for trial in study.trials if getattr(trial.state, "name", "") == "COMPLETE" and trial.value is not None]
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    keep = {trial.number for trial in completed[:top_k]}
    trials_root = study_root / "trials"
    if not trials_root.exists():
        return
    for path in trials_root.iterdir():
        if not path.is_dir() or not path.name.startswith("trial_"):
            continue
        try:
            trial_number = int(path.name.split("_")[-1])
        except ValueError:
            continue
        if trial_number not in keep:
            shutil.rmtree(path)


def _write_trials_csv(study, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["number", "state", "value", "params_json", "user_attrs_json"],
        )
        writer.writeheader()
        for trial in study.trials:
            writer.writerow(
                {
                    "number": trial.number,
                    "state": str(trial.state),
                    "value": trial.value,
                    "params_json": json.dumps(trial.params, sort_keys=True),
                    "user_attrs_json": json.dumps(trial.user_attrs, sort_keys=True, default=str),
                }
            )


def _write_best_artifacts(study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        best_trial = study.best_trial
    except ValueError:
        _write_json(output_dir / "best_trial.json", {"best_trial": None})
        return
    payload = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
    }
    _write_json(output_dir / "best_trial.json", payload)
    partition_path = best_trial.user_attrs.get("partition_path")
    if partition_path:
        shutil.copy2(partition_path, output_dir / "best_partition.json")
        metrics_path = Path(partition_path).with_name("metrics.json")
        params_path = Path(partition_path).with_name("params.json")
        if metrics_path.exists():
            shutil.copy2(metrics_path, output_dir / "best_metrics.json")
        if params_path.exists():
            shutil.copy2(params_path, output_dir / "best_params.json")


def _write_summary(study, output_path: Path, *, args, stages: list[StageConfig], study_name: str) -> None:
    lines = [
        f"# Embedding Partition Tuning: {study_name}",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Study Mode: `{args.study_mode}`",
        f"- Objective: `{args.objective_metric}` maximize",
        f"- Scorer Model: `{args.scorer_model}`",
        f"- Stages: `{[(stage.fraction, stage.metric_subsample) for stage in stages]}`",
        f"- Trials: `{len(study.trials)}`",
        "",
    ]
    try:
        best_trial = study.best_trial
    except ValueError:
        lines.extend(["No completed trials."])
    else:
        lines.extend(
            [
                "## Best Trial",
                "",
                f"- Trial: `{best_trial.number}`",
                f"- Objective: `{best_trial.value}`",
                f"- Params: `{json.dumps(best_trial.params, sort_keys=True)}`",
                f"- Clients: `{best_trial.user_attrs.get('n_clients')}`",
                f"- FID Mean: `{best_trial.user_attrs.get('fid_mean')}`",
                f"- Centroid Distance Mean: `{best_trial.user_attrs.get('centroid_dist_mean')}`",
                f"- Feature Variance Mean: `{best_trial.user_attrs.get('feature_variance_mean')}`",
                f"- Balance CV: `{best_trial.user_attrs.get('balance_cv')}`",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_stage_datasets(train_dataset, stages: list[StageConfig], seed: int) -> tuple[list[Any], list[list[int]]]:
    progressive_indices = _build_progressive_indices(len(train_dataset), stages, seed)
    stage_datasets = []
    for stage, indices in zip(stages, progressive_indices):
        if stage.fraction >= 1.0:
            stage_datasets.append(train_dataset)
        else:
            stage_datasets.append(SubsetDataset(train_dataset, indices))
    return stage_datasets, progressive_indices


def _validate_args(args) -> list[StageConfig]:
    stages = _parse_stage_configs(args)
    _parse_restricted_csv_items(
        args.fixed_clustering_choices,
        DEFAULT_FIXED_CLUSTERING,
        allowed=STUDY_CLUSTERING_CHOICES,
        arg_name="--fixed_clustering_choices",
    )
    _parse_restricted_csv_items(
        args.discovery_clustering_choices,
        DEFAULT_DISCOVERY_CLUSTERING,
        allowed=STUDY_CLUSTERING_CHOICES,
        arg_name="--discovery_clustering_choices",
    )
    _parse_restricted_csv_items(
        args.token_choices,
        DEFAULT_TOKEN_CHOICES,
        allowed=set(DEFAULT_TOKEN_CHOICES),
        arg_name="--token_choices",
    ) if args.token_choices is not None else None
    _parse_restricted_csv_items(
        args.pooling_choices,
        DEFAULT_POOLING_CHOICES,
        allowed=STUDY_POOLING_CHOICES,
        arg_name="--pooling_choices",
    ) if args.pooling_choices is not None else None
    _parse_restricted_csv_items(
        args.layer_aggregation_choices,
        DEFAULT_LAYER_AGGREGATIONS,
        allowed=set(DEFAULT_LAYER_AGGREGATIONS),
        arg_name="--layer_aggregation_choices",
    )
    _parse_restricted_csv_items(
        args.lastk_pooling_choices,
        DEFAULT_LASTK_POOLING,
        allowed=LASTK_TOKEN_AGGREGATION_CHOICES,
        arg_name="--lastk_pooling_choices",
    )
    if args.study_mode == "fixed" and (args.target_n_clients is None or int(args.target_n_clients) < 2):
        raise ValueError("--target_n_clients >= 2 is required in fixed study mode.")
    if args.study_mode == "discovery" and int(args.min_discovered_clients) > int(args.max_discovered_clients):
        raise ValueError("--min_discovered_clients must be <= --max_discovered_clients.")
    if args.scorer_model != "inception_v3":
        log.warning("Using scorer model '%s'. The current recommended fixed scorer is inception_v3.", args.scorer_model)
    return stages


def run_cli(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    stages = _validate_args(args)
    base_path = Path(args.base_path).expanduser().resolve()
    study_dir = Path(args.study_dir).expanduser().resolve()
    study_name = _resolve_study_name(args)
    study_root = study_dir / study_name
    study_root.mkdir(parents=True, exist_ok=True)

    train_dataset = BaseDataset(args.dataset, str(base_path), "train")
    stage_datasets, stage_indices = _prepare_stage_datasets(train_dataset, stages, seed=int(args.seed))
    _write_stage_manifest(
        output_path=study_root / "stage_manifest.json",
        dataset_name=str(args.dataset),
        stages=stages,
        indices_per_stage=stage_indices,
    )

    if args.dry_run:
        log.info("Dataset: %s", args.dataset)
        log.info("Study mode: %s", args.study_mode)
        log.info("Study root: %s", study_root)
        log.info("Objective: %s", args.objective_metric)
        log.info("Scorer model: %s", args.scorer_model)
        log.info("Stage sizes: %s", [len(dataset) for dataset in stage_datasets])
        return 0

    optuna = _require_optuna()
    if args.sampler == "brute_force":
        sampler = optuna.samplers.BruteForceSampler(
            seed=int(args.seed),
            avoid_premature_stop=bool(args.bruteforce_avoid_premature_stop),
        )
        pruner = optuna.pruners.NopPruner()
    else:
        sampler = optuna.samplers.TPESampler(seed=int(args.seed))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    storage_path = study_root / "study.sqlite3"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    log.info(
        "Starting embedding partition study '%s' | dataset=%s mode=%s trials=%s objective=%s scorer=%s",
        study_name,
        args.dataset,
        args.study_mode,
        args.n_trials,
        args.objective_metric,
        args.scorer_model,
    )

    def objective(trial):
        try:
            return _run_trial(
                trial,
                args=args,
                study_root=study_root,
                stage_datasets=stage_datasets,
                stages=stages,
            )
        except TrialPruneSignal as exc:
            trial.set_user_attr("pruned_reason", str(exc))
            raise optuna.TrialPruned() from exc
        except TrialRejected as exc:
            trial.set_user_attr("rejected_reason", str(exc))
            raise optuna.TrialPruned() from exc
        finally:
            gc.collect()

    study.optimize(objective, n_trials=int(args.n_trials), timeout=args.timeout, gc_after_trial=True)

    _cleanup_top_k_artifacts(study, study_root, top_k=int(args.artifact_top_k))
    _write_trials_csv(study, study_root / "trials.csv")
    _write_best_artifacts(study, study_root / "best")
    _write_summary(study, study_root / "summary.md", args=args, stages=stages, study_name=study_name)
    log.info("Study complete. Artifacts written to %s", study_root)
    return 0
