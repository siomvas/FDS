#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from data_loading.dataset import BaseDataset
from evaluation import metrics as feature_metrics
from partitioning.embedding_splitter import EmbeddingSplitter

log = logging.getLogger(__name__)

DEFAULT_DATASETS = ("food101", "ade20k", "PascalVOC")


@dataclass(frozen=True)
class CandidateTrial:
    dataset: str
    study_dir: Path
    study_name: str
    trial_number: int
    trial_value: float
    params: dict[str, Any]
    user_attrs: dict[str, Any]
    n_clients: int


def _sorted_client_ids(splits: dict[str, Any]) -> list[str]:
    return sorted(splits["clients"].keys(), key=lambda value: int(value.split("_")[-1]))


def _client_datasets_from_splits(splits: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return [splits["clients"][client_id]["train"] for client_id in _sorted_client_ids(splits)]


def _trial_signature(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _get_n_clients(user_attrs: dict[str, Any]) -> int | None:
    direct = user_attrs.get("n_clients")
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            pass
    stage_keys = [key for key in user_attrs if key.startswith("stage_") and key.endswith("_n_clients")]
    for key in sorted(stage_keys, reverse=True):
        value = user_attrs.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _discover_studies(
    source_root: Path,
    datasets: set[str],
    *,
    path_contains: list[str] | None = None,
) -> dict[str, list[tuple[Path, str]]]:
    result: dict[str, list[tuple[Path, str]]] = {dataset: [] for dataset in datasets}
    for sqlite_path in source_root.rglob("study.sqlite3"):
        if path_contains:
            raw = str(sqlite_path)
            if not any(token in raw for token in path_contains):
                continue
        study_dir = sqlite_path.parent
        manifest_path = study_dir / "stage_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        dataset = str(manifest.get("dataset", ""))
        if dataset not in datasets:
            continue
        storage = f"sqlite:///{sqlite_path}"
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage)
        except Exception:
            continue
        for summary in summaries:
            result[dataset].append((study_dir, summary.study_name))
    return result


def _load_trials_for_study(study_dir: Path, study_name: str) -> list[CandidateTrial]:
    storage = f"sqlite:///{study_dir / 'study.sqlite3'}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    dataset = "unknown"
    manifest_path = study_dir / "stage_manifest.json"
    if manifest_path.exists():
        try:
            dataset = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset", "unknown"))
        except json.JSONDecodeError:
            pass

    trials: list[CandidateTrial] = []
    for trial in study.trials:
        if getattr(trial.state, "name", "") != "COMPLETE" or trial.value is None:
            continue
        params = dict(trial.params)
        if params.get("clustering") != "kmeans":
            continue
        n_clients = _get_n_clients(dict(trial.user_attrs))
        if n_clients is None:
            continue
        trials.append(
            CandidateTrial(
                dataset=dataset,
                study_dir=study_dir,
                study_name=study_name,
                trial_number=int(trial.number),
                trial_value=float(trial.value),
                params=params,
                user_attrs=dict(trial.user_attrs),
                n_clients=int(n_clients),
            )
        )
    return trials


def _dedup_trials_by_params(trials: list[CandidateTrial]) -> list[CandidateTrial]:
    buckets: dict[str, list[CandidateTrial]] = {}
    for trial in trials:
        buckets.setdefault(_trial_signature(trial.params), []).append(trial)
    deduped: list[CandidateTrial] = []
    for _, group in buckets.items():
        group_sorted = sorted(group, key=lambda item: item.trial_number)
        mean_value = float(np.mean([item.trial_value for item in group_sorted]))
        base = group_sorted[0]
        deduped.append(
            CandidateTrial(
                dataset=base.dataset,
                study_dir=base.study_dir,
                study_name=base.study_name,
                trial_number=base.trial_number,
                trial_value=mean_value,
                params=base.params,
                user_attrs=base.user_attrs,
                n_clients=base.n_clients,
            )
        )
    deduped.sort(key=lambda item: item.trial_value, reverse=True)
    return deduped


def _select_candidates(
    source_root: Path,
    datasets: list[str],
    *,
    path_contains: list[str] | None = None,
) -> dict[str, dict[str, CandidateTrial]]:
    studies = _discover_studies(source_root, set(datasets), path_contains=path_contains)
    selected: dict[str, dict[str, CandidateTrial]] = {}
    for dataset in datasets:
        all_trials: list[CandidateTrial] = []
        for study_dir, study_name in studies.get(dataset, []):
            all_trials.extend(_load_trials_for_study(study_dir, study_name))
        if not all_trials:
            raise RuntimeError(f"No completed kmeans trials found for dataset '{dataset}' under {source_root}")
        all_trials = _dedup_trials_by_params(all_trials)
        top = all_trials[0]
        same_study = [item for item in all_trials if item.study_dir == top.study_dir and item.study_name == top.study_name]
        non_inception = next((item for item in same_study if item.params.get("extractor") != "inception_v3"), None)
        if non_inception is None:
            non_inception = next((item for item in all_trials if item.params.get("extractor") != "inception_v3"), None)
        if non_inception is None:
            raise RuntimeError(f"No non-inception trial found for dataset '{dataset}'")
        selected[dataset] = {"top": top, "top_non_inception": non_inception}
    return selected


def _build_splitter_from_candidate(
    *,
    candidate: CandidateTrial,
    train_dataset: Any,
    dataset_name: str,
    base_path: Path,
    cache_dir: Path,
    embed_batch_size: int,
    device: str | None,
    seed: int,
    no_cache: bool,
    proxy: str | None,
    num_workers: int,
) -> EmbeddingSplitter:
    params = dict(candidate.params)
    extractor_name = str(params["extractor"])
    token_strategy = str(params.get("tokens") or "cls")
    token_params: dict[str, Any] = {}
    if "lastk_k" in params:
        token_params["lastk_k"] = int(params["lastk_k"])
        token_params["layer_aggregation"] = "mean"

    pooling_strategy = str(params.get("pooling") or "cls")
    pooling_params: dict[str, Any] = {}
    if pooling_strategy == "gem" and "gem_p" in params:
        pooling_params["p"] = float(params["gem_p"])

    preprocessing_params: dict[str, Any] = {}
    if bool(params.get("normalize")):
        preprocessing_params["normalization"] = "l2"
    if params.get("dim_reduction") == "umap":
        preprocessing_params["dim_reduction"] = "umap"
        preprocessing_params["dr_dims"] = int(params["dr_dims"])
        preprocessing_params["umap_neighbors"] = int(params["umap_neighbors"])
        preprocessing_params["umap_min_dist"] = float(params["umap_min_dist"])
        preprocessing_params["umap_metric"] = "euclidean"

    clustering_method = str(params.get("clustering") or "kmeans")
    if clustering_method != "kmeans":
        raise ValueError(f"Expected kmeans candidate, got '{clustering_method}'")
    clustering_params = {"n_clusters": int(candidate.n_clients), "random_state": int(seed)}

    return EmbeddingSplitter(
        train_dataset=train_dataset,
        val_dataset=None,
        extractor_name=extractor_name,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        token_extraction_strategy=token_strategy,
        token_params=token_params,
        pooling_strategy=pooling_strategy,
        pooling_params=pooling_params,
        preprocessing_params=preprocessing_params,
        valset_type="none",
        seed=int(seed),
        cache_dir=str(cache_dir),
        dataset_name=dataset_name,
        base_path=str(base_path),
        device=device,
        batch_size=int(embed_batch_size),
        use_cache=not bool(no_cache),
        proxy=proxy,
        num_workers=int(num_workers),
    )


def _score_candidate_with_scorer(
    *,
    candidate: CandidateTrial,
    dataset_name: str,
    base_path: Path,
    cache_dir: Path,
    scorer_model: str,
    embed_batch_size: int,
    metric_batch_size: int,
    device: str | None,
    seed: int,
    no_cache: bool,
    proxy: str | None,
    num_workers: int,
) -> dict[str, Any]:
    train_dataset = BaseDataset(dataset_name, str(base_path), "train")
    splitter = _build_splitter_from_candidate(
        candidate=candidate,
        train_dataset=train_dataset,
        dataset_name=dataset_name,
        base_path=base_path,
        cache_dir=cache_dir,
        embed_batch_size=embed_batch_size,
        device=device,
        seed=seed,
        no_cache=no_cache,
        proxy=proxy,
        num_workers=num_workers,
    )
    splits = splitter.get_client_splits()
    client_datasets = _client_datasets_from_splits(splits)
    mmd_values = feature_metrics.MMD(
        client_datasets,
        model_name=scorer_model,
        batch_size=int(metric_batch_size),
        use_cache=not bool(no_cache),
        cache_dir=str(cache_dir),
        dataset_name=dataset_name,
        base_path=str(base_path),
        proxy=proxy,
        return_all=True,
    )
    mmd_mean = float(np.mean(mmd_values))
    return {
        "mmd_mean": mmd_mean,
        "mmd_pairwise_count": int(len(mmd_values)),
        "n_clients": int(candidate.n_clients),
        "client_sizes": [int(len(client_ds)) for client_ds in client_datasets],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-score top inception and top non-inception partitioning configs with UAE scorer features."
    )
    parser.add_argument("--source-root", required=True, type=str, help="Root directory containing completed study folders.")
    parser.add_argument("--base-path", required=True, type=str, help="Dataset base path.")
    parser.add_argument("--cache-dir", required=True, type=str, help="Embedding/metric cache directory.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory where outputs are written.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), type=str, help="Comma-separated datasets.")
    parser.add_argument("--scorer-model", default="UAE", type=str, help="Scorer feature model.")
    parser.add_argument(
        "--study-path-contains",
        default="",
        type=str,
        help="Comma-separated substrings required in study.sqlite3 paths (optional filter).",
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed used for deterministic splitter behavior.")
    parser.add_argument("--embed-batch-size", default=100, type=int, help="Embedding extraction batch size.")
    parser.add_argument("--metric-batch-size", default=128, type=int, help="Metric extraction batch size.")
    parser.add_argument("--num-workers", default=4, type=int, help="Number of dataloader workers.")
    parser.add_argument("--device", default=None, type=str, help="Device override for extraction and scoring.")
    parser.add_argument("--proxy", default=None, type=str, help="Optional HTTP proxy.")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding and metric caches.")
    parser.add_argument("--select-only", action="store_true", help="Only select and report candidate trials without rescoring.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    if not datasets:
        raise ValueError("No datasets provided.")
    source_root = Path(args.source_root).expanduser().resolve()
    base_path = Path(args.base_path).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path_contains = [item.strip() for item in str(args.study_path_contains).split(",") if item.strip()]
    selected = _select_candidates(source_root=source_root, datasets=datasets, path_contains=path_contains or None)
    selection_payload: dict[str, Any] = {}
    results_payload: dict[str, Any] = {}

    for dataset in datasets:
        top = selected[dataset]["top"]
        alt = selected[dataset]["top_non_inception"]
        selection_payload[dataset] = {
            "top": {
                "study_dir": str(top.study_dir),
                "study_name": top.study_name,
                "trial_number": top.trial_number,
                "trial_value": top.trial_value,
                "n_clients": top.n_clients,
                "extractor": top.params.get("extractor"),
                "params": top.params,
            },
            "top_non_inception": {
                "study_dir": str(alt.study_dir),
                "study_name": alt.study_name,
                "trial_number": alt.trial_number,
                "trial_value": alt.trial_value,
                "n_clients": alt.n_clients,
                "extractor": alt.params.get("extractor"),
                "params": alt.params,
            },
        }
        log.info(
            "Dataset=%s selected top trial #%s (%s, inception_score=%.6f) vs non-inception trial #%s (%s, inception_score=%.6f)",
            dataset,
            top.trial_number,
            top.params.get("extractor"),
            top.trial_value,
            alt.trial_number,
            alt.params.get("extractor"),
            alt.trial_value,
        )

        if not args.select_only:
            top_score = _score_candidate_with_scorer(
                candidate=top,
                dataset_name=dataset,
                base_path=base_path,
                cache_dir=cache_dir,
                scorer_model=str(args.scorer_model),
                embed_batch_size=int(args.embed_batch_size),
                metric_batch_size=int(args.metric_batch_size),
                device=args.device,
                seed=int(args.seed),
                no_cache=bool(args.no_cache),
                proxy=args.proxy,
                num_workers=int(args.num_workers),
            )
            alt_score = _score_candidate_with_scorer(
                candidate=alt,
                dataset_name=dataset,
                base_path=base_path,
                cache_dir=cache_dir,
                scorer_model=str(args.scorer_model),
                embed_batch_size=int(args.embed_batch_size),
                metric_batch_size=int(args.metric_batch_size),
                device=args.device,
                seed=int(args.seed),
                no_cache=bool(args.no_cache),
                proxy=args.proxy,
                num_workers=int(args.num_workers),
            )
            winner_uae = "top" if float(top_score["mmd_mean"]) >= float(alt_score["mmd_mean"]) else "top_non_inception"
            results_payload[dataset] = {
                "top": top_score,
                "top_non_inception": alt_score,
                "winner_by_inception_score": "top",
                "winner_by_uae_score": winner_uae,
                "ranking_preserved": bool(winner_uae == "top"),
                "uae_margin_top_minus_non_inception": float(top_score["mmd_mean"]) - float(alt_score["mmd_mean"]),
            }
            log.info(
                "Dataset=%s UAE score top=%.6f non_inception=%.6f preserved=%s",
                dataset,
                float(top_score["mmd_mean"]),
                float(alt_score["mmd_mean"]),
                winner_uae == "top",
            )

    output_payload = {
        "source_root": str(source_root),
        "datasets": datasets,
        "scorer_model": str(args.scorer_model),
        "study_path_contains": path_contains,
        "select_only": bool(args.select_only),
        "selection": selection_payload,
        "results": results_payload,
    }
    report_path = output_dir / "uae_scorer_confound_report.json"
    report_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
