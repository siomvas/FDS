#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loading.dataset import BaseDataset
from embedding.cache_manager import CacheManager
from embedding.embedding_extractors import ModelRegistry
from partitioning.tuning.embedding_optuna import (
    DEFAULT_DISCOVERY_EXTRACTORS,
    DEFAULT_DISCOVERY_POOLING_CHOICES,
    DEFAULT_DISCOVERY_TOKEN_CHOICES,
    DEFAULT_EXTRACTORS,
    DEFAULT_GEM_P,
    DEFAULT_LASTK_CHOICES,
    DEFAULT_POOLING_CHOICES,
    DEFAULT_TOKEN_CHOICES,
    EXTRACTOR_FAMILY_ALIASES,
    TOKEN_AWARE_EXTRACTOR_FAMILIES,
)

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    default_cache_dir = os.environ.get("FDS_CACHE_DIR", str(Path.cwd() / "embedding_cache"))
    parser = argparse.ArgumentParser(description="Warm the shared embedding and scorer-feature cache for a study dataset.")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name as used by FDS.")
    parser.add_argument("--base_path", required=True, type=str, help="Raw dataset base path.")
    parser.add_argument("--cache_dir", default=default_cache_dir, type=str, help="Shared cache directory.")
    parser.add_argument("--study_mode", default="fixed", choices=["fixed", "discovery"], help="Study mode to mirror.")
    parser.add_argument("--scorer_model", default="inception_v3", type=str, help="Scorer model to precompute.")
    parser.add_argument("--device", default=None, type=str, help="Optional device override.")
    parser.add_argument("--embed_batch_size", default=100, type=int, help="Embedding extractor batch size.")
    parser.add_argument("--metric_batch_size", default=128, type=int, help="Scorer feature batch size.")
    parser.add_argument("--num_workers", default=0, type=int, help="Data loader worker count.")
    parser.add_argument("--proxy", default=None, type=str, help="Optional proxy override.")
    parser.add_argument("--manifest_path", default=None, type=str, help="Optional manifest output path.")
    parser.add_argument(
        "--jobs_json",
        default=None,
        type=str,
        help="Optional JSON file containing an explicit list of cache jobs to run. When provided, --dataset is still used for validation but the job list comes from this file.",
    )
    parser.add_argument("--skip_study_embeddings", action="store_true", help="Skip warming study-embedding variants.")
    parser.add_argument("--skip_scorer_features", action="store_true", help="Skip warming scorer-model features.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned cache jobs and exit.")
    parser.add_argument("--log_level", default="INFO", type=str, help="Python logging level.")
    return parser.parse_args()


def _extractor_family(model_name: str) -> str:
    family = model_name.split("-", 1)[0]
    return EXTRACTOR_FAMILY_ALIASES.get(family, family)


def _study_extractors(study_mode: str) -> list[str]:
    if study_mode == "discovery":
        return list(DEFAULT_DISCOVERY_EXTRACTORS)
    return list(DEFAULT_EXTRACTORS)


def _study_token_choices(study_mode: str) -> list[str]:
    if study_mode == "discovery":
        return list(DEFAULT_DISCOVERY_TOKEN_CHOICES)
    return list(DEFAULT_TOKEN_CHOICES)


def _study_pooling_choices(study_mode: str) -> list[str]:
    if study_mode == "discovery":
        return list(DEFAULT_DISCOVERY_POOLING_CHOICES)
    return list(DEFAULT_POOLING_CHOICES)


def _build_jobs(study_mode: str, scorer_model: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for extractor_name in _study_extractors(study_mode):
        family = _extractor_family(extractor_name)
        if family not in TOKEN_AWARE_EXTRACTOR_FAMILIES:
            jobs.append(
                {
                    "kind": "study_embedding",
                    "model_name": extractor_name,
                    "return_which_tokens": "cls",
                    "k_last": None,
                    "pooling_strategy": None,
                    "pooling_params": None,
                }
            )
            continue

        for token_choice in _study_token_choices(study_mode):
            if token_choice == "cls":
                jobs.append(
                    {
                        "kind": "study_embedding",
                        "model_name": extractor_name,
                        "return_which_tokens": "cls",
                        "k_last": None,
                        "pooling_strategy": None,
                        "pooling_params": None,
                    }
                )
            elif token_choice == "k_last_cls":
                for k_last in DEFAULT_LASTK_CHOICES:
                    jobs.append(
                        {
                            "kind": "study_embedding",
                            "model_name": extractor_name,
                            "return_which_tokens": "k_last_cls",
                            "k_last": int(k_last),
                            "pooling_strategy": None,
                            "pooling_params": None,
                        }
                    )
            elif token_choice == "all_last":
                for pooling_strategy in _study_pooling_choices(study_mode):
                    if pooling_strategy == "gem":
                        for gem_p in DEFAULT_GEM_P:
                            jobs.append(
                                {
                                    "kind": "study_embedding",
                                    "model_name": extractor_name,
                                    "return_which_tokens": "all_last",
                                    "k_last": None,
                                    "pooling_strategy": "gem",
                                    "pooling_params": {"p": float(gem_p)},
                                }
                            )
                    else:
                        jobs.append(
                            {
                                "kind": "study_embedding",
                                "model_name": extractor_name,
                                "return_which_tokens": "all_last",
                                "k_last": None,
                                "pooling_strategy": pooling_strategy,
                                "pooling_params": None,
                            }
                        )
    jobs.append(
        {
            "kind": "scorer_feature",
            "model_name": scorer_model,
            "return_which_tokens": None,
            "k_last": None,
            "pooling_strategy": None,
            "pooling_params": None,
        }
    )
    return jobs


def _load_explicit_jobs(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if not isinstance(payload, list):
        raise ValueError("--jobs_json must point to a JSON list of job objects.")
    jobs: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid job entry in --jobs_json: {item!r}")
        job = dict(item)
        if "kind" not in job:
            model_name = str(job.get("model_name") or "")
            return_which_tokens = job.get("return_which_tokens")
            pooling_strategy = job.get("pooling_strategy")
            k_last = job.get("k_last")
            if return_which_tokens is None and pooling_strategy is None and k_last is None and model_name == "inception_v3":
                job["kind"] = "scorer_feature"
            else:
                job["kind"] = "study_embedding"
        jobs.append(job)
    return jobs


def _job_dataset(job: dict[str, Any], default_dataset: str) -> str:
    dataset_name = job.get("dataset") or default_dataset
    return str(dataset_name)


def _manifest_path(args: argparse.Namespace) -> Path | None:
    if not args.manifest_path:
        return None
    return Path(args.manifest_path).expanduser().resolve()


def _write_manifest(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def _run_job(
    *,
    job: dict[str, Any],
    images_by_dataset: dict[str, list[str]],
    default_dataset: str,
    base_path: str,
    cache_manager: CacheManager,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset_name = _job_dataset(job, default_dataset)
    images = images_by_dataset[dataset_name]
    batch_size = int(args.metric_batch_size) if job["kind"] == "scorer_feature" else int(args.embed_batch_size)
    extractor = ModelRegistry.create(
        job["model_name"],
        device=args.device,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        proxy=args.proxy,
        cache_manager=cache_manager,
    )
    start = time.time()
    kwargs: dict[str, Any] = {}
    if job["return_which_tokens"] is not None:
        kwargs["return_which_tokens"] = job["return_which_tokens"]
    if job["k_last"] is not None:
        kwargs["k_last"] = int(job["k_last"])
    if job["pooling_strategy"] is not None:
        kwargs["pooling_strategy"] = str(job["pooling_strategy"])
    if job["pooling_params"] is not None:
        kwargs["pooling_params"] = dict(job["pooling_params"])

    embeddings = extractor.extract_embeddings(
        images=images,
        use_cache=True,
        dataset_name=dataset_name,
        base_path=base_path,
        **kwargs,
    )
    duration = round(time.time() - start, 3)
    shape = list(embeddings.shape) if hasattr(embeddings, "shape") else None
    return {
        **job,
        "duration_seconds": duration,
        "shape": shape,
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    cache_dir = str(Path(args.cache_dir).expanduser().resolve())
    base_path = str(Path(args.base_path).expanduser().resolve())
    manifest_path = _manifest_path(args)

    jobs = _load_explicit_jobs(args.jobs_json) or _build_jobs(args.study_mode, args.scorer_model)
    if args.skip_study_embeddings:
        jobs = [job for job in jobs if job["kind"] != "study_embedding"]
    if args.skip_scorer_features:
        jobs = [job for job in jobs if job["kind"] != "scorer_feature"]

    if args.dry_run:
        payload = {
            "dataset": args.dataset,
            "base_path": base_path,
            "cache_dir": cache_dir,
            "study_mode": args.study_mode,
            "num_images": None,
            "jobs": jobs,
        }
        print(json.dumps(payload, indent=2))
        return 0

    datasets_to_load = sorted({_job_dataset(job, args.dataset) for job in jobs})
    images_by_dataset: dict[str, list[str]] = {}
    for dataset_name in datasets_to_load:
        dataset = BaseDataset(
            dataset_name=dataset_name,
            base_path=base_path,
            split="train",
        )
        images_by_dataset[dataset_name] = [str(path) for path in dataset.data]

    total_images = {name: len(images) for name, images in images_by_dataset.items()}
    log.info(
        "Warming cache for datasets=%s study_mode=%s cache_dir=%s",
        ",".join(datasets_to_load),
        args.study_mode,
        cache_dir,
    )
    cache_manager = CacheManager(cache_dir=cache_dir, enabled=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    completed_jobs: list[dict[str, Any]] = []

    for index, job in enumerate(jobs, start=1):
        dataset_name = _job_dataset(job, args.dataset)
        log.info(
            "[%d/%d] dataset=%s kind=%s model=%s tokens=%s pooling=%s",
            index,
            len(jobs),
            dataset_name,
            job["kind"],
            job["model_name"],
            job["return_which_tokens"],
            job["pooling_strategy"],
        )
        completed_jobs.append(
            _run_job(
                job=job,
                images_by_dataset=images_by_dataset,
                default_dataset=args.dataset,
                base_path=base_path,
                cache_manager=cache_manager,
                args=args,
            )
        )
        _write_manifest(
            manifest_path,
            {
                "dataset": args.dataset,
                "datasets": total_images,
                "base_path": base_path,
                "cache_dir": cache_dir,
                "study_mode": args.study_mode,
                "started_at": started_at,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "num_images": total_images,
                "jobs": completed_jobs,
                "cache_info": cache_manager.get_cache_info(),
            },
        )

    final_payload = {
        "dataset": args.dataset,
        "datasets": total_images,
        "base_path": base_path,
        "cache_dir": cache_dir,
        "study_mode": args.study_mode,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_images": total_images,
        "jobs": completed_jobs,
        "cache_info": cache_manager.get_cache_info(),
    }
    _write_manifest(manifest_path, final_payload)
    log.info("Cache warmup complete: %s", cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
