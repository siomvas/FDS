#!/usr/bin/env python3
"""Run all CLEVR-4 criteria while loading each large model only once.

The script is intended for a two-GPU InternVL3.5-38B + Llama-3.3-70B run.
It preserves the same four semantic stages and ground-truth isolation as
``partitioning.vlm.clevr4_pipeline`` while avoiding four repeated 70B loads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
from typing import Any, Sequence

from partitioning.vlm import clevr4_pipeline as pipeline


LOGGER = logging.getLogger("clevr4-large-models")
DEFAULT_INTERNVL_MODEL = "OpenGVLab/InternVL3_5-38B"
DEFAULT_INTERNVL_REVISION = "de99855be3642cd44fe97c9b72d70e5ce2c07f69"
DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_LLAMA_REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
STAGE_ORDER = ("step1", "step2a", "step2b", "step3", "validate")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_root_manifest(path: Path, value: dict[str, Any]) -> None:
    pipeline.atomic_write_json(path, value)


def _root_manifest_path(args: argparse.Namespace) -> Path:
    if args.worker_name is None:
        return args.output_root / "large_model_run_manifest.json"
    if re.fullmatch(r"[A-Za-z0-9_.-]+", args.worker_name) is None:
        raise ValueError(
            "--worker-name may contain only letters, digits, underscores, dots, and dashes."
        )
    return args.output_root / f"large_model_run_manifest_{args.worker_name}.json"


def _criterion_args(args: argparse.Namespace, criterion: str) -> SimpleNamespace:
    max_pixels = pipeline.DEFAULT_MAX_PIXELS[criterion]
    return SimpleNamespace(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        criterion=criterion,
        prompt_dir=args.prompt_dir,
        vlm_model=args.vlm_model,
        vlm_revision=args.vlm_revision,
        llm_model=args.llm_model,
        llm_revision=args.llm_revision,
        device=args.device,
        vlm_batch_size=args.vlm_batch_size,
        llm_batch_size=args.llm_batch_size,
        max_pixels=max_pixels,
        internvl_max_tiles=args.internvl_max_tiles,
        min_label_frequency=args.min_label_frequency,
        seed=args.seed,
    )


def _start_manifest(
    args: argparse.Namespace,
    criterion: str,
    records: Sequence[pipeline.Clevr4Record],
) -> dict[str, Any]:
    paths = pipeline.criterion_paths(args.output_root, criterion)
    manifest = pipeline._manifest_base(_criterion_args(args, criterion), criterion, records)
    manifest["source_sha256s"]["runner"] = pipeline.sha256_file(__file__)
    if paths.manifest.exists():
        try:
            previous = json.loads(paths.manifest.read_text(encoding="utf-8"))
            previous_stages = previous.get("completed_stages", [])
            if (
                previous.get("status") == "complete"
                and isinstance(previous_stages, list)
                and set(previous_stages) == set(STAGE_ORDER)
            ):
                return previous
            manifest["original_started_at"] = previous.get(
                "original_started_at", previous.get("started_at")
            )
            stage_sources = previous.get("stage_source_sha256s", {})
            if not isinstance(stage_sources, dict):
                stage_sources = {}
            else:
                stage_sources = dict(stage_sources)
            previous_sources = previous.get("source_sha256s")
            if not isinstance(previous_sources, dict):
                previous_sources = {"pipeline": previous.get("source_sha256")}
            for stage in previous.get("completed_stages", []):
                stage_sources.setdefault(stage, dict(previous_sources))
            if stage_sources:
                manifest["stage_source_sha256s"] = stage_sources
            if isinstance(previous_stages, list):
                completed = [stage for stage in STAGE_ORDER if stage in previous_stages]
                if completed:
                    manifest["completed_stages"] = completed
        except (json.JSONDecodeError, OSError):
            pass
    pipeline.atomic_write_json(paths.manifest, manifest)
    return manifest


def _assert_complete_step1(
    records: Sequence[pipeline.Clevr4Record],
    output_root: Path,
    criterion: str,
) -> None:
    expected_ids = {record.image_id for record in records}
    completed = pipeline.completed_image_ids(
        pipeline.criterion_paths(output_root, criterion).step1
    )
    if completed != expected_ids:
        raise ValueError(
            f"{criterion} step 1 is incomplete: expected {len(expected_ids)} unique IDs, "
            f"found {len(completed)}."
        )


def _record_completed_stage(
    paths: pipeline.PipelinePaths,
    manifest: dict[str, Any],
    stage: str,
) -> None:
    completed = manifest.setdefault("completed_stages", [])
    if stage not in completed:
        completed.append(stage)
    manifest["completed_stages"] = [
        name for name in STAGE_ORDER if name in completed
    ]
    current_sources = manifest.get("source_sha256s")
    if isinstance(current_sources, dict):
        manifest.setdefault("stage_source_sha256s", {}).setdefault(
            stage, dict(current_sources)
        )
    manifest[f"{stage}_completed_at"] = utc_now()
    pipeline.atomic_write_json(paths.manifest, manifest)


def run_caption_stage(
    args: argparse.Namespace,
    records: Sequence[pipeline.Clevr4Record],
    manifests: dict[str, dict[str, Any]],
) -> None:
    LOGGER.info("Loading shared VLM %s", args.vlm_model)
    model, processor = pipeline._load_vlm(
        args.vlm_model,
        args.vlm_revision,
        args.device,
        pipeline.DEFAULT_MAX_PIXELS["texture"],
        internvl_max_tiles=args.internvl_max_tiles,
    )
    try:
        for criterion in args.criteria:
            paths = pipeline.criterion_paths(args.output_root, criterion)
            paths.output_dir.mkdir(parents=True, exist_ok=True)
            pipeline.run_step1(
                records,
                criterion,
                paths,
                args.prompt_dir,
                model,
                processor,
                device=args.device,
                batch_size=args.vlm_batch_size,
            )
            _assert_complete_step1(records, args.output_root, criterion)
            _record_completed_stage(paths, manifests[criterion], "step1")
    finally:
        del model, processor
        gc.collect()
        pipeline._torch_cleanup()


def run_text_stages(
    args: argparse.Namespace,
    records: Sequence[pipeline.Clevr4Record],
    manifests: dict[str, dict[str, Any]],
) -> None:
    for criterion in args.criteria:
        _assert_complete_step1(records, args.output_root, criterion)
    LOGGER.info("Loading shared LLM %s", args.llm_model)
    model, tokenizer = pipeline._load_llm(
        args.llm_model, args.llm_revision, args.device
    )
    try:
        for criterion in args.criteria:
            paths = pipeline.criterion_paths(args.output_root, criterion)
            if (
                manifests[criterion].get("status") == "complete"
                and set(manifests[criterion].get("completed_stages", []))
                == set(STAGE_ORDER)
            ):
                LOGGER.info("All text stages already complete for %s; skipping.", criterion)
                continue
            pipeline.run_step2a(
                criterion,
                paths,
                args.prompt_dir,
                model,
                tokenizer,
                device=args.device,
                batch_size=args.llm_batch_size,
            )
            _record_completed_stage(paths, manifests[criterion], "step2a")
            categories = pipeline.run_step2b(
                criterion,
                paths,
                args.prompt_dir,
                model,
                tokenizer,
                device=args.device,
                min_label_frequency=args.min_label_frequency,
            )
            _record_completed_stage(paths, manifests[criterion], "step2b")
            pipeline.run_step3(
                criterion,
                paths,
                args.prompt_dir,
                categories,
                model,
                tokenizer,
                device=args.device,
                batch_size=args.llm_batch_size,
            )
            _record_completed_stage(paths, manifests[criterion], "step3")
            report = pipeline.validate_criterion(
                records,
                criterion,
                paths,
                expected_samples=args.expected_samples,
            )
            manifests[criterion].update(
                status="complete",
                completed_stages=["step1", "step2a", "step2b", "step3", "validate"],
                completed_at=utc_now(),
                best_matching_taxonomy=report["best_matching_taxonomy"],
                target_is_best_by_nmi=report["target_is_best_by_nmi"],
            )
            _record_completed_stage(paths, manifests[criterion], "validate")
    finally:
        del model, tokenizer
        gc.collect()
        pipeline._torch_cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "caption", "text", "validate"), default="all"
    )
    parser.add_argument(
        "--criteria", nargs="+", choices=pipeline.CLEVR4_CRITERIA, default=list(pipeline.CLEVR4_CRITERIA)
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path(pipeline.__file__).resolve().parent / "data" / "clevr4",
    )
    parser.add_argument("--vlm-model", default=DEFAULT_INTERNVL_MODEL)
    parser.add_argument("--vlm-revision", default=DEFAULT_INTERNVL_REVISION)
    parser.add_argument("--llm-model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--llm-revision", default=DEFAULT_LLAMA_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlm-batch-size", type=int, default=1)
    parser.add_argument("--llm-batch-size", type=int, default=32)
    parser.add_argument("--internvl-max-tiles", type=int, default=12)
    parser.add_argument("--min-label-frequency", type=int, default=31)
    parser.add_argument("--expected-samples", type=int, default=10_531)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--worker-name",
        help=(
            "Optional collision-safe suffix for concurrent, disjoint criterion workers. "
            "Per-criterion artifacts are unchanged."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.prompt_dir = args.prompt_dir.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    try:
        import numpy as np
        import torch

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    records = pipeline.load_clevr4_records(
        args.dataset_root, expected_samples=args.expected_samples
    )
    root_manifest_path = _root_manifest_path(args)
    root_manifest: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "dataset": "clevr_4_10k_v1",
        "sample_count": len(records),
        "criteria": list(args.criteria),
        "worker_name": args.worker_name,
        "vlm_model": args.vlm_model,
        "vlm_revision": args.vlm_revision,
        "llm_model": args.llm_model,
        "llm_revision": args.llm_revision,
        "internvl_max_tiles": args.internvl_max_tiles,
        "ground_truth_usage": "evaluation_only_after_step3",
        "source_sha256s": {
            "pipeline": pipeline.sha256_file(pipeline.__file__),
            "runner": pipeline.sha256_file(__file__),
        },
    }
    from partitioning.vlm import internvl_adapter

    root_manifest["source_sha256s"]["internvl_adapter"] = pipeline.sha256_file(
        internvl_adapter.__file__
    )
    _write_root_manifest(root_manifest_path, root_manifest)
    manifests = {
        criterion: _start_manifest(args, criterion, records)
        for criterion in args.criteria
    }
    first_manifest = manifests[args.criteria[0]]
    root_manifest["runtime"] = first_manifest["runtime"]
    root_manifest["vlm_attention_backend"] = first_manifest[
        "vlm_attention_backend"
    ]
    root_manifest["llm_attention_backend"] = first_manifest[
        "llm_attention_backend"
    ]
    _write_root_manifest(root_manifest_path, root_manifest)
    try:
        if args.stage in {"all", "caption"}:
            run_caption_stage(args, records, manifests)
        if args.stage in {"all", "text"}:
            run_text_stages(args, records, manifests)
        if args.stage == "validate":
            for criterion in args.criteria:
                paths = pipeline.criterion_paths(args.output_root, criterion)
                report = pipeline.validate_criterion(
                    records,
                    criterion,
                    paths,
                    expected_samples=args.expected_samples,
                )
                manifests[criterion].update(
                    status="complete",
                    completed_at=utc_now(),
                    best_matching_taxonomy=report["best_matching_taxonomy"],
                    target_is_best_by_nmi=report["target_is_best_by_nmi"],
                )
                pipeline.atomic_write_json(paths.manifest, manifests[criterion])
        if set(args.criteria) == set(pipeline.CLEVR4_CRITERIA) and args.stage in {
            "all",
            "text",
            "validate",
        }:
            alignment = pipeline.build_alignment_report(args.output_root)
            root_manifest["alignment_passed"] = alignment[
                "all_criteria_best_match_their_true_taxonomy"
            ]
            if not root_manifest["alignment_passed"]:
                raise RuntimeError("One or more criteria failed true-taxonomy alignment.")
        completed_stages = (
            ["step1"]
            if args.stage == "caption"
            else ["step1", "step2a", "step2b", "step3", "validate"]
        )
        root_manifest.update(
            status="caption_complete" if args.stage == "caption" else "complete",
            completed_stages=completed_stages,
            completed_at=utc_now(),
        )
        _write_root_manifest(root_manifest_path, root_manifest)
    except BaseException as exc:
        root_manifest.update(status="failed", failed_at=utc_now(), error=repr(exc))
        _write_root_manifest(root_manifest_path, root_manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
