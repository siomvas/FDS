#!/usr/bin/env python3
"""Archive CLEVR-4 text artifacts and reset manifests to completed step 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

from partitioning.vlm import clevr4_pipeline as pipeline


TEXT_ARTIFACT_NAMES = (
    "step2a_result.jsonl",
    "step2b_result.txt",
    "step2b_metadata.json",
    "step3_result.jsonl",
    "validation.json",
)
ROOT_TEXT_ARTIFACT_NAMES = (
    "semantic_alignment_report.json",
    "artifact_verification.json",
)


def _source_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    values = manifest.get("source_sha256s")
    if isinstance(values, dict) and values:
        return dict(values)
    value = manifest.get("source_sha256")
    return {"pipeline": value} if isinstance(value, str) and value else {}


def _reset_manifest(
    previous: dict[str, Any], *, backup_name: str, reason: str
) -> dict[str, Any]:
    reset = dict(previous)
    for key in (
        "best_matching_taxonomy",
        "target_is_best_by_nmi",
        "completed_at",
        "step2a_completed_at",
        "step2b_completed_at",
        "step3_completed_at",
        "validate_completed_at",
    ):
        reset.pop(key, None)
    step1_sources = _source_hashes(previous)
    existing_stage_sources = previous.get("stage_source_sha256s")
    if isinstance(existing_stage_sources, dict):
        candidate = existing_stage_sources.get("step1")
        if isinstance(candidate, dict) and candidate:
            step1_sources = dict(candidate)
    reset.update(
        status="caption_complete",
        completed_stages=["step1"],
        stage_source_sha256s={"step1": step1_sources},
        text_reset_at=pipeline.utc_now(),
        text_reset_reason=reason,
        text_reset_backup=backup_name,
    )
    return reset


def prepare_text_rerun(
    dataset_root: Path,
    output_root: Path,
    *,
    backup_name: str,
    reason: str,
    expected_samples: int = 10_531,
) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", backup_name) is None:
        raise ValueError("backup_name may contain only safe filename characters")
    records = pipeline.load_clevr4_records(
        dataset_root, expected_samples=expected_samples
    )
    expected_ids = {record.image_id for record in records}
    backup_root = output_root / backup_name
    if backup_root.exists():
        raise FileExistsError(f"Backup path already exists: {backup_root}")

    manifests: dict[str, dict[str, Any]] = {}
    criteria_with_text: set[str] = set()
    for criterion in pipeline.CLEVR4_CRITERIA:
        paths = pipeline.criterion_paths(output_root, criterion)
        if pipeline.completed_image_ids(paths.step1) != expected_ids:
            raise ValueError(f"{criterion} step 1 does not have exact dataset coverage")
        if not paths.manifest.is_file():
            raise FileNotFoundError(f"{criterion} manifest is missing: {paths.manifest}")
        text_paths = [paths.output_dir / name for name in TEXT_ARTIFACT_NAMES]
        existing_text = [path for path in text_paths if path.is_file()]
        if existing_text and len(existing_text) != len(text_paths):
            missing = [str(path) for path in text_paths if not path.is_file()]
            raise FileNotFoundError(
                f"{criterion} has a partial set of text artifacts; missing: {missing}"
            )
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if existing_text:
            if manifest.get("status") != "complete":
                raise ValueError(
                    f"{criterion} has text artifacts but its manifest is not complete"
                )
            criteria_with_text.add(criterion)
        elif manifest.get("status") == "complete":
            raise ValueError(
                f"{criterion} manifest is complete but its text artifacts are missing"
            )
        manifests[criterion] = manifest

    backup_root.mkdir(parents=True)
    moved: list[str] = []
    for criterion in pipeline.CLEVR4_CRITERIA:
        paths = pipeline.criterion_paths(output_root, criterion)
        criterion_backup = backup_root / paths.output_dir.name
        criterion_backup.mkdir(parents=True)
        sources = [paths.manifest]
        if criterion in criteria_with_text:
            sources.extend(paths.output_dir / name for name in TEXT_ARTIFACT_NAMES)
        for source in sources:
            destination = criterion_backup / source.name
            os.replace(source, destination)
            moved.append(str(destination.relative_to(output_root)))
        pipeline.atomic_write_json(
            paths.manifest,
            _reset_manifest(
                manifests[criterion], backup_name=backup_name, reason=reason
            ),
        )

    for name in ROOT_TEXT_ARTIFACT_NAMES:
        source = output_root / name
        if source.exists():
            destination = backup_root / name
            os.replace(source, destination)
            moved.append(str(destination.relative_to(output_root)))

    report = {
        "status": "prepared",
        "dataset": "clevr_4_10k_v1",
        "sample_count": len(records),
        "criteria": list(pipeline.CLEVR4_CRITERIA),
        "reason": reason,
        "backup": str(backup_root),
        "criteria_with_archived_text": sorted(criteria_with_text),
        "moved_artifacts": moved,
        "created_at": pipeline.utc_now(),
    }
    pipeline.atomic_write_json(output_root / "text_rerun_reset_manifest.json", report)
    first_manifest = manifests[pipeline.CLEVR4_CRITERIA[0]]
    pipeline.atomic_write_json(
        output_root / "large_model_run_manifest.json",
        {
            "status": "caption_complete",
            "completed_stages": ["step1"],
            "completed_at": pipeline.utc_now(),
            "dataset": "clevr_4_10k_v1",
            "sample_count": len(records),
            "criteria": list(pipeline.CLEVR4_CRITERIA),
            "vlm_model": first_manifest.get("vlm_model"),
            "vlm_revision": first_manifest.get("vlm_revision"),
            "llm_model": first_manifest.get("llm_model"),
            "llm_revision": first_manifest.get("llm_revision"),
            "ground_truth_usage": "evaluation_only_after_step3",
            "text_reset_manifest": str(
                output_root / "text_rerun_reset_manifest.json"
            ),
            "source_sha256s": {
                "reset_script": pipeline.sha256_file(__file__),
            },
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backup-name", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--expected-samples", type=int, default=10_531)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_text_rerun(
        args.dataset_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        backup_name=args.backup_name,
        reason=args.reason,
        expected_samples=args.expected_samples,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
