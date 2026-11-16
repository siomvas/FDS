#!/usr/bin/env python3
"""Strictly verify a completed CLEVR-4 semantic-pipeline artifact tree."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

from partitioning.vlm import clevr4_pipeline as pipeline


STAGE_FIELDS = {
    "step1": ("image", "split", "criterion", "description"),
    "step2a": (
        "image",
        "split",
        "criterion",
        "description",
        "label",
        "step2a_raw_response",
    ),
    "step3": (
        "image",
        "split",
        "criterion",
        "description",
        "class",
        "step3_raw_response",
        "step3_match_method",
    ),
}
GROUND_TRUTH_FIELD_NAMES = {
    "texture",
    "shape",
    "color",
    "count",
    "ground_truth",
    "ground_truth_label",
    "target",
    "true_label",
}


def _is_hex_digest(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ) is not None


def _check_rows(
    path: Path,
    *,
    stage: str,
    criterion: str,
    expected_ids: set[str],
    errors: list[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    label = f"{criterion}/{stage}"
    try:
        rows = pipeline.read_jsonl(path)
    except (OSError, ValueError) as exc:
        errors.append(f"{label}: could not read {path}: {exc}")
        return [], set()
    ids = [str(row.get("image_id", "")) for row in rows]
    unique_ids = set(ids)
    if len(rows) != len(unique_ids):
        errors.append(f"{label}: {len(rows) - len(unique_ids)} duplicate image IDs")
    missing = expected_ids - unique_ids
    extras = unique_ids - expected_ids
    if missing or extras:
        errors.append(
            f"{label}: missing={len(missing)} extras={len(extras)}; "
            f"first_missing={sorted(missing)[:5]} first_extras={sorted(extras)[:5]}"
        )
    required_fields = STAGE_FIELDS[stage]
    invalid = [
        row.get("image_id")
        for row in rows
        if any(
            not isinstance(row.get(field), str) or not row[field].strip()
            for field in required_fields
        )
    ]
    if invalid:
        errors.append(
            f"{label}: {len(invalid)} rows have blank required fields; first={invalid[:5]}"
        )
    wrong_criterion = [
        row.get("image_id") for row in rows if row.get("criterion") != criterion
    ]
    if wrong_criterion:
        errors.append(
            f"{label}: {len(wrong_criterion)} rows have the wrong criterion; "
            f"first={wrong_criterion[:5]}"
        )
    forbidden_fields = set(GROUND_TRUTH_FIELD_NAMES)
    if stage == "step1":
        forbidden_fields.update(("label", "class"))
    elif stage == "step2a":
        forbidden_fields.add("class")
    elif stage == "step3":
        forbidden_fields.add("label")
    leaked_fields = [
        {
            "image_id": row.get("image_id"),
            "fields": sorted(forbidden_fields.intersection(row)),
        }
        for row in rows
        if forbidden_fields.intersection(row)
    ]
    if leaked_fields:
        errors.append(
            f"{label}: {len(leaked_fields)} rows contain forbidden label/ground-truth "
            f"fields; first={leaked_fields[:5]}"
        )
    return rows, unique_ids


def _load_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        errors.append(f"{label}: could not read {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}: expected a JSON object in {path}")
        return {}
    return value


def _static_prompt_value_leaks(
    records: list[pipeline.Clevr4Record], prompt_dir: Path
) -> list[dict[str, str]]:
    """Find literal true-label values in static prompt/config text.

    Dynamic step-2b labels are model observations and are intentionally not
    inspected here. This audit covers only text supplied before seeing model
    outputs: the prompt templates, criterion names, and criterion guidance.
    """

    leaks: list[dict[str, str]] = []
    for criterion in pipeline.CLEVR4_CRITERIA:
        config = pipeline.CRITERION_CONFIG[criterion]
        prompt_replacements = {
            "step1_prompt.txt": {
                "CRITERION": config["prompt_label"],
                "CRITERION_GUIDANCE": config["guidance"],
            },
            "step2a_prompt.txt": {
                "CRITERION": config["prompt_label"],
                "CRITERION_GUIDANCE": config["guidance"],
            },
            "step2b_prompt.txt": {
                "CRITERION": config["prompt_label"],
                "NUM_CLASSES_CLUSTER": config["num_classes"],
                "LEN": 0,
            },
            "step3_prompt.txt": {
                "CRITERION": config["prompt_label"],
                "CLASSES": "[]",
            },
        }
        for prompt_name, replacements in prompt_replacements.items():
            rendered = pipeline.render_prompt(prompt_dir, prompt_name, replacements)
            for value in sorted({str(getattr(record, criterion)) for record in records}):
                # Count annotations are numeric, while prompt structure must use
                # numbers for answer indices and taxonomy size. Treating those as
                # semantic label leaks would be a false positive. Word-valued
                # exemplars are prohibited separately below.
                if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
                    continue
                if re.search(
                    rf"(?<!\w){re.escape(value)}(?!\w)", rendered, re.IGNORECASE
                ):
                    leaks.append(
                        {
                            "criterion": criterion,
                            "prompt": prompt_name,
                            "value": value,
                        }
                    )
    return leaks


def verify_run(
    dataset_root: Path,
    output_root: Path,
    *,
    expected_samples: int = 10_531,
) -> dict[str, Any]:
    records = pipeline.load_clevr4_records(
        dataset_root, expected_samples=expected_samples
    )
    expected_ids = {record.image_id for record in records}
    records_by_id = {record.image_id: record for record in records}
    annotation_path = Path(records[0].image).parent.parent / "clevr_4_annots.json"
    annotation_sha256 = pipeline.sha256_file(annotation_path)
    errors: list[str] = []
    criteria: dict[str, Any] = {}
    validations: dict[str, dict[str, Any]] = {}
    recomputed_alignment: dict[str, bool] = {}
    prompt_dir = Path(pipeline.__file__).resolve().parent / "data" / "clevr4"
    expected_prompt_names = {
        "step1_prompt.txt",
        "step2a_prompt.txt",
        "step2b_prompt.txt",
        "step3_prompt.txt",
    }
    current_prompt_hashes = {
        name: pipeline.sha256_file(prompt_dir / name)
        for name in expected_prompt_names
    }
    static_prompt_value_leaks = _static_prompt_value_leaks(records, prompt_dir)
    exemplar_configuration_present = any(
        "exemplar" in config for config in pipeline.CRITERION_CONFIG.values()
    ) or any(
        "EXEMPLAR" in (prompt_dir / name).read_text(encoding="utf-8")
        for name in expected_prompt_names
    )
    if exemplar_configuration_present:
        errors.append(
            "static_prompts: class exemplars or an exemplar placeholder are present"
        )
    if static_prompt_value_leaks:
        errors.append(
            "static_prompts: ground-truth label values are present in model prompt "
            f"text: {static_prompt_value_leaks}"
        )

    for criterion in pipeline.CLEVR4_CRITERIA:
        paths = pipeline.criterion_paths(output_root, criterion)
        stage_rows: dict[str, list[dict[str, Any]]] = {}
        for stage, path in (
            ("step1", paths.step1),
            ("step2a", paths.step2a),
            ("step3", paths.step3),
        ):
            rows, _ = _check_rows(
                path,
                stage=stage,
                criterion=criterion,
                expected_ids=expected_ids,
                errors=errors,
            )
            stage_rows[stage] = rows

        rows_by_stage = {
            stage: {str(row.get("image_id", "")): row for row in rows}
            for stage, rows in stage_rows.items()
        }
        metadata_mismatches: list[str] = []
        lineage_mismatches: list[str] = []
        for image_id in sorted(expected_ids):
            step1 = rows_by_stage["step1"].get(image_id)
            if step1 is None:
                continue
            record = records_by_id[image_id]
            if (
                step1.get("image") != record.image
                or step1.get("split") != record.split
                or step1.get("criterion") != criterion
            ):
                metadata_mismatches.append(image_id)
            for stage in ("step2a", "step3"):
                row = rows_by_stage[stage].get(image_id)
                if row is None:
                    continue
                if any(
                    row.get(field) != step1.get(field)
                    for field in ("image", "split", "criterion", "description")
                ):
                    lineage_mismatches.append(f"{stage}:{image_id}")
        if metadata_mismatches:
            errors.append(
                f"{criterion}/step1: {len(metadata_mismatches)} rows disagree with "
                f"official image metadata; first={metadata_mismatches[:5]}"
            )
        if lineage_mismatches:
            errors.append(
                f"{criterion}/lineage: {len(lineage_mismatches)} downstream rows do "
                f"not preserve step1 fields; first={lineage_mismatches[:5]}"
            )

        categories: list[str] = []
        try:
            categories = pipeline.parse_categories(
                paths.step2b.read_text(encoding="utf-8"), expected=10
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            errors.append(f"{criterion}/step2b: {exc}")
        step2b_metadata = _load_json(
            paths.step2b_metadata, errors, f"{criterion}/step2b_metadata"
        )
        if step2b_metadata.get("criterion") != criterion:
            errors.append(f"{criterion}/step2b_metadata: criterion does not match")
        if step2b_metadata.get("categories") != categories:
            errors.append(
                f"{criterion}/step2b_metadata: categories do not match step2b output"
            )
        if not isinstance(step2b_metadata.get("all_label_counts"), dict):
            errors.append(f"{criterion}/step2b_metadata: all_label_counts is missing")
        unknown_classes = [
            row.get("image_id")
            for row in stage_rows["step3"]
            if row.get("class") == "Unknown"
        ]
        invalid_classes = [
            row.get("image_id")
            for row in stage_rows["step3"]
            if row.get("class") not in categories
            and row.get("class") != "Unknown"
        ]
        if invalid_classes:
            errors.append(
                f"{criterion}/step3: {len(invalid_classes)} classes are outside the "
                f"consolidated taxonomy; first={invalid_classes[:5]}"
            )

        manifest = _load_json(paths.manifest, errors, f"{criterion}/manifest")
        if manifest.get("status") != "complete":
            errors.append(
                f"{criterion}/manifest: status is {manifest.get('status')!r}, not 'complete'"
            )
        expected_stages = {"step1", "step2a", "step2b", "step3", "validate"}
        completed_stages = manifest.get("completed_stages")
        if completed_stages is not None and (
            not isinstance(completed_stages, list)
            or set(completed_stages) != expected_stages
        ):
            errors.append(
                f"{criterion}/manifest: completed_stages does not contain every stage"
            )
        for field, expected in (
            ("sample_count", expected_samples),
            ("criterion", criterion),
            ("true_criteria", list(pipeline.CLEVR4_CRITERIA)),
            ("ground_truth_usage", "evaluation_only_after_step3"),
        ):
            if manifest.get(field) != expected:
                errors.append(
                    f"{criterion}/manifest: {field} is {manifest.get(field)!r}, "
                    f"expected {expected!r}"
                )
        for field in ("vlm_model", "vlm_revision", "llm_model", "llm_revision"):
            if not str(manifest.get(field, "")).strip():
                errors.append(f"{criterion}/manifest: {field} is missing")
        for field in ("vlm_revision", "llm_revision"):
            if not _is_hex_digest(manifest.get(field), 40):
                errors.append(f"{criterion}/manifest: {field} is not a pinned commit")
        if manifest.get("annotation_sha256") != annotation_sha256:
            errors.append(
                f"{criterion}/manifest: annotation_sha256 does not match the dataset"
            )
        prompt_hashes = manifest.get("prompt_sha256")
        if not isinstance(prompt_hashes, dict) or set(prompt_hashes) != expected_prompt_names:
            errors.append(f"{criterion}/manifest: prompt_sha256 is incomplete")
        elif any(not _is_hex_digest(value, 64) for value in prompt_hashes.values()):
            errors.append(f"{criterion}/manifest: prompt_sha256 contains an invalid hash")
        elif prompt_hashes != current_prompt_hashes:
            errors.append(
                f"{criterion}/manifest: prompt_sha256 does not match the audited "
                "prompt templates"
            )
        source_sha256 = manifest.get("source_sha256")
        if not _is_hex_digest(source_sha256, 64):
            errors.append(f"{criterion}/manifest: source_sha256 is invalid")
        source_sha256s = manifest.get("source_sha256s")
        if source_sha256s is not None:
            if not isinstance(source_sha256s, dict) or not source_sha256s:
                errors.append(f"{criterion}/manifest: source_sha256s is invalid")
            else:
                invalid_source_names = [
                    name
                    for name, value in source_sha256s.items()
                    if not isinstance(name, str)
                    or not name.strip()
                    or not _is_hex_digest(value, 64)
                ]
                if invalid_source_names:
                    errors.append(
                        f"{criterion}/manifest: source_sha256s contains invalid "
                        f"entries: {invalid_source_names}"
                    )
                if source_sha256s.get("pipeline") != source_sha256:
                    errors.append(
                        f"{criterion}/manifest: source_sha256 does not match "
                        "source_sha256s.pipeline"
                    )
        stage_source_sha256s = manifest.get("stage_source_sha256s")
        if stage_source_sha256s is not None:
            if not isinstance(stage_source_sha256s, dict):
                errors.append(
                    f"{criterion}/manifest: stage_source_sha256s is invalid"
                )
            else:
                missing_stage_sources = expected_stages - set(stage_source_sha256s)
                extra_stage_sources = set(stage_source_sha256s) - expected_stages
                if missing_stage_sources or extra_stage_sources:
                    errors.append(
                        f"{criterion}/manifest: stage_source_sha256s has "
                        f"missing={sorted(missing_stage_sources)} "
                        f"extra={sorted(extra_stage_sources)}"
                    )
                for stage, sources in stage_source_sha256s.items():
                    if not isinstance(sources, dict) or not sources:
                        errors.append(
                            f"{criterion}/manifest: stage_source_sha256s[{stage!r}] "
                            "is invalid"
                        )
                        continue
                    invalid_stage_sources = [
                        name
                        for name, value in sources.items()
                        if not isinstance(name, str)
                        or not name.strip()
                        or not _is_hex_digest(value, 64)
                    ]
                    if invalid_stage_sources:
                        errors.append(
                            f"{criterion}/manifest: stage_source_sha256s[{stage!r}] "
                            f"contains invalid entries: {invalid_stage_sources}"
                        )
        validation = _load_json(paths.validation, errors, f"{criterion}/validation")
        validations[criterion] = validation
        if validation.get("sample_count") != expected_samples:
            errors.append(
                f"{criterion}/validation: sample_count is "
                f"{validation.get('sample_count')!r}, expected {expected_samples}"
            )
        if validation.get("best_matching_taxonomy") != criterion:
            errors.append(
                f"{criterion}/validation: best match is "
                f"{validation.get('best_matching_taxonomy')!r}"
            )
        if not validation.get("target_is_best_by_nmi"):
            errors.append(f"{criterion}/validation: target_is_best_by_nmi is not true")
        if validation.get("ground_truth_usage") != "evaluation_only_after_step3":
            errors.append(f"{criterion}/validation: ground_truth_usage is invalid")
        if validation.get("unknown_count") != len(unknown_classes):
            errors.append(
                f"{criterion}/validation: unknown_count is "
                f"{validation.get('unknown_count')!r}, recomputed "
                f"{len(unknown_classes)}"
            )

        recomputed: dict[str, dict[str, Any]] = {}
        if set(rows_by_stage["step3"]) == expected_ids:
            predicted = [
                str(rows_by_stage["step3"][record.image_id].get("class", "Unknown"))
                for record in records
            ]
            for taxonomy in pipeline.CLEVR4_CRITERIA:
                truth = [str(getattr(record, taxonomy)) for record in records]
                recomputed[taxonomy] = pipeline._taxonomy_metrics(truth, predicted)
            target_nmi = recomputed[criterion]["nmi"]
            best_nmi = max(value["nmi"] for value in recomputed.values())
            target_is_best = target_nmi >= best_nmi - 1e-12
            best_matching = max(
                recomputed, key=lambda name: recomputed[name]["nmi"]
            )
            recomputed_alignment[criterion] = (
                target_is_best and best_matching == criterion
            )
            if not target_is_best or best_matching != criterion:
                errors.append(
                    f"{criterion}/recomputed_validation: best match is "
                    f"{best_matching!r}, target_is_best={target_is_best}"
                )
            if validation.get("best_matching_taxonomy") != best_matching:
                errors.append(
                    f"{criterion}/validation: reported best match disagrees with "
                    f"recomputed value {best_matching!r}"
                )
            if validation.get("target_is_best_by_nmi") is not target_is_best:
                errors.append(
                    f"{criterion}/validation: reported target_is_best_by_nmi "
                    "disagrees with recomputed value"
                )
            reported_comparisons = validation.get("taxonomy_comparisons", {})
            for taxonomy, computed_metrics in recomputed.items():
                reported_metrics = reported_comparisons.get(taxonomy, {})
                for metric, computed_value in computed_metrics.items():
                    reported_value = reported_metrics.get(metric)
                    if isinstance(computed_value, float):
                        matches = isinstance(reported_value, (int, float)) and math.isclose(
                            float(reported_value), computed_value, rel_tol=1e-12, abs_tol=1e-12
                        )
                    else:
                        matches = reported_value == computed_value
                    if not matches:
                        errors.append(
                            f"{criterion}/validation: {taxonomy}.{metric} is "
                            f"{reported_value!r}, recomputed {computed_value!r}"
                        )
        else:
            recomputed_alignment[criterion] = False

        criteria[criterion] = {
            "stage_row_counts": {
                stage: len(rows) for stage, rows in stage_rows.items()
            },
            "categories": categories,
            "invalid_class_count": len(invalid_classes),
            "unknown_class_count": len(unknown_classes),
            "best_matching_taxonomy": validation.get("best_matching_taxonomy"),
            "target_nmi": recomputed.get(criterion, {}).get("nmi"),
        }

    combined_path = output_root / "semantic_alignment_report.json"
    combined = _load_json(combined_path, errors, "combined_alignment")
    if not combined.get("all_criteria_best_match_their_true_taxonomy"):
        errors.append(
            "combined_alignment: all_criteria_best_match_their_true_taxonomy is not true"
        )
    if combined.get("true_criteria") != list(pipeline.CLEVR4_CRITERIA):
        errors.append("combined_alignment: true_criteria does not match CLEVR-4")
    if combined.get("dataset") != "clevr_4_10k_v1":
        errors.append("combined_alignment: dataset identifier is invalid")
    if combined.get("criterion_reports") != validations:
        errors.append(
            "combined_alignment: embedded criterion reports do not match validation files"
        )
    if not all(recomputed_alignment.get(name, False) for name in pipeline.CLEVR4_CRITERIA):
        errors.append("combined_alignment: independently recomputed alignment failed")

    report = {
        "status": "passed" if not errors else "failed",
        "dataset": "clevr_4_10k_v1",
        "expected_samples": expected_samples,
        "criteria": criteria,
        "class_exemplar_configuration_present": exemplar_configuration_present,
        "static_prompt_ground_truth_value_leaks": static_prompt_value_leaks,
        "errors": errors,
        "created_at": pipeline.utc_now(),
    }
    pipeline.atomic_write_json(output_root / "artifact_verification.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=10_531)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = verify_run(
        args.dataset_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        expected_samples=args.expected_samples,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
