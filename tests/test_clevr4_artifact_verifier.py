import json
from itertools import product
from pathlib import Path

from partitioning.vlm import clevr4_pipeline as pipeline
from partitioning.vlm.clevr4_pipeline import (
    CLEVR4_CRITERIA,
    build_alignment_report,
    criterion_paths,
    load_clevr4_records,
    sha256_file,
    validate_criterion,
)
from scripts.verify_clevr4_semantic_run import verify_run


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _completed_fixture(tmp_path: Path) -> tuple[Path, Path, int]:
    dataset_root = tmp_path / "dataset"
    images = dataset_root / "images"
    images.mkdir(parents=True)
    annotations = {}
    combinations = list(product(range(2), repeat=4))
    for index, (texture, shape, color, count_value) in enumerate(combinations):
        image_id = f"image-{index:02d}"
        (images / f"{image_id}.png").write_bytes(b"fixture")
        annotations[image_id] = {
            "texture": f"texture-{texture}",
            "shape": f"shape-{shape}",
            "color": f"color-{color}",
            "count": count_value + 1,
            "split": "train",
        }
    (dataset_root / "clevr_4_annots.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )

    output_root = tmp_path / "outputs"
    records = load_clevr4_records(dataset_root, expected_samples=len(combinations))
    prompt_dir = Path(pipeline.__file__).resolve().parent / "data" / "clevr4"
    prompt_hashes = {
        name: sha256_file(prompt_dir / name)
        for name in (
            "step1_prompt.txt",
            "step2a_prompt.txt",
            "step2b_prompt.txt",
            "step3_prompt.txt",
        )
    }
    for criterion in CLEVR4_CRITERIA:
        paths = criterion_paths(output_root, criterion)
        true_categories = sorted(
            {str(annotation[criterion]) for annotation in annotations.values()}
        )
        categories = [
            *true_categories,
            *(f"unused-{index}" for index in range(10 - len(true_categories))),
        ]
        base_rows = [
            {
                "image_id": image_id,
                "image": str(images / f"{image_id}.png"),
                "split": "train",
                "criterion": criterion,
                "description": f"description {index}",
            }
            for index, image_id in enumerate(sorted(annotations))
        ]
        _write_jsonl(paths.step1, base_rows)
        _write_jsonl(
            paths.step2a,
            [
                {
                    **row,
                    "label": str(annotations[row["image_id"]][criterion]),
                    "step2a_raw_response": str(
                        annotations[row["image_id"]][criterion]
                    ),
                }
                for row in base_rows
            ],
        )
        paths.step2b.write_text(
            "\n".join(
                f"Answer {index + 1}: {category}"
                for index, category in enumerate(categories)
            )
            + "\n",
            encoding="utf-8",
        )
        paths.step2b_metadata.write_text(
            json.dumps(
                {
                    "criterion": criterion,
                    "categories": categories,
                    "all_label_counts": {
                        category: sum(
                            str(annotation[criterion]) == category
                            for annotation in annotations.values()
                        )
                        for category in true_categories
                    },
                }
            ),
            encoding="utf-8",
        )
        _write_jsonl(
            paths.step3,
            [
                {
                    **row,
                    "class": str(annotations[row["image_id"]][criterion]),
                    "step3_raw_response": str(
                        annotations[row["image_id"]][criterion]
                    ),
                    "step3_match_method": "exact",
                }
                for row in base_rows
            ],
        )
        paths.manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "completed_stages": [
                        "step1",
                        "step2a",
                        "step2b",
                        "step3",
                        "validate",
                    ],
                    "sample_count": len(combinations),
                    "criterion": criterion,
                    "true_criteria": list(CLEVR4_CRITERIA),
                    "ground_truth_usage": "evaluation_only_after_step3",
                    "vlm_model": "fixture/vlm",
                    "vlm_revision": "a" * 40,
                    "llm_model": "fixture/llm",
                    "llm_revision": "b" * 40,
                    "annotation_sha256": sha256_file(
                        dataset_root / "clevr_4_annots.json"
                    ),
                    "prompt_sha256": prompt_hashes,
                    "source_sha256": "5" * 64,
                    "source_sha256s": {"pipeline": "5" * 64},
                    "stage_source_sha256s": {
                        stage: {"pipeline": "5" * 64}
                        for stage in (
                            "step1",
                            "step2a",
                            "step2b",
                            "step3",
                            "validate",
                        )
                    },
                }
            ),
            encoding="utf-8",
        )
        validate_criterion(
            records,
            criterion,
            paths,
            expected_samples=len(combinations),
        )
    build_alignment_report(output_root)
    return dataset_root, output_root, len(combinations)


def test_completed_artifact_tree_passes_strict_verification(tmp_path):
    dataset_root, output_root, count = _completed_fixture(tmp_path)
    report = verify_run(dataset_root, output_root, expected_samples=count)
    assert report["status"] == "passed"
    assert report["errors"] == []
    assert all(
        result["stage_row_counts"]["step3"] == count
        for result in report["criteria"].values()
    )


def test_incomplete_or_running_artifact_tree_fails_verification(tmp_path):
    dataset_root, output_root, count = _completed_fixture(tmp_path)
    paths = criterion_paths(output_root, "shape")
    paths.step1.write_text(
        "\n".join(paths.step1.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
        encoding="utf-8",
    )
    paths.manifest.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    report = verify_run(dataset_root, output_root, expected_samples=count)
    assert report["status"] == "failed"
    assert any("shape/step1: missing=1" in error for error in report["errors"])
    assert any("shape/manifest: status is 'running'" in error for error in report["errors"])


def test_forged_validation_metrics_fail_independent_recomputation(tmp_path):
    dataset_root, output_root, count = _completed_fixture(tmp_path)
    paths = criterion_paths(output_root, "color")
    validation = json.loads(paths.validation.read_text(encoding="utf-8"))
    validation["taxonomy_comparisons"]["color"]["nmi"] = 0.123
    paths.validation.write_text(json.dumps(validation), encoding="utf-8")

    report = verify_run(dataset_root, output_root, expected_samples=count)

    assert report["status"] == "failed"
    assert any(
        "color/validation: color.nmi" in error and "recomputed" in error
        for error in report["errors"]
    )


def test_invalid_stage_source_provenance_fails_verification(tmp_path):
    dataset_root, output_root, count = _completed_fixture(tmp_path)
    paths = criterion_paths(output_root, "texture")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["stage_source_sha256s"]["step1"]["pipeline"] = "not-a-digest"
    manifest["stage_source_sha256s"].pop("step3")
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_run(dataset_root, output_root, expected_samples=count)

    assert report["status"] == "failed"
    assert any(
        "texture/manifest: stage_source_sha256s has missing=['step3']" in error
        for error in report["errors"]
    )
    assert any(
        "texture/manifest: stage_source_sha256s['step1'] contains invalid entries"
        in error
        for error in report["errors"]
    )


def test_static_true_label_value_in_prompt_guidance_fails_verification(
    tmp_path, monkeypatch
):
    dataset_root, output_root, count = _completed_fixture(tmp_path)
    monkeypatch.setitem(
        pipeline.CRITERION_CONFIG["texture"],
        "guidance",
        "Use the literal known class texture-0.",
    )

    report = verify_run(dataset_root, output_root, expected_samples=count)

    assert report["status"] == "failed"
    assert report["static_prompt_ground_truth_value_leaks"] == [
        {
            "criterion": "texture",
            "prompt": "step1_prompt.txt",
            "value": "texture-0",
        },
        {
            "criterion": "texture",
            "prompt": "step2a_prompt.txt",
            "value": "texture-0",
        },
    ]
    assert any("static_prompts: ground-truth label values" in e for e in report["errors"])


def test_ground_truth_field_in_model_stage_row_fails_verification(tmp_path):
    dataset_root, output_root, count = _completed_fixture(tmp_path)
    paths = criterion_paths(output_root, "shape")
    rows = [json.loads(line) for line in paths.step1.read_text().splitlines()]
    rows[0]["shape"] = "shape-0"
    _write_jsonl(paths.step1, rows)

    report = verify_run(dataset_root, output_root, expected_samples=count)

    assert report["status"] == "failed"
    assert any(
        "shape/step1: 1 rows contain forbidden label/ground-truth fields" in error
        for error in report["errors"]
    )


def test_unmatched_unknown_sentinel_passes_strict_verification(tmp_path):
    dataset_root, output_root, count = _completed_fixture(tmp_path)
    paths = criterion_paths(output_root, "texture")
    rows = [json.loads(line) for line in paths.step3.read_text().splitlines()]
    rows[0]["class"] = "Unknown"
    rows[0]["step3_match_method"] = "unmatched"
    _write_jsonl(paths.step3, rows)

    records = load_clevr4_records(dataset_root, expected_samples=count)
    validate_criterion(records, "texture", paths, expected_samples=count)
    build_alignment_report(output_root)

    report = verify_run(dataset_root, output_root, expected_samples=count)

    assert report["status"] == "passed"
    assert report["criteria"]["texture"]["unknown_class_count"] == 1
    assert report["criteria"]["texture"]["invalid_class_count"] == 0


def test_non_sentinel_class_outside_taxonomy_fails_verification(tmp_path):
    dataset_root, output_root, count = _completed_fixture(tmp_path)
    paths = criterion_paths(output_root, "shape")
    rows = [json.loads(line) for line in paths.step3.read_text().splitlines()]
    rows[0]["class"] = "not-a-learned-category"
    _write_jsonl(paths.step3, rows)

    report = verify_run(dataset_root, output_root, expected_samples=count)

    assert report["status"] == "failed"
    assert any(
        "shape/step3: 1 classes are outside the consolidated taxonomy" in error
        for error in report["errors"]
    )
