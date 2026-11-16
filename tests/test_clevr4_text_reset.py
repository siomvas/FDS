import json
from pathlib import Path

from partitioning.vlm import clevr4_pipeline as pipeline
from scripts.reset_clevr4_text_stages import (
    ROOT_TEXT_ARTIFACT_NAMES,
    TEXT_ARTIFACT_NAMES,
    prepare_text_rerun,
)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_prepare_text_rerun_archives_only_downstream_artifacts(tmp_path):
    dataset_root = tmp_path / "dataset"
    images = dataset_root / "images"
    images.mkdir(parents=True)
    annotations = {
        f"image-{index}": {
            "texture": f"texture-{index}",
            "shape": f"shape-{index}",
            "color": f"color-{index}",
            "count": index + 1,
            "split": "train",
        }
        for index in range(2)
    }
    (dataset_root / "clevr_4_annots.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    for image_id in annotations:
        (images / f"{image_id}.png").write_bytes(b"fixture")

    output_root = tmp_path / "outputs"
    for criterion in pipeline.CLEVR4_CRITERIA:
        paths = pipeline.criterion_paths(output_root, criterion)
        rows = [{"image_id": image_id} for image_id in annotations]
        _write_jsonl(paths.step1, rows)
        for name in TEXT_ARTIFACT_NAMES:
            path = paths.output_dir / name
            path.write_text("completed\n", encoding="utf-8")
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
                    "source_sha256": "a" * 64,
                    "source_sha256s": {"pipeline": "a" * 64},
                    "best_matching_taxonomy": criterion,
                    "target_is_best_by_nmi": True,
                }
            ),
            encoding="utf-8",
        )
    for name in ROOT_TEXT_ARTIFACT_NAMES:
        (output_root / name).write_text("stale\n", encoding="utf-8")
    texture_paths = pipeline.criterion_paths(output_root, "texture")
    for name in TEXT_ARTIFACT_NAMES:
        (texture_paths.output_dir / name).unlink()
    texture_manifest = json.loads(
        texture_paths.manifest.read_text(encoding="utf-8")
    )
    texture_manifest["status"] = "running"
    texture_manifest["completed_stages"] = []
    texture_paths.manifest.write_text(
        json.dumps(texture_manifest), encoding="utf-8"
    )

    report = prepare_text_rerun(
        dataset_root,
        output_root,
        backup_name="attempt_before_prompt_fix",
        reason="remove class exemplars",
        expected_samples=2,
    )

    assert report["status"] == "prepared"
    assert report["criteria_with_archived_text"] == ["color", "count", "shape"]
    backup = output_root / "attempt_before_prompt_fix"
    for criterion in pipeline.CLEVR4_CRITERIA:
        paths = pipeline.criterion_paths(output_root, criterion)
        assert paths.step1.exists()
        assert not paths.step2a.exists()
        assert (
            backup / paths.output_dir.name / "step2a_result.jsonl"
        ).exists() is (criterion != "texture")
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        assert manifest["status"] == "caption_complete"
        assert manifest["completed_stages"] == ["step1"]
        assert manifest["stage_source_sha256s"] == {
            "step1": {"pipeline": "a" * 64}
        }
        assert "best_matching_taxonomy" not in manifest
    assert (backup / "semantic_alignment_report.json").exists()
    assert (backup / "artifact_verification.json").exists()
    root_manifest = json.loads(
        (output_root / "large_model_run_manifest.json").read_text(encoding="utf-8")
    )
    assert root_manifest["status"] == "caption_complete"
    assert root_manifest["completed_stages"] == ["step1"]
    assert root_manifest["sample_count"] == 2
