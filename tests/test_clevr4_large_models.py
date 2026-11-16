import json
from pathlib import Path
from types import SimpleNamespace

from partitioning.vlm import clevr4_pipeline as pipeline
from scripts import run_clevr4_large_models as runner


def _records(tmp_path: Path):
    return [
        pipeline.Clevr4Record(
            image_id=f"image-{index}",
            image=str(tmp_path / f"image-{index}.png"),
            split="train",
            texture="metal",
            shape="sphere",
            color="red",
            count="1",
        )
        for index in range(2)
    ]


def _args(tmp_path: Path):
    return SimpleNamespace(
        dataset_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        prompt_dir=tmp_path,
        criteria=list(pipeline.CLEVR4_CRITERIA),
        vlm_model=runner.DEFAULT_INTERNVL_MODEL,
        vlm_revision=runner.DEFAULT_INTERNVL_REVISION,
        llm_model=runner.DEFAULT_LLAMA_MODEL,
        llm_revision=runner.DEFAULT_LLAMA_REVISION,
        device="cpu",
        vlm_batch_size=1,
        llm_batch_size=2,
        internvl_max_tiles=12,
        min_label_frequency=31,
        expected_samples=2,
        seed=42,
    )


def test_caption_stage_loads_large_vlm_once_for_all_criteria(tmp_path, monkeypatch):
    args = _args(tmp_path)
    records = _records(tmp_path)
    manifests = {criterion: {} for criterion in args.criteria}
    loads = []
    calls = []

    def load_vlm(*load_args, **load_kwargs):
        loads.append((load_args, load_kwargs))
        return object(), object()

    def run_step1(records_arg, criterion, paths, *unused_args, **unused_kwargs):
        calls.append(criterion)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.step1.write_text(
            "".join(
                json.dumps({"image_id": record.image_id, "description": criterion})
                + "\n"
                for record in records_arg
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline, "_load_vlm", load_vlm)
    monkeypatch.setattr(pipeline, "run_step1", run_step1)
    monkeypatch.setattr(pipeline, "_torch_cleanup", lambda: None)

    runner.run_caption_stage(args, records, manifests)

    assert len(loads) == 1
    assert calls == list(pipeline.CLEVR4_CRITERIA)
    assert all(manifest["completed_stages"] == ["step1"] for manifest in manifests.values())


def test_large_model_checkpoint_defaults_are_pinned():
    assert len(runner.DEFAULT_INTERNVL_REVISION) == 40
    assert len(runner.DEFAULT_LLAMA_REVISION) == 40
    assert runner.DEFAULT_INTERNVL_MODEL == "OpenGVLab/InternVL3_5-38B"
    assert runner.DEFAULT_LLAMA_MODEL == "meta-llama/Llama-3.3-70B-Instruct"


def test_worker_name_selects_collision_safe_root_manifest(tmp_path):
    args = SimpleNamespace(output_root=tmp_path, worker_name="gpu0-texture_color")
    assert runner._root_manifest_path(args) == (
        tmp_path / "large_model_run_manifest_gpu0-texture_color.json"
    )

    args.worker_name = "../escape"
    try:
        runner._root_manifest_path(args)
    except ValueError as exc:
        assert "worker-name" in str(exc)
    else:
        raise AssertionError("Unsafe worker names must be rejected.")


def test_start_manifest_preserves_caption_stage_source_hashes(tmp_path, monkeypatch):
    args = _args(tmp_path)
    criterion = "texture"
    paths = pipeline.criterion_paths(args.output_root, criterion)
    paths.output_dir.mkdir(parents=True)
    caption_sources = {"pipeline": "a" * 64, "internvl_adapter": "b" * 64}
    paths.manifest.write_text(
        json.dumps(
            {
                "started_at": "caption-start",
                "completed_stages": ["step1"],
                "source_sha256s": caption_sources,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline,
        "_manifest_base",
        lambda *unused_args, **unused_kwargs: {
            "source_sha256": "c" * 64,
            "source_sha256s": {"pipeline": "c" * 64},
        },
    )

    manifest = runner._start_manifest(args, criterion, _records(tmp_path))

    assert manifest["original_started_at"] == "caption-start"
    assert manifest["completed_stages"] == ["step1"]
    assert manifest["stage_source_sha256s"]["step1"] == caption_sources


def test_start_manifest_keeps_a_completed_criterion_immutable(tmp_path, monkeypatch):
    args = _args(tmp_path)
    criterion = "texture"
    paths = pipeline.criterion_paths(args.output_root, criterion)
    paths.output_dir.mkdir(parents=True)
    previous = {
        "status": "complete",
        "completed_stages": list(runner.STAGE_ORDER),
        "source_sha256": "a" * 64,
        "source_sha256s": {"pipeline": "a" * 64},
    }
    paths.manifest.write_text(json.dumps(previous), encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "_manifest_base",
        lambda *unused_args, **unused_kwargs: {
            "status": "running",
            "source_sha256": "b" * 64,
            "source_sha256s": {"pipeline": "b" * 64},
        },
    )

    manifest = runner._start_manifest(args, criterion, _records(tmp_path))

    assert manifest == previous
    assert json.loads(paths.manifest.read_text(encoding="utf-8")) == previous
