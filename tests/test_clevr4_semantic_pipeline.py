import json
from pathlib import Path

import pytest

from partitioning.vlm.clevr4_pipeline import (
    CLEVR4_CRITERIA,
    CRITERION_CONFIG,
    append_jsonl,
    build_alignment_report,
    clean_step2a_label,
    criterion_paths,
    load_clevr4_records,
    match_category,
    parse_categories,
    resolve_llm_device_map,
    run_step2b,
    validate_criterion,
)


def test_large_llm_device_map_balances_multiple_visible_gpus():
    assert resolve_llm_device_map("cuda:0", 2) == "balanced"
    assert resolve_llm_device_map("cuda:0", 1) is None
    assert resolve_llm_device_map("auto", 0) == "auto"
    assert resolve_llm_device_map("cpu", 2) is None


def test_texture_guidance_prioritizes_pattern_structure_over_generic_finish():
    guidance = str(CRITERION_CONFIG["texture"]["guidance"]).lower()
    assert "repeated motif" in guidance
    assert "generic finish" in guidance
    assert "object color" in guidance


def test_step2a_prompt_has_no_class_exemplar():
    prompt_path = (
        Path(__file__).parents[1]
        / "partitioning"
        / "vlm"
        / "data"
        / "clevr4"
        / "step2a_prompt.txt"
    )
    assert "EXEMPLAR" not in prompt_path.read_text(encoding="utf-8")
    assert all("exemplar" not in config for config in CRITERION_CONFIG.values())


def test_jsonl_append_rolls_back_a_partial_disk_write(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, [{"image_id": "existing", "description": "complete"}])
    original = path.read_bytes()
    real_write = __import__("os").write
    calls = 0

    def fail_after_partial_write(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, payload[:17])
        raise OSError("simulated disk quota exhaustion")

    monkeypatch.setattr("partitioning.vlm.clevr4_pipeline.os.write", fail_after_partial_write)
    with pytest.raises(OSError, match="disk quota"):
        append_jsonl(path, [{"image_id": "new", "description": "not durable"}])

    assert path.read_bytes() == original


def _make_dataset(root: Path, annotations: dict) -> Path:
    dataset = root / "clevr_4_10k_v1"
    images = dataset / "images"
    images.mkdir(parents=True)
    (dataset / "clevr_4_annots.json").write_text(json.dumps(annotations), encoding="utf-8")
    for image_id in annotations:
        (images / f"{image_id}.png").write_bytes(b"not-opened-by-loader")
    return root


def test_loader_discovers_nested_official_layout_and_true_criteria(tmp_path):
    annotations = {
        "CLEVR_new_000001": {
            "texture": "metal",
            "shape": "sphere",
            "color": "red",
            "count": 3,
            "split": "train",
        },
        "CLEVR_new_000002": {
            "texture": "rubber",
            "shape": "cube",
            "color": "blue",
            "count": 8,
            "split": "val",
        },
    }
    root = _make_dataset(tmp_path, annotations)
    records = load_clevr4_records(root, expected_samples=2)
    assert [record.image_id for record in records] == sorted(annotations)
    assert records[0].texture == "metal"
    assert records[0].shape == "sphere"
    assert records[0].color == "red"
    assert records[0].count == "3"


def test_loader_split_filter_is_strict(tmp_path):
    annotations = {
        f"image_{index}": {
            "texture": "metal",
            "shape": "sphere",
            "color": "red",
            "count": 3,
            "split": split,
        }
        for index, split in enumerate(("train", "val", "test"))
    }
    root = _make_dataset(tmp_path, annotations)
    records = load_clevr4_records(root, splits=["train", "val"], expected_samples=2)
    assert {record.split for record in records} == {"train", "val"}


def test_step2b_parser_requires_exact_unique_category_count():
    response = "\n".join(f"Answer {index}: category-{index}" for index in range(1, 11))
    assert len(parse_categories(response)) == 10
    with pytest.raises(ValueError, match="Expected 10 unique"):
        parse_categories(response + "\nAnswer 11: extra")


def test_response_cleaning_and_category_matching():
    assert clean_step2a_label('Assistant: "Metal."') == "metal"
    categories = ["metal", "rubber", "zigzag"]
    assert match_category("Metal", categories) == ("metal", "exact")
    assert match_category("The category is rubber.", categories)[0] == "rubber"


def test_step2b_can_expand_fewer_observed_labels_into_requested_taxonomy(
    tmp_path, monkeypatch
):
    """Collapsed VLM labels must not prevent a fixed-size semantic taxonomy."""

    response = "\n".join(f"Answer {index}: count-{index}" for index in range(1, 11))
    paths = criterion_paths(tmp_path, "count")
    paths.output_dir.mkdir(parents=True)
    rows = [
        {"image_id": f"image-{index}", "label": f"observed-{index % 5}"}
        for index in range(160)
    ]
    paths.step2a.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        "partitioning.vlm.clevr4_pipeline._generate_text",
        lambda *args, **kwargs: [response],
    )
    prompt_dir = Path(__file__).parents[1] / "partitioning" / "vlm" / "data" / "clevr4"
    categories = run_step2b(
        "count",
        paths,
        prompt_dir,
        model=None,
        tokenizer=None,
        device="cpu",
        min_label_frequency=31,
    )
    assert categories[-1] == "count-10"


def test_step2b_grounds_fixed_taxonomy_with_infrequent_observed_labels(
    tmp_path, monkeypatch
):
    response = "\n".join(f"Answer {index}: count-{index}" for index in range(1, 11))
    paths = criterion_paths(tmp_path, "count")
    paths.output_dir.mkdir(parents=True)
    observed_counts = {
        **{f"frequent-{index}": 40 + index for index in range(9)},
        "infrequent-tenth": 21,
        "singleton-eleventh": 1,
    }
    rows = []
    image_index = 0
    for label, count in observed_counts.items():
        for _ in range(count):
            rows.append({"image_id": f"image-{image_index}", "label": label})
            image_index += 1
    paths.step2a.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    prompts = []

    def generate(*args, **kwargs):
        prompts.extend(args[2])
        return [response]

    monkeypatch.setattr(
        "partitioning.vlm.clevr4_pipeline._generate_text",
        generate,
    )
    prompt_dir = Path(__file__).parents[1] / "partitioning" / "vlm" / "data" / "clevr4"
    run_step2b(
        "count",
        paths,
        prompt_dir,
        model=None,
        tokenizer=None,
        device="cpu",
        min_label_frequency=31,
    )

    assert '"infrequent-tenth": 21' in prompts[0]
    assert "singleton-eleventh" not in prompts[0]
    metadata = json.loads(paths.step2b_metadata.read_text(encoding="utf-8"))
    assert metadata["retained_below_frequency_threshold"] == {
        "infrequent-tenth": 21
    }


def test_step2b_does_not_replace_taxonomy_after_step3_has_started(tmp_path):
    paths = criterion_paths(tmp_path, "texture")
    paths.output_dir.mkdir(parents=True)
    paths.step2b.write_text("invalid taxonomy\n", encoding="utf-8")
    paths.step3.write_text(
        json.dumps({"image_id": "image-1", "class": "existing-category"}) + "\n",
        encoding="utf-8",
    )
    prompt_dir = Path(__file__).parents[1] / "partitioning" / "vlm" / "data" / "clevr4"

    with pytest.raises(RuntimeError, match="Refusing to regenerate the taxonomy"):
        run_step2b(
            "texture",
            paths,
            prompt_dir,
            model=None,
            tokenizer=None,
            device="cpu",
            min_label_frequency=31,
        )


def test_validation_identifies_each_true_taxonomy_as_best_match(tmp_path):
    annotations = {}
    values = ("a", "b")
    index = 0
    for texture in values:
        for shape in values:
            for color in values:
                for count in (1, 2):
                    image_id = f"image_{index:03d}"
                    annotations[image_id] = {
                        "texture": f"texture-{texture}",
                        "shape": f"shape-{shape}",
                        "color": f"color-{color}",
                        "count": count,
                        "split": "train",
                    }
                    index += 1
    dataset_root = _make_dataset(tmp_path / "data", annotations)
    records = load_clevr4_records(dataset_root, expected_samples=len(annotations))
    output_root = tmp_path / "outputs"
    for criterion in CLEVR4_CRITERIA:
        paths = criterion_paths(output_root, criterion)
        paths.output_dir.mkdir(parents=True)
        rows = [
            {
                "image_id": record.image_id,
                "class": str(getattr(record, criterion)),
                "step3_match_method": "exact",
            }
            for record in records
        ]
        paths.step3.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        report = validate_criterion(
            records,
            criterion,
            paths,
            expected_samples=len(records),
        )
        assert report["best_matching_taxonomy"] == criterion
        assert report["target_is_best_by_nmi"] is True
        assert report["taxonomy_comparisons"][criterion]["nmi"] == pytest.approx(1.0)
        assert report["taxonomy_comparisons"][criterion][
            "hungarian_predicted_to_true"
        ] == {value: value for value in report["true_categories"]}

    combined = build_alignment_report(output_root)
    assert combined["all_criteria_best_match_their_true_taxonomy"] is True
