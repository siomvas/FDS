import json
from pathlib import Path

import pytest

from partitioning.vlm.clevr4_pipeline import Clevr4Record
from scripts.run_clevr4_step1_shard import merge_shards, shard_records


def _records(count=12):
    return [
        Clevr4Record(
            image_id=f"image-{index:02d}",
            image=f"/images/image-{index:02d}.png",
            split="train",
            texture="texture",
            shape="shape",
            color="color",
            count="1",
        )
        for index in range(count)
    ]


def _write_rows(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"image_id": record.image_id, "description": record.image_id}) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_shards_partition_only_pending_records_without_overlap():
    records = _records()
    completed = {"image-00", "image-01"}
    shards = [shard_records(records, completed, index, 3) for index in range(3)]
    flattened = [record.image_id for shard in shards for record in shard]
    assert len(flattened) == len(set(flattened)) == 10
    assert set(flattened).isdisjoint(completed)


def test_merge_is_strict_atomic_and_canonical(tmp_path):
    records = _records(8)
    canonical = tmp_path / "step1_result.jsonl"
    shard_dir = tmp_path / "shards"
    _write_rows(canonical, records[:2])
    _write_rows(shard_dir / "shard_00.jsonl", records[2::2])
    _write_rows(shard_dir / "shard_01.jsonl", records[3::2])
    assert merge_shards(records, canonical, shard_dir, run_id="test") == 8
    merged = [json.loads(line) for line in canonical.read_text().splitlines()]
    assert [row["image_id"] for row in merged] == [record.image_id for record in records]
    backup = tmp_path / "step1_pre_shard_test.jsonl"
    assert [
        json.loads(line)["image_id"] for line in backup.read_text().splitlines()
    ] == [record.image_id for record in records[:2]]


def test_merge_rejects_duplicates(tmp_path):
    records = _records(3)
    canonical = tmp_path / "step1_result.jsonl"
    shard_dir = tmp_path / "shards"
    _write_rows(canonical, records[:2])
    _write_rows(shard_dir / "shard_00.jsonl", records[1:])
    with pytest.raises(ValueError, match="Duplicate image_id"):
        merge_shards(records, canonical, shard_dir, run_id="test")


def test_merge_rejects_incomplete_coverage(tmp_path):
    records = _records(4)
    canonical = tmp_path / "step1_result.jsonl"
    shard_dir = tmp_path / "shards"
    _write_rows(canonical, records[:1])
    _write_rows(shard_dir / "shard_00.jsonl", records[1:3])
    with pytest.raises(ValueError, match="missing=1"):
        merge_shards(records, canonical, shard_dir, run_id="test")
