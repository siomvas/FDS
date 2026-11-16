#!/usr/bin/env python3
"""Run or merge an independent CLEVR-4 step-1 shard.

This utility is intended for work stealing after other criterion workers finish.
It never writes the canonical step-1 file during inference; each worker appends
to a private JSONL file, and ``--merge`` performs a strict, atomic merge only
after all shard workers have exited.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

import partitioning.vlm.clevr4_pipeline as pipeline


def shard_records(records, completed_ids: set[str], shard_index: int, num_shards: int):
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    pending = [record for record in records if record.image_id not in completed_ids]
    return pending[shard_index::num_shards]


def merge_shards(
    records,
    canonical_path: Path,
    shard_dir: Path,
    *,
    run_id: str,
) -> int:
    sources = [canonical_path] if canonical_path.exists() else []
    sources.extend(sorted(shard_dir.glob("shard_*.jsonl")))
    if len(sources) < 2:
        raise ValueError(f"Expected a canonical file and shard files under {shard_dir}.")

    by_id: dict[str, dict] = {}
    source_for_id: dict[str, str] = {}
    for source in sources:
        for row in pipeline.read_jsonl(source):
            image_id = str(row.get("image_id", ""))
            if not image_id:
                raise ValueError(f"Missing image_id in {source}.")
            if image_id in by_id:
                raise ValueError(
                    f"Duplicate image_id {image_id} in {source} and {source_for_id[image_id]}."
                )
            by_id[image_id] = row
            source_for_id[image_id] = str(source)

    expected_ids = [record.image_id for record in records]
    missing = [image_id for image_id in expected_ids if image_id not in by_id]
    extras = sorted(set(by_id) - set(expected_ids))
    if missing or extras:
        raise ValueError(
            f"Shard merge is incomplete: missing={len(missing)} extras={len(extras)}; "
            f"first_missing={missing[:5]} first_extras={extras[:5]}"
        )

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=canonical_path.parent, delete=False
    ) as handle:
        for image_id in expected_ids:
            handle.write(json.dumps(by_id[image_id], ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    backup = canonical_path.with_name(f"step1_pre_shard_{run_id}.jsonl")
    if canonical_path.exists() and not backup.exists():
        # Preserve the prior durable prefix, then atomically replace the
        # canonical path.  Moving it to the backup first would create a crash
        # window in which no canonical file exists.
        shutil.copy2(canonical_path, backup)
    temporary.replace(canonical_path)
    return len(by_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--criterion", choices=pipeline.CLEVR4_CRITERIA, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--vlm-model", default=pipeline.DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-revision", default=pipeline.DEFAULT_VLM_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=28 * 28 * 512)
    parser.add_argument("--internvl-max-tiles", type=int, default=12)
    parser.add_argument("--expected-samples", type=int, default=10_531)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = pipeline.load_clevr4_records(
        args.dataset_root,
        expected_samples=args.expected_samples,
    )
    paths = pipeline.criterion_paths(args.output_root.resolve(), args.criterion)
    shard_dir = paths.output_dir / "step1_shards" / args.run_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        count = merge_shards(records, paths.step1, shard_dir, run_id=args.run_id)
        pipeline.atomic_write_json(
            shard_dir / "merge_manifest.json",
            {"status": "complete", "row_count": count, "merged_at": pipeline.utc_now()},
        )
        return 0

    if args.shard_index is None or args.num_shards is None:
        raise SystemExit("--shard-index and --num-shards are required unless --merge is used.")
    completed_ids = pipeline.completed_image_ids(paths.step1)
    assigned = shard_records(records, completed_ids, args.shard_index, args.num_shards)
    shard_path = shard_dir / f"shard_{args.shard_index:02d}.jsonl"
    shard_paths = replace(paths, step1=shard_path)
    pipeline.atomic_write_json(
        shard_dir / f"shard_{args.shard_index:02d}_manifest.json",
        {
            "status": "running",
            "criterion": args.criterion,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "assigned_count": len(assigned),
            "canonical_completed_snapshot_count": len(completed_ids),
            "source_sha256": pipeline.sha256_file(pipeline.__file__),
            "started_at": pipeline.utc_now(),
        },
    )
    model, processor = pipeline._load_vlm(
        args.vlm_model,
        args.vlm_revision,
        args.device,
        args.max_pixels,
        internvl_max_tiles=args.internvl_max_tiles,
    )
    pipeline.run_step1(
        assigned,
        args.criterion,
        shard_paths,
        Path(pipeline.__file__).resolve().parent / "data" / "clevr4",
        model,
        processor,
        device=args.device,
        batch_size=args.batch_size,
    )
    pipeline.atomic_write_json(
        shard_dir / f"shard_{args.shard_index:02d}_manifest.json",
        {
            "status": "complete",
            "criterion": args.criterion,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "assigned_count": len(assigned),
            "written_count": len(pipeline.completed_image_ids(shard_path)),
            "source_sha256": pipeline.sha256_file(pipeline.__file__),
            "completed_at": pipeline.utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
