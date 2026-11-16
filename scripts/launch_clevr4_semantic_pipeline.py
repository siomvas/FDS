#!/usr/bin/env python3
"""Launch one resumable CLEVR-4 semantic-pipeline worker per GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from partitioning.vlm.clevr4_pipeline import (
    CLEVR4_CRITERIA,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_REVISION,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_REVISION,
    build_alignment_report,
)


def _atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-revision", default=DEFAULT_VLM_REVISION)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-revision", default=DEFAULT_LLM_REVISION)
    parser.add_argument("--vlm-batch-size", type=int, default=8)
    parser.add_argument("--llm-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="Zero uses the runner's criterion-aware visual pixel budgets.",
    )
    parser.add_argument("--min-label-frequency", type=int, default=31)
    parser.add_argument("--internvl-max-tiles", type=int, default=12)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=10_531,
        help="Official 10k-v1 archive size (the release contains 10,531 images).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if len(gpu_ids) != len(CLEVR4_CRITERIA):
        raise SystemExit(
            f"Expected {len(CLEVR4_CRITERIA)} GPU ids for {CLEVR4_CRITERIA}, got {gpu_ids}."
        )

    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.expanduser().resolve()
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    processes: list[tuple[str, str, subprocess.Popen[bytes], object, list[str]]] = []
    for criterion, gpu_id in zip(CLEVR4_CRITERIA, gpu_ids):
        command = [
            sys.executable,
            "-m",
            "partitioning.vlm.clevr4_pipeline",
            "--dataset-root",
            str(args.dataset_root.expanduser().resolve()),
            "--output-root",
            str(output_root),
            "--criterion",
            criterion,
            "--vlm-model",
            args.vlm_model,
            "--vlm-revision",
            args.vlm_revision,
            "--llm-model",
            args.llm_model,
            "--llm-revision",
            args.llm_revision,
            "--vlm-batch-size",
            str(args.vlm_batch_size),
            "--llm-batch-size",
            str(args.llm_batch_size),
            "--max-pixels",
            str(args.max_pixels),
            "--min-label-frequency",
            str(args.min_label_frequency),
            "--internvl-max-tiles",
            str(args.internvl_max_tiles),
            "--expected-samples",
            str(args.expected_samples),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["PYTHONUNBUFFERED"] = "1"
        log_handle = (log_root / f"{criterion}.log").open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((criterion, gpu_id, process, log_handle, command))

    manifest_path = output_root / "launcher_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "status": "running",
            "started_at": started_at,
            "workers": [
                {
                    "criterion": criterion,
                    "physical_gpu": gpu_id,
                    "pid": process.pid,
                    "command": command,
                }
                for criterion, gpu_id, process, _, command in processes
            ],
        },
    )

    return_codes: dict[str, int] = {}
    for criterion, _, process, log_handle, _ in processes:
        return_codes[criterion] = process.wait()
        log_handle.close()
    failed = {criterion: code for criterion, code in return_codes.items() if code != 0}
    completed_at = datetime.now(timezone.utc).isoformat()
    if failed:
        _atomic_json(
            manifest_path,
            {
                "status": "failed",
                "started_at": started_at,
                "completed_at": completed_at,
                "return_codes": return_codes,
            },
        )
        return 1

    report = build_alignment_report(output_root)
    _atomic_json(
        manifest_path,
        {
            "status": "complete",
            "started_at": started_at,
            "completed_at": completed_at,
            "return_codes": return_codes,
            "alignment_passed": report["all_criteria_best_match_their_true_taxonomy"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
