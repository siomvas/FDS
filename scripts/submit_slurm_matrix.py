#!/usr/bin/env python3
"""Submit bulk-experiment matrices to SLURM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from research.io import slugify
from scripts.matrix_utils import REPO_ROOT, STAGE_ORDER, load_matrix, select_rows

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a bulk-experiment matrix to SLURM.")
    parser.add_argument("--matrix", type=str, required=True, help="Path to matrix JSONL/JSON file.")
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch. Default is dry-run.")
    parser.add_argument("--stage", type=str, default=None, help="Optional stage filter.")
    parser.add_argument("--resource-class", type=str, default=None, help="Optional resource-class filter.")
    parser.add_argument(
        "--dependency-mode",
        type=str,
        default="afterany",
        choices=["afterany", "afterok"],
        help="Stage-level dependency mode between submitted arrays.",
    )
    return parser.parse_args()


def _slurm_args(slurm: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for key, value in slurm.items():
        if value in (None, ""):
            continue
        argv.append(f"--{key}={value}")
    return argv


def build_submission_plan(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []
    for stage in STAGE_ORDER:
        stage_rows = [row for row in rows if row["stage"] == stage]
        resource_classes = sorted({row["resource_class"] for row in stage_rows})
        for resource_class in resource_classes:
            group = select_rows(stage_rows, resource_class=resource_class)
            if not group:
                continue
            plans.append(
                {
                    "stage": stage,
                    "resource_class": resource_class,
                    "rows": group,
                    "slurm": dict(group[0].get("slurm", {})),
                }
            )
    return plans


def build_sbatch_command(
    *,
    matrix_path: Path,
    plan: Dict[str, Any],
    dependency_job_ids: List[str],
    dependency_mode: str,
) -> List[str]:
    rows = plan["rows"]
    stage = plan["stage"]
    resource_class = plan["resource_class"]
    job_name = slugify(f"{matrix_path.stem}-{stage}-{resource_class}")
    worker = REPO_ROOT / "scripts" / "run_matrix_task.py"
    wrapped = [
        sys.executable,
        str(worker),
        "--matrix",
        str(matrix_path),
        "--stage",
        stage,
        "--resource-class",
        resource_class,
        "--row-index",
        "${SLURM_ARRAY_TASK_ID}",
    ]
    argv = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--array=0-{len(rows) - 1}",
        *_slurm_args(plan["slurm"]),
    ]
    if dependency_job_ids:
        argv.append(f"--dependency={dependency_mode}:{':'.join(dependency_job_ids)}")
    argv.extend(["--wrap", shlex.join(wrapped)])
    return argv


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    matrix_path = Path(args.matrix).resolve()
    rows = select_rows(load_matrix(matrix_path), stage=args.stage, resource_class=args.resource_class)
    if not rows:
        raise ValueError(f"No rows matched matrix filters for {matrix_path}")

    plans = build_submission_plan(rows)
    previous_stage_job_ids: List[str] = []
    current_stage = None

    for plan in plans:
        if plan["stage"] != current_stage:
            dependency_job_ids = previous_stage_job_ids
            previous_stage_job_ids = []
            current_stage = plan["stage"]
        else:
            dependency_job_ids = []

        cmd = build_sbatch_command(
            matrix_path=matrix_path,
            plan=plan,
            dependency_job_ids=dependency_job_ids,
            dependency_mode=args.dependency_mode,
        )
        if not args.submit:
            log.info("%s", shlex.join(cmd))
            previous_stage_job_ids.append(f"<{plan['stage']}:{plan['resource_class']}>")
            continue

        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = proc.stdout.strip()
        log.info("Submitted %s/%s as job %s", plan["stage"], plan["resource_class"], job_id)
        previous_stage_job_ids.append(job_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
