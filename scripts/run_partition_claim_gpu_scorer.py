#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score claimed partition artifacts on a single GPU pod.")
    parser.add_argument("--claim-db", required=True, type=str, help="Shared sqlite claim DB path.")
    parser.add_argument("--jobs-root", required=True, type=str, help="Shared jobs root.")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset key.")
    parser.add_argument("--base-path", required=True, type=str, help="Dataset base path.")
    parser.add_argument("--cache-dir", default="/workspace/caches/fds", type=str, help="Feature cache directory.")
    parser.add_argument("--feature-model", default="inception_v3", type=str, help="Scorer feature model.")
    parser.add_argument("--batch-size", default=128, type=int, help="Scorer batch size.")
    parser.add_argument("--subsample", default=0, type=int, help="Samples per client for metrics (0 = all).")
    parser.add_argument("--poll-seconds", default=30, type=int, help="Polling interval when no scoreable jobs exist.")
    parser.add_argument("--idle-timeout-seconds", default=1800, type=int, help="Exit if no scoreable jobs for this long.")
    parser.add_argument("--worker-id", default=None, type=str, help="Stable scorer worker id.")
    return parser.parse_args()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _claim_score_job(con: sqlite3.Connection, *, worker_id: str) -> tuple[int, str] | None:
    con.execute("BEGIN IMMEDIATE")
    row = con.execute(
        """
        SELECT id, partition_path
        FROM jobs
        WHERE status = 'partition_done' AND score_status = 'pending'
        ORDER BY id
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        con.execute("COMMIT")
        return None
    job_id = int(row[0])
    partition_path = str(row[1])
    updated = con.execute(
        """
        UPDATE jobs
        SET score_status = 'claimed',
            score_claim_worker = ?,
            score_claim_time = ?
        WHERE id = ? AND status = 'partition_done' AND score_status = 'pending'
        """,
        (worker_id, _utc_now(), job_id),
    ).rowcount
    if updated != 1:
        con.execute("ROLLBACK")
        return None
    con.execute("COMMIT")
    return job_id, partition_path


def _mark_score(
    con: sqlite3.Connection,
    *,
    job_id: int,
    score_status: str,
    score_output_dir: Path | None = None,
    mmd: float | None = None,
    score_time_seconds: float | None = None,
    error: str | None = None,
) -> None:
    con.execute("BEGIN IMMEDIATE")
    con.execute(
        """
        UPDATE jobs
        SET score_status = ?,
            score_output_dir = ?,
            mmd_features_mean = ?,
            score_time_seconds = ?,
            error = COALESCE(?, error)
        WHERE id = ?
        """,
        (
            score_status,
            str(score_output_dir) if score_output_dir is not None else None,
            mmd,
            score_time_seconds,
            error,
            int(job_id),
        ),
    )
    if score_status == "scored":
        con.execute("UPDATE jobs SET status = 'scored' WHERE id = ?", (int(job_id),))
    elif score_status == "score_failed":
        con.execute("UPDATE jobs SET status = 'score_failed' WHERE id = ?", (int(job_id),))
    con.execute("COMMIT")


def _extract_mmd_from_output(score_output_dir: Path) -> float | None:
    result_files = sorted(score_output_dir.rglob("result.json"), key=lambda p: p.stat().st_mtime)
    if not result_files:
        return None
    payload = json.loads(result_files[-1].read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    value = metrics.get("mmd_features_mean")
    return float(value) if value is not None else None


def main() -> int:
    args = _parse_args()
    claim_db = Path(args.claim_db).expanduser().resolve()
    jobs_root = Path(args.jobs_root).expanduser().resolve()
    score_root = jobs_root / "scores"
    score_logs = jobs_root / "score_logs"
    score_root.mkdir(parents=True, exist_ok=True)
    score_logs.mkdir(parents=True, exist_ok=True)

    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    con = sqlite3.connect(claim_db, timeout=60, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")

    idle_deadline = time.monotonic() + float(args.idle_timeout_seconds)
    while True:
        claimed = _claim_score_job(con, worker_id=worker_id)
        if claimed is None:
            if time.monotonic() >= idle_deadline:
                break
            time.sleep(float(args.poll_seconds))
            continue
        idle_deadline = time.monotonic() + float(args.idle_timeout_seconds)
        job_id, partition_path = claimed
        output_dir = score_root / f"job_{job_id:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        try:
            cmd = [
                "python",
                "-m",
                "evaluation.evaluate_partitions",
                "--partition",
                str(partition_path),
                "--dataset",
                str(args.dataset),
                "--base_path",
                str(args.base_path),
                "--output",
                str(output_dir),
                "--feature_model",
                str(args.feature_model),
                "--batch_size",
                str(int(args.batch_size)),
                "--subsample",
                str(int(args.subsample)),
                "--cache_dir",
                str(args.cache_dir),
                "--feature_only",
                "--force",
            ]
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
            elapsed = time.monotonic() - start
            log_path = score_logs / f"job_{job_id:04d}.log"
            log_path.write_text(
                "\n".join(
                    [
                        f"[worker_id] {worker_id}",
                        f"[started_at] {_utc_now()}",
                        f"[elapsed_seconds] {elapsed:.3f}",
                        f"[command] {' '.join(cmd)}",
                        f"[returncode] {proc.returncode}",
                        "",
                        "[stdout]",
                        proc.stdout or "",
                        "",
                        "[stderr]",
                        proc.stderr or "",
                    ]
                ),
                encoding="utf-8",
            )
            if proc.returncode != 0:
                _mark_score(
                    con,
                    job_id=job_id,
                    score_status="score_failed",
                    score_output_dir=output_dir,
                    score_time_seconds=elapsed,
                    error=f"evaluate_partitions failed (rc={proc.returncode})",
                )
                continue
            mmd = _extract_mmd_from_output(output_dir)
            if mmd is None:
                _mark_score(
                    con,
                    job_id=job_id,
                    score_status="score_failed",
                    score_output_dir=output_dir,
                    score_time_seconds=elapsed,
                    error="mmd_features_mean missing from result.json",
                )
                continue
            _mark_score(
                con,
                job_id=job_id,
                score_status="scored",
                score_output_dir=output_dir,
                mmd=mmd,
                score_time_seconds=elapsed,
                error=None,
            )
        except Exception as exc:  # pragma: no cover - operational safety
            elapsed = time.monotonic() - start
            _mark_score(
                con,
                job_id=job_id,
                score_status="score_failed",
                score_output_dir=output_dir,
                score_time_seconds=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
