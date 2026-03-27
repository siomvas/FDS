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
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claim one-or-more partition jobs from a central sqlite table.")
    parser.add_argument("--claim-db", required=True, type=str, help="Shared sqlite claim DB path.")
    parser.add_argument("--jobs-root", required=True, type=str, help="Shared jobs root on network volume.")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset key (e.g., tiny_imagenet).")
    parser.add_argument("--base-path", required=True, type=str, help="Dataset base path.")
    parser.add_argument("--cache-dir", default="/workspace/caches/fds", type=str, help="Embedding cache dir.")
    parser.add_argument("--seed", default=42, type=int, help="Seed for partition generation.")
    parser.add_argument("--embed-batch-size", default=100, type=int, help="Embedding batch size.")
    parser.add_argument("--num-workers", default=2, type=int, help="Data-loader workers for extraction.")
    parser.add_argument("--min-client-size", default=20, type=int, help="Minimum valid client size.")
    parser.add_argument("--min-discovered-clients", default=2, type=int, help="Lower bound on discovered clients.")
    parser.add_argument("--max-discovered-clients", default=50, type=int, help="Upper bound on discovered clients.")
    parser.add_argument("--max-claims", default=1, type=int, help="Max number of jobs to process before exit.")
    parser.add_argument("--poll-seconds", default=15, type=int, help="Sleep when no pending jobs are available.")
    parser.add_argument("--idle-timeout-seconds", default=300, type=int, help="Exit if no jobs are claimed within this timeout.")
    parser.add_argument("--worker-id", default=None, type=str, help="Stable worker id for claim tracing.")
    return parser.parse_args()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _claim_one(con: sqlite3.Connection, *, worker_id: str, host: str) -> tuple[int, dict[str, Any]] | None:
    con.execute("BEGIN IMMEDIATE")
    row = con.execute(
        "SELECT id, params_json FROM jobs WHERE status = 'pending' ORDER BY id LIMIT 1"
    ).fetchone()
    if row is None:
        con.execute("COMMIT")
        return None
    job_id = int(row[0])
    params = json.loads(str(row[1]))
    updated = con.execute(
        """
        UPDATE jobs
        SET status = 'claimed',
            claim_worker = ?,
            claim_host = ?,
            claim_time = ?,
            error = NULL
        WHERE id = ? AND status = 'pending'
        """,
        (worker_id, host, _utc_now(), job_id),
    ).rowcount
    if updated != 1:
        con.execute("ROLLBACK")
        return None
    con.execute("COMMIT")
    return job_id, params


def _build_partition_command(*, args: argparse.Namespace, params: dict[str, Any], output_path: Path) -> list[str]:
    cmd = [
        "python",
        "scripts/generate_embedding_partitions.py",
        "--dataset",
        str(args.dataset),
        "--base_path",
        str(args.base_path),
        "--output",
        str(output_path),
        "--seed",
        str(int(args.seed)),
        "--extractor",
        str(params["extractor"]),
        "--batch_size",
        str(int(args.embed_batch_size)),
        "--num_workers",
        str(int(args.num_workers)),
        "--tokens",
        str(params["tokens"]),
        "--clustering",
        "hdbscan",
        "--min_cluster_size",
        str(int(params["min_cluster_size"])),
        "--min_samples",
        str(int(params["min_samples"])),
        "--hdbscan_metric",
        str(params["hdbscan_metric"]),
        "--hdbscan_method",
        str(params["hdbscan_method"]),
        "--cache_dir",
        str(args.cache_dir),
        "--valset_type",
        "none",
    ]
    if str(params["tokens"]) != "cls":
        cmd.extend(["--pooling", str(params["pooling"])])
    if bool(params["normalize"]):
        cmd.append("--normalize")
    if str(params["dim_reduction"]) == "umap":
        cmd.extend(
            [
                "--dim_reduction",
                "umap",
                "--dr_dims",
                str(int(params["dr_dims"])),
                "--umap_neighbors",
                str(int(params["umap_neighbors"])),
                "--umap_min_dist",
                str(float(params["umap_min_dist"])),
            ]
        )
    return cmd


def _partition_stats(partition_path: Path) -> tuple[int, int]:
    payload = json.loads(partition_path.read_text(encoding="utf-8"))
    clients = payload.get("clients") or {}
    sizes: list[int] = []
    for value in clients.values():
        train_split = value.get("train") if isinstance(value, dict) else []
        sizes.append(len(train_split or []))
    n_clients = len(sizes)
    min_size = min(sizes) if sizes else 0
    return n_clients, min_size


def _mark_done(
    con: sqlite3.Connection,
    *,
    job_id: int,
    status: str,
    partition_path: Path | None = None,
    n_clients: int | None = None,
    min_client_size_observed: int | None = None,
    partition_time_seconds: float | None = None,
    score_status: str | None = None,
    error: str | None = None,
) -> None:
    con.execute("BEGIN IMMEDIATE")
    con.execute(
        """
        UPDATE jobs
        SET status = ?,
            partition_path = ?,
            n_clients = ?,
            min_client_size_observed = ?,
            partition_time_seconds = ?,
            score_status = COALESCE(?, score_status),
            error = ?
        WHERE id = ?
        """,
        (
            status,
            str(partition_path) if partition_path is not None else None,
            n_clients,
            min_client_size_observed,
            partition_time_seconds,
            score_status,
            error,
            int(job_id),
        ),
    )
    con.execute("COMMIT")


def main() -> int:
    args = _parse_args()
    claim_db = Path(args.claim_db).expanduser().resolve()
    jobs_root = Path(args.jobs_root).expanduser().resolve()
    partitions_dir = jobs_root / "partitions"
    logs_dir = jobs_root / "worker_logs"
    partitions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    host = socket.gethostname()
    con = sqlite3.connect(claim_db, timeout=60, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")

    claimed_count = 0
    idle_deadline = time.monotonic() + float(args.idle_timeout_seconds)
    while claimed_count < int(args.max_claims):
        claimed = _claim_one(con, worker_id=worker_id, host=host)
        if claimed is None:
            if time.monotonic() >= idle_deadline:
                break
            time.sleep(float(args.poll_seconds))
            continue
        job_id, params = claimed
        claimed_count += 1
        output_path = partitions_dir / f"job_{job_id:04d}.json"
        start = time.monotonic()
        try:
            cmd = _build_partition_command(args=args, params=params, output_path=output_path)
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
            elapsed = time.monotonic() - start
            worker_log = logs_dir / f"job_{job_id:04d}.log"
            worker_log.write_text(
                "\n".join(
                    [
                        f"[worker_id] {worker_id}",
                        f"[host] {host}",
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
                _mark_done(
                    con,
                    job_id=job_id,
                    status="partition_failed",
                    partition_time_seconds=elapsed,
                    score_status="skip",
                    error=f"partition command failed (rc={proc.returncode})",
                )
                continue
            if not output_path.exists():
                _mark_done(
                    con,
                    job_id=job_id,
                    status="partition_failed",
                    partition_time_seconds=elapsed,
                    score_status="skip",
                    error=f"partition output missing: {output_path}",
                )
                continue

            n_clients, min_size = _partition_stats(output_path)
            out_of_bounds = not (int(args.min_discovered_clients) <= n_clients <= int(args.max_discovered_clients))
            too_small = int(min_size) < int(args.min_client_size)
            if out_of_bounds or too_small:
                reason_parts: list[str] = []
                if out_of_bounds:
                    reason_parts.append("partition_client_count_out_of_bounds")
                if too_small:
                    reason_parts.append("partition_has_client_below_min_size")
                _mark_done(
                    con,
                    job_id=job_id,
                    status="partition_invalid",
                    partition_path=output_path,
                    n_clients=n_clients,
                    min_client_size_observed=min_size,
                    partition_time_seconds=elapsed,
                    score_status="skip",
                    error=";".join(reason_parts),
                )
            else:
                _mark_done(
                    con,
                    job_id=job_id,
                    status="partition_done",
                    partition_path=output_path,
                    n_clients=n_clients,
                    min_client_size_observed=min_size,
                    partition_time_seconds=elapsed,
                    score_status="pending",
                    error=None,
                )
        except Exception as exc:  # pragma: no cover - operational safety
            elapsed = time.monotonic() - start
            _mark_done(
                con,
                job_id=job_id,
                status="partition_failed",
                partition_time_seconds=elapsed,
                score_status="skip",
                error=f"{type(exc).__name__}: {exc}",
            )

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
