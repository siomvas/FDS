#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PartitionJob:
    extractor: str
    tokens: str
    pooling: str | None
    normalize: bool
    dim_reduction: str
    dr_dims: int | None
    umap_neighbors: int | None
    umap_min_dist: float | None
    min_cluster_size: int
    min_samples: int
    hdbscan_method: str
    hdbscan_metric: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize central claim DB for partition-only workers.")
    parser.add_argument("--db-path", required=True, type=str, help="Output sqlite path.")
    parser.add_argument("--dataset", default="tiny_imagenet", type=str, help="Dataset identifier.")
    parser.add_argument("--n-jobs", default=100, type=int, help="Number of jobs to enqueue.")
    parser.add_argument("--seed", default=42, type=int, help="Shuffle seed.")
    parser.add_argument("--min-client-size", default=20, type=int, help="Minimum client size used for min_cluster_size multipliers.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle generated combinations before truncating to n-jobs.")
    return parser.parse_args()


def _all_jobs(min_client_size: int) -> list[PartitionJob]:
    extractors = ["dinov2-base", "clip-base-patch16", "mae-base"]
    tokens = ["cls", "all_last"]
    pooling_non_cls = ["mean", "max"]
    normalizations = [False, True]
    dim_reductions = ["none", "umap"]
    dr_dims_choices = [32, 128]
    umap_neighbors_choices = [15]
    umap_min_dist_choices = [0.1]
    min_cluster_size_choices = [min_client_size * m for m in (1, 2, 4)]
    min_samples_choices = [1, 5, 10]
    hdbscan_methods = ["eom", "leaf"]

    jobs: list[PartitionJob] = []
    for extractor in extractors:
        for token in tokens:
            pooling_choices = [None] if token == "cls" else pooling_non_cls
            for pooling in pooling_choices:
                for normalize in normalizations:
                    metric_choices = ["euclidean"] if normalize else ["euclidean", "cosine"]
                    for dim_reduction in dim_reductions:
                        if dim_reduction == "none":
                            reduction_configs = [(None, None, None)]
                        else:
                            reduction_configs = [
                                (dr_dims, neighbors, min_dist)
                                for dr_dims in dr_dims_choices
                                for neighbors in umap_neighbors_choices
                                for min_dist in umap_min_dist_choices
                            ]
                        for dr_dims, neighbors, min_dist in reduction_configs:
                            for min_cluster_size in min_cluster_size_choices:
                                for min_samples in min_samples_choices:
                                    for hdbscan_method in hdbscan_methods:
                                        for hdbscan_metric in metric_choices:
                                            jobs.append(
                                                PartitionJob(
                                                    extractor=extractor,
                                                    tokens=token,
                                                    pooling=pooling,
                                                    normalize=normalize,
                                                    dim_reduction=dim_reduction,
                                                    dr_dims=dr_dims,
                                                    umap_neighbors=neighbors,
                                                    umap_min_dist=min_dist,
                                                    min_cluster_size=min_cluster_size,
                                                    min_samples=min_samples,
                                                    hdbscan_method=hdbscan_method,
                                                    hdbscan_metric=hdbscan_metric,
                                                )
                                            )
    return jobs


def _job_hash(job: PartitionJob) -> str:
    payload = json.dumps(asdict(job), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _create_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT NOT NULL,
            job_hash TEXT NOT NULL UNIQUE,
            params_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            claim_worker TEXT,
            claim_host TEXT,
            claim_time TEXT,
            partition_path TEXT,
            n_clients INTEGER,
            min_client_size_observed INTEGER,
            partition_time_seconds REAL,
            score_status TEXT NOT NULL DEFAULT 'pending',
            score_claim_worker TEXT,
            score_claim_time TEXT,
            score_output_dir TEXT,
            mmd_features_mean REAL,
            score_time_seconds REAL,
            error TEXT
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_score_status ON jobs(score_status)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_dataset ON jobs(dataset)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    con.commit()


def main() -> int:
    args = _parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = _all_jobs(min_client_size=int(args.min_client_size))
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(jobs)
    selected = jobs[: int(args.n_jobs)]
    if len(selected) < int(args.n_jobs):
        raise SystemExit(f"Requested {args.n_jobs} jobs but only {len(selected)} combinations available.")

    con = sqlite3.connect(db_path)
    try:
        _create_schema(con)
        con.execute("DELETE FROM jobs")
        con.execute("DELETE FROM metadata")
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metadata = {
            "dataset": str(args.dataset),
            "n_jobs": str(len(selected)),
            "seed": str(int(args.seed)),
            "min_client_size": str(int(args.min_client_size)),
            "shuffle": "1" if args.shuffle else "0",
            "created_at": created_at,
        }
        con.executemany(
            "INSERT INTO metadata(key, value) VALUES(?, ?)",
            [(k, v) for k, v in metadata.items()],
        )
        rows: list[tuple[str, str, str]] = []
        for job in selected:
            params_json = json.dumps(asdict(job), sort_keys=True, separators=(",", ":"))
            rows.append((str(args.dataset), _job_hash(job), params_json))
        con.executemany(
            "INSERT INTO jobs(dataset, job_hash, params_json) VALUES(?, ?, ?)",
            rows,
        )
        con.commit()
        print(f"Initialized claim DB: {db_path}")
        print(f"Dataset: {args.dataset}")
        print(f"Jobs inserted: {len(rows)}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
