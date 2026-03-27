#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runpod_utils import (  # noqa: E402
    RunpodClient,
    RunpodPodConfig,
    RunpodPodConnection,
    archive_repo_snapshot,
    pod_ssh_connection,
    run_ssh,
    upload_file,
    wait_for_pod_ssh,
)

log = logging.getLogger(__name__)

CPU_IMAGE = "runpod/base:1.0.2-ubuntu2404"
GPU_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"


@dataclass
class PodLaunchRecord:
    pod_id: str
    name: str
    role: str
    public_ip: str | None = None
    ssh_port: int | None = None
    status: str = "created"
    launch_error: str | None = None


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch CPU claim workers + one GPU scorer on Runpod.")
    parser.add_argument("--api-key", required=True, type=str, help="Runpod API key.")
    parser.add_argument("--network-volume-id", required=True, type=str, help="Runpod network volume id.")
    parser.add_argument("--dataset", default="tiny_imagenet", type=str, help="Dataset key.")
    parser.add_argument("--base-path", default="/workspace/datasets/raw", type=str, help="Dataset base path on pods.")
    parser.add_argument("--cpu-pods", default=100, type=int, help="Number of CPU worker pods.")
    parser.add_argument("--n-jobs", default=100, type=int, help="Number of partition jobs to enqueue.")
    parser.add_argument("--cpu-flavor-id", default="cpu5c", type=str, help="CPU flavor id.")
    parser.add_argument(
        "--cpu-name-prefix",
        default=None,
        type=str,
        help="Prefix for CPU pod names. Defaults to fds-partition-<cpu-flavor-id>-",
    )
    parser.add_argument("--cpu-vcpus", default=4, type=int, help="vCPU count per CPU pod.")
    parser.add_argument("--gpu-type-id", default="NVIDIA L40S", type=str, help="GPU type id/display name.")
    parser.add_argument("--container-disk-gb", default=50, type=int, help="Container disk size.")
    parser.add_argument("--launch-delay-seconds", default=5, type=int, help="Delay between pod create attempts.")
    parser.add_argument("--launch-passes", default=6, type=int, help="Number of retry passes for CPU pod creation.")
    parser.add_argument("--ssh-key-path", default="~/.ssh/id_ed25519", type=str, help="SSH key path.")
    parser.add_argument("--startup-timeout-seconds", default=1800, type=int, help="Per-pod startup timeout.")
    parser.add_argument("--poll-interval-seconds", default=10, type=int, help="SSH readiness poll interval.")
    parser.add_argument("--embed-batch-size", default=100, type=int, help="Partition extraction batch size.")
    parser.add_argument("--num-workers", default=2, type=int, help="Partition extraction dataloader workers.")
    parser.add_argument("--scorer-model", default="inception_v3", type=str, help="Scorer feature model for GPU stage.")
    parser.add_argument("--metric-batch-size", default=128, type=int, help="Scoring batch size.")
    parser.add_argument("--run-stamp", default=None, type=str, help="Run stamp override.")
    parser.add_argument("--log-level", default="INFO", type=str, help="Logging level.")
    return parser.parse_args()


def _remote_launch_worker_cmd(*, run_root: str, dataset: str, base_path: str, worker_id: str, embed_batch_size: int, num_workers: int) -> str:
    return (
        "set -euo pipefail;"
        f" RUN_ROOT={json.dumps(run_root)};"
        " REPO_DIR=\"$RUN_ROOT/repo\";"
        " CLAIM_DB=\"$RUN_ROOT/claims.sqlite\";"
        " VENV=/root/.venvs/fds;"
        " mkdir -p \"$RUN_ROOT/pod_logs/" + worker_id + "\";"
        " test -d \"$REPO_DIR\";"
        " test -f \"$CLAIM_DB\";"
        " if [[ ! -x \"$VENV/bin/python\" ]]; then python3 -m venv \"$VENV\"; fi;"
        " source \"$VENV/bin/activate\";"
        " python -m pip install --upgrade pip >/dev/null 2>&1;"
        " python -m pip install -e \"$REPO_DIR[partitioning,embedding,evaluation]\" >/dev/null 2>&1;"
        " export PYTHONUNBUFFERED=1;"
        " export FDS_CACHE_DIR=/workspace/caches/fds;"
        " cd \"$REPO_DIR\";"
        " nohup \"$VENV/bin/python\" scripts/run_partition_claim_worker.py"
        " --claim-db \"$CLAIM_DB\""
        f" --jobs-root \"$RUN_ROOT\" --dataset {dataset} --base-path {base_path}"
        f" --embed-batch-size {int(embed_batch_size)} --num-workers {int(num_workers)}"
        f" --worker-id {worker_id} --max-claims 1"
        f" > \"$RUN_ROOT/pod_logs/{worker_id}/worker.log\" 2>&1 < /dev/null &"
        f" echo $! > \"$RUN_ROOT/pod_logs/{worker_id}/pid\""
    )


def _remote_launch_scorer_cmd(*, run_root: str, dataset: str, base_path: str, worker_id: str, scorer_model: str, metric_batch_size: int) -> str:
    return (
        "set -euo pipefail;"
        f" RUN_ROOT={json.dumps(run_root)};"
        " REPO_DIR=\"$RUN_ROOT/repo\";"
        " CLAIM_DB=\"$RUN_ROOT/claims.sqlite\";"
        " VENV=/root/.venvs/fds;"
        " mkdir -p \"$RUN_ROOT/pod_logs/" + worker_id + "\";"
        " test -d \"$REPO_DIR\";"
        " test -f \"$CLAIM_DB\";"
        " if [[ ! -x \"$VENV/bin/python\" ]]; then python3 -m venv \"$VENV\"; fi;"
        " source \"$VENV/bin/activate\";"
        " python -m pip install --upgrade pip >/dev/null 2>&1;"
        " python -m pip install -e \"$REPO_DIR[partitioning,embedding,evaluation]\" >/dev/null 2>&1;"
        " export PYTHONUNBUFFERED=1;"
        " export FDS_CACHE_DIR=/workspace/caches/fds;"
        " cd \"$REPO_DIR\";"
        " nohup \"$VENV/bin/python\" scripts/run_partition_claim_gpu_scorer.py"
        " --claim-db \"$CLAIM_DB\""
        f" --jobs-root \"$RUN_ROOT\" --dataset {dataset} --base-path {base_path}"
        f" --feature-model {scorer_model} --batch-size {int(metric_batch_size)}"
        f" --worker-id {worker_id} --idle-timeout-seconds 14400"
        f" > \"$RUN_ROOT/pod_logs/{worker_id}/scorer.log\" 2>&1 < /dev/null &"
        f" echo $! > \"$RUN_ROOT/pod_logs/{worker_id}/pid\""
    )


def _init_claim_db(local_claim_db: Path, *, dataset: str, n_jobs: int) -> None:
    cmd = [
        "python3",
        str(REPO_ROOT / "scripts" / "init_partition_claim_db.py"),
        "--db-path",
        str(local_claim_db),
        "--dataset",
        str(dataset),
        "--n-jobs",
        str(int(n_jobs)),
        "--seed",
        "42",
        "--shuffle",
    ]
    subprocess.run(cmd, check=True, text=True)


def _create_pod(client: RunpodClient, cfg: RunpodPodConfig) -> dict[str, Any]:
    return client.create_pod(cfg)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    stamp = args.run_stamp or _timestamp()
    local_run_root = (REPO_ROOT / "local_artifacts" / "runpod" / "partition_claim_rollout" / stamp).resolve()
    local_run_root.mkdir(parents=True, exist_ok=True)
    key_path = Path(args.ssh_key_path).expanduser().resolve()
    client = RunpodClient(args.api_key)

    run_root = f"/workspace/runs/partition_claim_{stamp}"
    repo_archive = archive_repo_snapshot(REPO_ROOT, output_path=local_run_root / "repo_snapshot.tar.gz")
    claim_db = local_run_root / "claims.sqlite"
    _init_claim_db(claim_db, dataset=str(args.dataset), n_jobs=int(args.n_jobs))

    records: list[PodLaunchRecord] = []
    cpu_name_prefix = str(args.cpu_name_prefix) if args.cpu_name_prefix else f"fds-partition-{args.cpu_flavor_id}-"
    target_names = [f"{cpu_name_prefix}{i:03d}" for i in range(1, int(args.cpu_pods) + 1)]
    created_ids: set[str] = set()

    for attempt in range(1, int(args.launch_passes) + 1):
        for name in target_names:
            if any(rec.name == name for rec in records):
                continue
            cfg = RunpodPodConfig(
                name=name,
                image_name=CPU_IMAGE,
                network_volume_id=str(args.network_volume_id),
                compute_type="CPU",
                cpu_flavor_ids=(str(args.cpu_flavor_id),),
                vcpu_count=int(args.cpu_vcpus),
                container_disk_in_gb=int(args.container_disk_gb),
                ports=("22/tcp",),
            )
            try:
                pod = _create_pod(client, cfg)
                pod_id = str(pod["id"])
                created_ids.add(pod_id)
                records.append(PodLaunchRecord(pod_id=pod_id, name=name, role="cpu_worker", status="created"))
                log.info("Created CPU pod %s (%s) [%s/%s]", name, pod_id, len(records), args.cpu_pods)
            except Exception as exc:
                log.warning("Create failed for %s on pass %s: %s", name, attempt, exc)
            time.sleep(float(args.launch_delay_seconds))
        if len(records) >= int(args.cpu_pods):
            break
        log.info("CPU create pass %s complete: %s/%s created", attempt, len(records), args.cpu_pods)

    if not records:
        raise SystemExit("Failed to create any CPU worker pods.")

    # Wait for first ready pod to seed shared run assets.
    seed_connection: RunpodPodConnection | None = None
    for rec in records:
        try:
            seed_connection = wait_for_pod_ssh(
                client,
                pod_id=rec.pod_id,
                key_path=key_path,
                timeout_seconds=int(args.startup_timeout_seconds),
                poll_interval_seconds=int(args.poll_interval_seconds),
            )
            rec.public_ip = seed_connection.public_ip
            rec.ssh_port = seed_connection.ssh_port
            rec.status = "ssh_ready"
            break
        except Exception as exc:
            rec.status = "ssh_failed"
            rec.launch_error = str(exc)
            log.warning("SSH readiness failed for seed candidate %s: %s", rec.name, exc)
    if seed_connection is None:
        raise SystemExit("No CPU pod reached SSH readiness; cannot seed shared assets.")

    run_ssh(
        seed_connection,
        key_path=key_path,
        remote_command=f"mkdir -p {json.dumps(run_root)} {json.dumps(run_root + '/repo')}",
        check=True,
        capture_output=False,
    )
    upload_file(seed_connection, key_path=key_path, local_path=repo_archive, remote_path=f"{run_root}/repo_snapshot.tar.gz")
    upload_file(seed_connection, key_path=key_path, local_path=claim_db, remote_path=f"{run_root}/claims.sqlite")
    run_ssh(
        seed_connection,
        key_path=key_path,
        remote_command=f"rm -rf {json.dumps(run_root + '/repo')} && mkdir -p {json.dumps(run_root + '/repo')} && tar --no-same-owner -xzf {json.dumps(run_root + '/repo_snapshot.tar.gz')} -C {json.dumps(run_root + '/repo')} --strip-components=1",
        check=True,
        capture_output=False,
    )
    log.info("Seeded shared repo + claim DB at %s", run_root)

    # Start all CPU worker jobs.
    launched_workers = 0
    for rec in records:
        try:
            conn = wait_for_pod_ssh(
                client,
                pod_id=rec.pod_id,
                key_path=key_path,
                timeout_seconds=int(args.startup_timeout_seconds),
                poll_interval_seconds=int(args.poll_interval_seconds),
            )
            rec.public_ip = conn.public_ip
            rec.ssh_port = conn.ssh_port
            run_ssh(
                conn,
                key_path=key_path,
                remote_command=_remote_launch_worker_cmd(
                    run_root=run_root,
                    dataset=str(args.dataset),
                    base_path=str(args.base_path),
                    worker_id=rec.pod_id,
                    embed_batch_size=int(args.embed_batch_size),
                    num_workers=int(args.num_workers),
                ),
                check=True,
                capture_output=False,
            )
            rec.status = "worker_started"
            launched_workers += 1
            log.info("Started worker on %s (%s/%s)", rec.name, launched_workers, len(records))
        except Exception as exc:
            rec.status = "worker_failed"
            rec.launch_error = str(exc)
            log.warning("Worker launch failed on %s: %s", rec.name, exc)

    # Launch scorer pod.
    scorer_record: PodLaunchRecord | None = None
    try:
        scorer_cfg = RunpodPodConfig(
            name=f"fds-partition-scorer-{stamp}",
            image_name=GPU_IMAGE,
            network_volume_id=str(args.network_volume_id),
            compute_type="GPU",
            gpu_type_ids=(str(args.gpu_type_id),),
            gpu_count=1,
            container_disk_in_gb=int(args.container_disk_gb),
            ports=("22/tcp",),
        )
        scorer = client.create_pod(scorer_cfg)
        scorer_record = PodLaunchRecord(
            pod_id=str(scorer["id"]),
            name=str(scorer.get("name") or scorer["id"]),
            role="gpu_scorer",
            status="created",
        )
        conn = wait_for_pod_ssh(
            client,
            pod_id=scorer_record.pod_id,
            key_path=key_path,
            timeout_seconds=int(args.startup_timeout_seconds),
            poll_interval_seconds=int(args.poll_interval_seconds),
        )
        scorer_record.public_ip = conn.public_ip
        scorer_record.ssh_port = conn.ssh_port
        run_ssh(
            conn,
            key_path=key_path,
            remote_command=_remote_launch_scorer_cmd(
                run_root=run_root,
                dataset=str(args.dataset),
                base_path=str(args.base_path),
                worker_id=scorer_record.pod_id,
                scorer_model=str(args.scorer_model),
                metric_batch_size=int(args.metric_batch_size),
            ),
            check=True,
            capture_output=False,
        )
        scorer_record.status = "scorer_started"
        log.info("Started GPU scorer pod %s", scorer_record.pod_id)
    except Exception as exc:
        if scorer_record is None:
            scorer_record = PodLaunchRecord(pod_id="unknown", name="gpu_scorer", role="gpu_scorer", status="failed", launch_error=str(exc))
        else:
            scorer_record.status = "failed"
            scorer_record.launch_error = str(exc)
        log.warning("GPU scorer launch failed: %s", exc)

    manifest = {
        "stamp": stamp,
        "dataset": args.dataset,
        "run_root": run_root,
        "local_run_root": str(local_run_root),
        "claim_db_local": str(claim_db),
        "repo_archive_local": str(repo_archive),
        "cpu_requested": int(args.cpu_pods),
        "cpu_created": len(records),
        "cpu_workers_started": launched_workers,
        "cpu_pods": [asdict(rec) for rec in records],
        "gpu_scorer": asdict(scorer_record) if scorer_record else None,
    }
    manifest_path = local_run_root / "rollout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Rollout manifest: %s", manifest_path)
    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
