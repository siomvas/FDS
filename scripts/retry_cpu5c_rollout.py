#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_runpod_embedding_tuning import _pod_name
from scripts.runpod_utils import RunpodClient, RunpodApiError

DEFAULT_LOCAL_ROOT = REPO_ROOT / "local_artifacts" / "runpod" / "embedding_tuning" / "cpu5c_staggered"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobSpec:
    label: str
    preset_key: str
    target_n_clients: int
    remote_run_root: str
    local_run_root: Path

    @property
    def pod_name(self) -> str:
        return _pod_name(self.preset_key, self.local_run_root.name)

    @property
    def preset_token(self) -> str:
        return self.preset_key.replace("_", "-")[:40]


def _build_job_specs(local_root: Path) -> list[JobSpec]:
    return [
        JobSpec(
            label="k10_tiny",
            preset_key="tiny_imagenet_embedding_fixed",
            target_n_clients=10,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data/20260320_135303",
            local_run_root=local_root / "k10_tiny",
        ),
        JobSpec(
            label="k10_food",
            preset_key="food101_embedding_fixed",
            target_n_clients=10,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data/20260320_135303",
            local_run_root=local_root / "k10_food",
        ),
        JobSpec(
            label="k10_ade",
            preset_key="ade20k_embedding_fixed",
            target_n_clients=10,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data/20260320_135303",
            local_run_root=local_root / "k10_ade",
        ),
        JobSpec(
            label="k10_pascal",
            preset_key="PascalVOC_embedding_fixed",
            target_n_clients=10,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data/20260320_135303",
            local_run_root=local_root / "k10_pascal",
        ),
        JobSpec(
            label="k5_tiny",
            preset_key="tiny_imagenet_embedding_fixed",
            target_n_clients=5,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data_k5/20260321_bruteforce_k5",
            local_run_root=local_root / "k5_tiny",
        ),
        JobSpec(
            label="k5_food",
            preset_key="food101_embedding_fixed",
            target_n_clients=5,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data_k5/20260321_bruteforce_k5",
            local_run_root=local_root / "k5_food",
        ),
        JobSpec(
            label="k5_ade",
            preset_key="ade20k_embedding_fixed",
            target_n_clients=5,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data_k5/20260321_bruteforce_k5",
            local_run_root=local_root / "k5_ade",
        ),
        JobSpec(
            label="k5_pascal",
            preset_key="PascalVOC_embedding_fixed",
            target_n_clients=5,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data_k5/20260321_bruteforce_k5",
            local_run_root=local_root / "k5_pascal",
        ),
        JobSpec(
            label="k20_tiny",
            preset_key="tiny_imagenet_embedding_fixed",
            target_n_clients=20,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data_k20/20260321_bruteforce_k20",
            local_run_root=local_root / "k20_tiny",
        ),
        JobSpec(
            label="k20_food",
            preset_key="food101_embedding_fixed",
            target_n_clients=20,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data_k20/20260321_bruteforce_k20",
            local_run_root=local_root / "k20_food",
        ),
        JobSpec(
            label="k20_ade",
            preset_key="ade20k_embedding_fixed",
            target_n_clients=20,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data_k20/20260321_bruteforce_k20",
            local_run_root=local_root / "k20_ade",
        ),
        JobSpec(
            label="k20_pascal",
            preset_key="PascalVOC_embedding_fixed",
            target_n_clients=20,
            remote_run_root="/workspace/runs/manual_embedding_tuning_full_data_k20/20260321_bruteforce_k20",
            local_run_root=local_root / "k20_pascal",
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry missing cpu5c RunPod launches until the full rollout exists.")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"), help="RunPod API key.")
    parser.add_argument("--network-volume-id", required=True, help="RunPod network volume id.")
    parser.add_argument("--data-center-id", default="EU-RO-1", help="Preferred RunPod data center.")
    parser.add_argument("--image-name", default="runpod/base:1.0.2-ubuntu2404", help="Pod image name.")
    parser.add_argument("--cpu-flavor-id", default="cpu5c", help="CPU flavor id.")
    parser.add_argument("--vcpu-count", default=32, type=int, help="vCPU count.")
    parser.add_argument("--container-disk-gb", default=50, type=int, help="Container disk size in GB.")
    parser.add_argument("--study-base-path", default="/workspace/datasets/raw", help="Remote dataset base path.")
    parser.add_argument("--cache-dir", default="/workspace/caches/fds", help="Remote cache root.")
    parser.add_argument("--n-trials", default=168, type=int, help="Trial cap passed to the study launcher.")
    parser.add_argument("--retry-sleep-seconds", default=300, type=int, help="Sleep between full retry passes.")
    parser.add_argument("--stagger-seconds", default=30, type=int, help="Sleep between launch attempts.")
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT), help="Local artifact root for manifests/logs.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def _pod_names_by_status(client: RunpodClient) -> set[str]:
    pods = client.list_pods()
    names = set()
    for pod in pods:
        desired = pod.get("desiredStatus")
        actual = pod.get("status")
        if desired == "TERMINATED" or actual == "TERMINATED":
            continue
        name = pod.get("name")
        if name:
            names.add(str(name))
    return names


def _manifest_exists(job: JobSpec) -> bool:
    return (job.local_run_root / "run_manifest.json").exists()


def _pod_matches_job(pod_name: str, job: JobSpec) -> bool:
    return job.preset_token in pod_name and f"k{job.target_n_clients}" in pod_name


def _launch_job(args: argparse.Namespace, job: JobSpec) -> int:
    launcher = REPO_ROOT / "scripts" / "launch_runpod_embedding_tuning.py"
    command = [
        "python3",
        str(launcher),
        "--api-key",
        str(args.api_key),
        "--compute-type",
        "CPU",
        "--cpu-flavor-id",
        str(args.cpu_flavor_id),
        "--vcpu-count",
        str(int(args.vcpu_count)),
        "--image-name",
        str(args.image_name),
        "--container-disk-gb",
        str(int(args.container_disk_gb)),
        "--network-volume-id",
        str(args.network_volume_id),
        "--data-center-id",
        str(args.data_center_id),
        "--study-base-path",
        str(args.study_base_path),
        "--cache-dir",
        str(args.cache_dir),
        "--n_trials",
        str(int(args.n_trials)),
        "--sampler",
        "brute_force",
        "--bruteforce-avoid-premature-stop",
        "--device",
        "cpu",
        "--keep-pods",
        "--remote-run-root",
        job.remote_run_root,
        "--local-run-root",
        str(job.local_run_root),
        "--target-n-clients",
        str(int(job.target_n_clients)),
        "--presets",
        job.preset_key,
    ]
    job.local_run_root.mkdir(parents=True, exist_ok=True)
    log_path = job.local_run_root / "retry_driver.log"
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n[retry-launch] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {job.label}\n")
        handle.flush()
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if not args.api_key:
        raise SystemExit("Missing RunPod API key. Pass --api-key or set RUNPOD_API_KEY.")

    client = RunpodClient(str(args.api_key))
    jobs = _build_job_specs(Path(args.local_root))
    while True:
        try:
            existing_pods = _pod_names_by_status(client)
        except RunpodApiError as exc:
            log.warning("Failed to list pods: %s", exc)
            time.sleep(max(30, int(args.retry_sleep_seconds)))
            continue

        missing = [
            job
            for job in jobs
            if not _manifest_exists(job) and not any(_pod_matches_job(name, job) for name in existing_pods)
        ]
        if not missing:
            log.info("All cpu5c rollout jobs now exist or have manifests. Exiting retry loop.")
            return 0

        log.info("Missing rollout jobs: %s", ", ".join(job.label for job in missing))
        for index, job in enumerate(missing, start=1):
            try:
                existing_pods = _pod_names_by_status(client)
            except RunpodApiError as exc:
                log.warning("Failed to refresh pods before %s: %s", job.label, exc)
                break
            if _manifest_exists(job) or any(_pod_matches_job(name, job) for name in existing_pods):
                continue

            log.info("Launching %s (%s/%s)", job.label, index, len(missing))
            returncode = _launch_job(args, job)
            if returncode == 0:
                log.info("Launch command for %s succeeded.", job.label)
            else:
                log.warning("Launch command for %s exited with code %s.", job.label, returncode)
            time.sleep(max(0, int(args.stagger_seconds)))

        log.info("Retry pass complete. Sleeping for %s seconds.", int(args.retry_sleep_seconds))
        time.sleep(max(30, int(args.retry_sleep_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
