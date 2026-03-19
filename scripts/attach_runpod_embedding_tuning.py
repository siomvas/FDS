#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partitioning.tuning.presets import PRESETS
from scripts.launch_runpod_embedding_tuning import (
    JobRecord,
    _best_trial_summary,
    _create_remote_script,
    _default_presets,
    _persist_manifest,
    _remote_probe_status,
    _study_command,
    _timestamp,
    build_remote_study_script,
)
from scripts.runpod_utils import (
    RunpodPodConnection,
    archive_repo_snapshot,
    create_directory_archive,
    maybe_unlink,
    upload_file,
    run_ssh,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManualPodSpec:
    preset_key: str
    public_ip: str
    ssh_port: int
    pod_name: str
    pod_id: str | None = None
    volume_mount_path: str = "/workspace"
    data_center_id: str | None = None

    def to_connection(self) -> RunpodPodConnection:
        pod_id = self.pod_id or self.pod_name or self.public_ip
        return RunpodPodConnection(
            pod_id=pod_id,
            name=self.pod_name,
            public_ip=self.public_ip,
            ssh_port=int(self.ssh_port),
            volume_mount_path=self.volume_mount_path,
            data_center_id=self.data_center_id,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach existing RunPod Pods to embedding Optuna preset execution over SSH.")
    parser.add_argument("--ssh-key-path", default="~/.ssh/id_ed25519", type=str, help="Private SSH key for Pod access.")
    parser.add_argument("--study-base-path", required=True, type=str, help="Dataset base path as seen from inside the Pod.")
    parser.add_argument(
        "--local-study-base-path",
        default=None,
        type=str,
        help="Optional local dataset directory to upload and extract to --study-base-path.",
    )
    parser.add_argument("--remote-worktree", default="/root/FDS", type=str, help="Remote checkout path inside the Pod.")
    parser.add_argument("--remote-venv-path", default="/root/.venvs/fds", type=str, help="Remote virtualenv path.")
    parser.add_argument("--cache-dir", default="/workspace/caches/fds", type=str, help="Remote FDS cache dir.")
    parser.add_argument("--hf-home", default="/workspace/caches/hf", type=str, help="Remote HF cache dir.")
    parser.add_argument("--torch-home", default="/workspace/caches/torch", type=str, help="Remote torch cache dir.")
    parser.add_argument("--run-stamp", default=None, type=str, help="Override the run timestamp/stamp.")
    parser.add_argument(
        "--remote-run-root",
        default=None,
        type=str,
        help="Remote persistent root for logs/studies. Defaults to /workspace/runs/manual_embedding_tuning/<stamp>.",
    )
    parser.add_argument(
        "--local-run-root",
        default=None,
        type=str,
        help="Local manifest root. Defaults to local_artifacts/runpod/manual_embedding_tuning/<stamp>.",
    )
    parser.add_argument(
        "--pod-specs-file",
        default=None,
        type=str,
        help="JSON file describing one or more manually created Pods.",
    )
    parser.add_argument("--preset", default=None, type=str, help="Single preset key when launching against one pod.")
    parser.add_argument("--host", default=None, type=str, help="Single Pod public IP/hostname.")
    parser.add_argument("--ssh-port", default=None, type=int, help="Single Pod SSH port.")
    parser.add_argument("--pod-name", default="manual-runpod-pod", type=str, help="Single Pod logical name.")
    parser.add_argument("--pod-id", default=None, type=str, help="Optional Pod id for manifest purposes.")
    parser.add_argument("--n_trials", default=100, type=int, help="Trials per preset.")
    parser.add_argument("--timeout", default=None, type=int, help="Optional timeout passed to Optuna.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--num_workers", default=4, type=int, help="Worker count for embeddings/metrics.")
    parser.add_argument("--device", default=None, type=str, help="Optional device override inside the Pod.")
    parser.add_argument("--proxy", default=None, type=str, help="Optional proxy override passed to the study.")
    parser.add_argument("--embed_batch_size", default=100, type=int, help="Embedding extraction batch size.")
    parser.add_argument("--metric_batch_size", default=128, type=int, help="Metric feature batch size.")
    parser.add_argument("--artifact_top_k", default=10, type=int, help="Best-trial artifacts retained per study.")
    parser.add_argument("--min_client_size", default=20, type=int, help="Minimum acceptable client size.")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding/metric caching in the study.")
    parser.add_argument("--poll-interval-seconds", default=30, type=int, help="Status polling interval.")
    parser.add_argument("--log-level", default="INFO", type=str, help="Python logging level.")
    return parser.parse_args()


def load_manual_pod_specs(args: argparse.Namespace) -> list[ManualPodSpec]:
    if args.pod_specs_file:
        path = Path(args.pod_specs_file).expanduser().resolve()
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        items = payload if isinstance(payload, list) else payload.get("pods")
        if not isinstance(items, list):
            raise ValueError(f"Pod specs file must contain a list or a 'pods' list: {path}")
        specs: list[ManualPodSpec] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Each pod spec must be a mapping: {item!r}")
            preset_key = item.get("preset_key") or item.get("preset")
            if not preset_key:
                raise ValueError(f"Pod spec is missing preset_key: {item}")
            specs.append(
                ManualPodSpec(
                    preset_key=str(preset_key),
                    public_ip=str(item["public_ip"] if "public_ip" in item else item["host"]),
                    ssh_port=int(item["ssh_port"]),
                    pod_name=str(item.get("pod_name") or item.get("name") or f"pod-{preset_key}"),
                    pod_id=str(item["pod_id"]) if item.get("pod_id") else None,
                    volume_mount_path=str(item.get("volume_mount_path") or "/workspace"),
                    data_center_id=str(item["data_center_id"]) if item.get("data_center_id") else None,
                )
            )
        return specs

    if not args.host or args.ssh_port is None:
        raise ValueError("Provide either --pod-specs-file or the single-pod flags --host --ssh-port --preset.")
    preset_key = args.preset or (_default_presets()[0] if args.preset is None else args.preset)
    return [
        ManualPodSpec(
            preset_key=preset_key,
            public_ip=str(args.host),
            ssh_port=int(args.ssh_port),
            pod_name=str(args.pod_name),
            pod_id=str(args.pod_id) if args.pod_id else None,
        )
    ]


def _launch_remote_script(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    remote_script_path: str,
) -> None:
    launch_command = f"chmod +x {json.dumps(remote_script_path)} && nohup bash {json.dumps(remote_script_path)} >/dev/null 2>&1 < /dev/null &"
    run_ssh(connection, key_path=key_path, remote_command=launch_command, check=True, capture_output=False)


def _prepare_manual_job(
    args: argparse.Namespace,
    *,
    connection: RunpodPodConnection,
    key_path: Path,
    repo_archive_path: Path,
    data_archive_path: Path | None,
    local_run_root: Path,
    remote_run_root: str,
    preset_key: str,
) -> JobRecord:
    if preset_key not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown embedding preset '{preset_key}'. Valid presets: {valid}")
    remote_run_dir = f"{remote_run_root.rstrip('/')}/{preset_key}"
    remote_log_path = f"{remote_run_dir}/stdout.log"
    remote_status_path = f"{remote_run_dir}/status.json"
    remote_study_dir = f"{remote_run_dir}/studies"
    repo_archive_remote_path = f"/tmp/{connection.pod_id}-fds-repo.tar.gz"
    upload_file(connection, key_path=key_path, local_path=repo_archive_path, remote_path=repo_archive_remote_path)
    data_archive_remote_path = None
    if data_archive_path is not None:
        data_archive_remote_path = f"/tmp/{connection.pod_id}-fds-data.tar.gz"
        upload_file(connection, key_path=key_path, local_path=data_archive_path, remote_path=data_archive_remote_path)
    remote_script_path = f"/tmp/{connection.pod_id}-{preset_key}.sh"
    script_text = build_remote_study_script(
        preset_key=preset_key,
        repo_archive_remote_path=repo_archive_remote_path,
        data_archive_remote_path=data_archive_remote_path,
        remote_worktree=args.remote_worktree,
        remote_venv_path=args.remote_venv_path,
        remote_run_dir=remote_run_dir,
        remote_log_path=remote_log_path,
        remote_status_path=remote_status_path,
        remote_study_dir=remote_study_dir,
        study_base_path=args.study_base_path,
        cache_dir=args.cache_dir,
        hf_home=args.hf_home,
        torch_home=args.torch_home,
        study_command=_study_command(args, preset_key=preset_key, remote_study_dir=remote_study_dir),
    )
    local_script_path = _create_remote_script(script_text, local_run_root=local_run_root, preset_key=preset_key)
    try:
        upload_file(connection, key_path=key_path, local_path=local_script_path, remote_path=remote_script_path)
    finally:
        maybe_unlink(local_script_path)
    _launch_remote_script(connection, key_path=key_path, remote_script_path=remote_script_path)
    return JobRecord(
        preset_key=preset_key,
        pod_id=connection.pod_id,
        pod_name=connection.name,
        public_ip=connection.public_ip,
        ssh_port=connection.ssh_port,
        remote_run_dir=remote_run_dir,
        remote_log_path=remote_log_path,
        remote_status_path=remote_status_path,
        remote_study_dir=remote_study_dir,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    stamp = args.run_stamp or _timestamp()
    remote_run_root = args.remote_run_root or f"/workspace/runs/manual_embedding_tuning/{stamp}"
    local_run_root = (
        Path(args.local_run_root).expanduser().resolve()
        if args.local_run_root
        else (REPO_ROOT / "local_artifacts" / "runpod" / "manual_embedding_tuning" / stamp).resolve()
    )
    local_run_root.mkdir(parents=True, exist_ok=True)
    args.remote_run_root = remote_run_root

    key_path = Path(args.ssh_key_path).expanduser().resolve()
    if not key_path.exists():
        raise SystemExit(f"SSH private key not found: {key_path}")

    pod_specs = load_manual_pod_specs(args)
    repo_archive_path = archive_repo_snapshot(REPO_ROOT, output_path=local_run_root / "repo_snapshot.tar.gz")
    data_archive_path = None
    if args.local_study_base_path:
        data_archive_path = local_run_root / "study_base_path.tar.gz"
        create_directory_archive(Path(args.local_study_base_path), output_path=data_archive_path)

    jobs: list[JobRecord] = []
    connections: dict[str, RunpodPodConnection] = {}
    failures: list[str] = []
    try:
        for spec in pod_specs:
            connection = spec.to_connection()
            connections[connection.pod_id] = connection
            log.info("Attaching preset=%s to existing pod=%s host=%s:%s", spec.preset_key, connection.name, connection.public_ip, connection.ssh_port)
            try:
                job = _prepare_manual_job(
                    args,
                    connection=connection,
                    key_path=key_path,
                    repo_archive_path=repo_archive_path,
                    data_archive_path=data_archive_path,
                    local_run_root=local_run_root,
                    remote_run_root=remote_run_root,
                    preset_key=spec.preset_key,
                )
            except Exception as exc:
                message = f"Failed to start preset {spec.preset_key} on pod {connection.name}: {exc}"
                log.exception(message)
                failures.append(message)
                continue
            jobs.append(job)
            _persist_manifest(local_run_root, args=args, jobs=jobs)

        pending = {job.pod_id: job for job in jobs}
        while pending:
            for pod_id, job in list(pending.items()):
                connection = connections[pod_id]
                status = _remote_probe_status(
                    connection,
                    key_path=key_path,
                    remote_status_path=job.remote_status_path,
                )
                if status is None:
                    continue
                job.status = "completed"
                job.returncode = int(status.get("returncode", 1))
                job.best_trial_summary = _best_trial_summary(
                    connection,
                    key_path=key_path,
                    remote_study_dir=job.remote_study_dir,
                )
                if job.returncode != 0:
                    failures.append(f"Preset {job.preset_key} failed with return code {job.returncode} on pod {job.pod_name}.")
                log.info("Preset %s on pod %s finished with rc=%s", job.preset_key, job.pod_name, job.returncode)
                pending.pop(pod_id)
                _persist_manifest(local_run_root, args=args, jobs=jobs)
            if pending:
                time.sleep(max(5, int(args.poll_interval_seconds)))
    finally:
        maybe_unlink(repo_archive_path)
        maybe_unlink(data_archive_path)

    summary = {
        "run_stamp": stamp,
        "remote_run_root": remote_run_root,
        "local_run_root": str(local_run_root),
        "pods": [asdict(spec) for spec in pod_specs],
        "failures": failures,
        "jobs": [asdict(job) for job in jobs],
    }
    with open(local_run_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
    if failures:
        for failure in failures:
            log.error("%s", failure)
        return 1
    log.info("Manual RunPod embedding tuning finished successfully. Local manifest root: %s", local_run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
