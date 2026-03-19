#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partitioning.tuning.presets import PRESETS
from scripts.runpod_utils import (
    RunpodClient,
    RunpodPodConfig,
    RunpodPodConnection,
    archive_repo_snapshot,
    create_directory_archive,
    env_or_none,
    maybe_unlink,
    resolve_network_volume_id,
    run_ssh,
    upload_file,
    wait_for_pod_ssh,
)

log = logging.getLogger(__name__)

DEFAULT_IMAGE_NAME = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
DEFAULT_GPU_TYPE_ID = "NVIDIA L40S"


@dataclass
class JobRecord:
    preset_key: str
    pod_id: str
    pod_name: str
    public_ip: str
    ssh_port: int
    remote_run_dir: str
    remote_log_path: str
    remote_status_path: str
    remote_study_dir: str
    status: str = "launched"
    returncode: int | None = None
    best_trial_summary: dict[str, Any] | None = None


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch embedding Optuna presets on RunPod Pods.")
    parser.add_argument("--api-key", default=None, type=str, help="RunPod API key. Defaults to RUNPOD_API_KEY.")
    parser.add_argument("--network-volume-id", default=None, type=str, help="Existing RunPod network volume id.")
    parser.add_argument("--network-volume-name", default=None, type=str, help="Existing RunPod network volume name.")
    parser.add_argument("--data-center-id", default=None, type=str, help="Optional RunPod data center constraint.")
    parser.add_argument("--gpu-type-id", default=DEFAULT_GPU_TYPE_ID, type=str, help="RunPod GPU type id/display name.")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME, type=str, help="Pod image to deploy.")
    parser.add_argument("--gpu-count", default=1, type=int, help="GPUs per pod.")
    parser.add_argument("--container-disk-gb", default=100, type=int, help="Ephemeral container disk size.")
    parser.add_argument("--ssh-key-path", default="~/.ssh/id_ed25519", type=str, help="Private SSH key for Pod access.")
    parser.add_argument(
        "--study-base-path",
        required=True,
        type=str,
        help="Dataset base path as seen from inside the Pod.",
    )
    parser.add_argument(
        "--local-study-base-path",
        default=None,
        type=str,
        help="Optional local directory to upload and extract to --study-base-path before running studies.",
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
        help="Remote persistent root for logs/studies. Defaults to /workspace/runs/embedding_tuning/<stamp>.",
    )
    parser.add_argument(
        "--local-run-root",
        default=None,
        type=str,
        help="Local manifest root. Defaults to local_artifacts/runpod/embedding_tuning/<stamp>.",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Embedding preset keys to launch. Defaults to the four overnight fixed presets.",
    )
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
    parser.add_argument("--keep-pods", action="store_true", help="Do not delete Pods after completion.")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding/metric caching in the study.")
    parser.add_argument("--startup-timeout-seconds", default=1800, type=int, help="Pod boot + SSH readiness timeout.")
    parser.add_argument("--poll-interval-seconds", default=30, type=int, help="Status polling interval.")
    parser.add_argument("--log-level", default="INFO", type=str, help="Python logging level.")
    return parser.parse_args()


def _default_presets() -> list[str]:
    return [
        "tiny_imagenet_embedding_fixed",
        "food101_embedding_fixed",
        "ade20k_embedding_fixed",
        "PascalVOC_embedding_fixed",
    ]


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def _study_command(args: argparse.Namespace, *, preset_key: str, remote_study_dir: str) -> list[str]:
    command = [
        "python",
        "scripts/tune_embedding_preset.py",
        preset_key,
        "--base_path",
        args.study_base_path,
        "--study_dir",
        remote_study_dir,
        "--n_trials",
        str(int(args.n_trials)),
        "--seed",
        str(int(args.seed)),
        "--num_workers",
        str(int(args.num_workers)),
        "--embed_batch_size",
        str(int(args.embed_batch_size)),
        "--metric_batch_size",
        str(int(args.metric_batch_size)),
        "--artifact_top_k",
        str(int(args.artifact_top_k)),
        "--min_client_size",
        str(int(args.min_client_size)),
        "--cache_dir",
        args.cache_dir,
    ]
    if args.timeout is not None:
        command.extend(["--timeout", str(int(args.timeout))])
    if args.device:
        command.extend(["--device", args.device])
    if args.proxy is not None:
        command.extend(["--proxy", args.proxy])
    if args.no_cache:
        command.append("--no_cache")
    return command


def build_remote_study_script(
    *,
    preset_key: str,
    repo_archive_remote_path: str,
    data_archive_remote_path: str | None,
    remote_worktree: str,
    remote_venv_path: str,
    remote_run_dir: str,
    remote_log_path: str,
    remote_status_path: str,
    remote_study_dir: str,
    study_base_path: str,
    cache_dir: str,
    hf_home: str,
    torch_home: str,
    study_command: list[str],
) -> str:
    install_target = f"{remote_worktree}[partitioning,embedding,evaluation,fl,dev]"
    data_sync_block = ""
    if data_archive_remote_path:
        ready_marker = f"{study_base_path.rstrip('/')}/.fds_runpod_sync_ready"
        lock_dir = f"{study_base_path.rstrip('/')}/.fds_runpod_sync_lock"
        data_sync_block = f"""
READY_MARKER={_q(ready_marker)}
LOCK_DIR={_q(lock_dir)}
if [[ ! -f "$READY_MARKER" ]]; then
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    if [[ ! -f "$READY_MARKER" ]]; then
      rm -rf {_q(study_base_path)}
      mkdir -p {_q(study_base_path)}
      tar -xzf {_q(data_archive_remote_path)} -C {_q(study_base_path)} --strip-components=1
      date -Is > "$READY_MARKER"
    fi
    rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
  else
    until [[ -f "$READY_MARKER" ]]; do
      sleep 15
    done
  fi
fi
"""
    return f"""#!/usr/bin/env bash
set -euo pipefail
RUN_DIR={_q(remote_run_dir)}
WORKTREE={_q(remote_worktree)}
VENV_PATH={_q(remote_venv_path)}
LOG_PATH={_q(remote_log_path)}
STATUS_PATH={_q(remote_status_path)}
STUDY_DIR={_q(remote_study_dir)}
mkdir -p "$RUN_DIR" "$(dirname "$LOG_PATH")" "$(dirname "$STATUS_PATH")" "$STUDY_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1
echo "[START] $(date -Is) preset={preset_key}"
rm -rf "$WORKTREE"
mkdir -p "$WORKTREE"
tar -xzf {_q(repo_archive_remote_path)} -C "$WORKTREE" --strip-components=1
{data_sync_block}if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e {_q(install_target)}
export FDS_CACHE_DIR={_q(cache_dir)}
export HF_HOME={_q(hf_home)}
export TORCH_HOME={_q(torch_home)}
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
unset FDS_HPC_PROXY || true
mkdir -p "$FDS_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
cd "$WORKTREE"
set +e
{_shell_join(study_command)}
rc=$?
set -e
finished_at="$(date -Is)"
printf '{{\\n  "preset": "{preset_key}",\\n  "returncode": %s,\\n  "finished_at": "%s"\\n}}\\n' \\
  "$rc" "$finished_at" > "$STATUS_PATH"
echo "[END] $(date -Is) preset={preset_key} rc=$rc"
exit "$rc"
"""


def _remote_probe_status(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    remote_status_path: str,
) -> dict[str, Any] | None:
    probe_command = f"if [[ -f {_q(remote_status_path)} ]]; then cat {_q(remote_status_path)}; fi"
    proc = run_ssh(
        connection,
        key_path=key_path,
        remote_command=probe_command,
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    raw = (proc.stdout or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _create_remote_script(
    script_text: str,
    *,
    local_run_root: Path,
    preset_key: str,
) -> Path:
    local_run_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"runpod-{preset_key}-",
        suffix=".sh",
        dir=local_run_root,
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        handle.write(script_text)
        temp_path = Path(handle.name)
    temp_path.chmod(0o755)
    return temp_path


def _launch_remote_script(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    remote_script_path: str,
) -> None:
    launch_command = f"chmod +x {_q(remote_script_path)} && nohup bash {_q(remote_script_path)} >/dev/null 2>&1 < /dev/null &"
    run_ssh(connection, key_path=key_path, remote_command=launch_command, check=True, capture_output=False)


def _pod_name(preset_key: str, stamp: str) -> str:
    safe = preset_key.replace("_", "-")[:40]
    return f"fds-{safe}-{stamp.lower()}"[:63]


def _prepare_job(
    args: argparse.Namespace,
    *,
    client: RunpodClient,
    key_path: Path,
    repo_archive_path: Path,
    data_archive_path: Path | None,
    local_run_root: Path,
    remote_run_root: str,
    preset_key: str,
    network_volume_id: str,
    cleanup_on_error: bool,
) -> JobRecord:
    if preset_key not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown embedding preset '{preset_key}'. Valid presets: {valid}")

    try:
        pod_config = RunpodPodConfig(
            name=_pod_name(preset_key, local_run_root.name),
            image_name=args.image_name,
            gpu_type_ids=(args.gpu_type_id,),
            network_volume_id=network_volume_id,
            gpu_count=int(args.gpu_count),
            container_disk_in_gb=int(args.container_disk_gb),
            data_center_ids=(args.data_center_id,) if args.data_center_id else (),
            env={},
        )
        pod = client.create_pod(pod_config)
        pod_id = str(pod["id"])
        connection = wait_for_pod_ssh(
            client,
            pod_id=pod_id,
            key_path=key_path,
            timeout_seconds=int(args.startup_timeout_seconds),
            poll_interval_seconds=max(5, int(args.poll_interval_seconds)),
        )
        remote_run_dir = f"{remote_run_root.rstrip('/')}/{preset_key}"
        remote_log_path = f"{remote_run_dir}/stdout.log"
        remote_status_path = f"{remote_run_dir}/status.json"
        remote_study_dir = f"{remote_run_dir}/studies"
        repo_archive_remote_path = f"/tmp/{pod_id}-fds-repo.tar.gz"
        upload_file(connection, key_path=key_path, local_path=repo_archive_path, remote_path=repo_archive_remote_path)
        data_archive_remote_path = None
        if data_archive_path is not None:
            data_archive_remote_path = f"/tmp/{pod_id}-fds-data.tar.gz"
            upload_file(connection, key_path=key_path, local_path=data_archive_path, remote_path=data_archive_remote_path)

        remote_script_path = f"/tmp/{pod_id}-{preset_key}.sh"
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
            pod_id=pod_id,
            pod_name=connection.name,
            public_ip=connection.public_ip,
            ssh_port=connection.ssh_port,
            remote_run_dir=remote_run_dir,
            remote_log_path=remote_log_path,
            remote_status_path=remote_status_path,
            remote_study_dir=remote_study_dir,
        )
    except Exception:
        if cleanup_on_error and "pod_id" in locals():
            try:
                client.delete_pod(pod_id)
            except Exception:  # pragma: no cover - depends on live API
                log.warning("Failed to delete partially prepared pod %s", pod_id, exc_info=True)
        raise


def _persist_manifest(local_run_root: Path, *, args: argparse.Namespace, jobs: list[JobRecord]) -> None:
    payload = {
        "run_stamp": local_run_root.name,
        "study_base_path": args.study_base_path,
        "local_study_base_path": args.local_study_base_path,
        "remote_run_root": args.remote_run_root,
        "cache_dir": args.cache_dir,
        "hf_home": args.hf_home,
        "torch_home": args.torch_home,
        "presets": [job.preset_key for job in jobs],
        "jobs": [asdict(job) for job in jobs],
    }
    _write_json(local_run_root / "run_manifest.json", payload)


def _best_trial_summary(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    remote_study_dir: str,
) -> dict[str, Any] | None:
    path = f"{remote_study_dir.rstrip('/')}/best/best_trial.json"
    proc = run_ssh(
        connection,
        key_path=key_path,
        remote_command=f"if [[ -f {_q(path)} ]]; then cat {_q(path)}; fi",
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return dict(json.loads(proc.stdout))
    except json.JSONDecodeError:
        return None


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    api_key = args.api_key or env_or_none("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("Missing RunPod API key. Pass --api-key or set RUNPOD_API_KEY.")

    stamp = args.run_stamp or _timestamp()
    remote_run_root = args.remote_run_root or f"/workspace/runs/embedding_tuning/{stamp}"
    local_run_root = (
        Path(args.local_run_root).expanduser().resolve()
        if args.local_run_root
        else (REPO_ROOT / "local_artifacts" / "runpod" / "embedding_tuning" / stamp).resolve()
    )
    local_run_root.mkdir(parents=True, exist_ok=True)
    args.remote_run_root = remote_run_root

    presets = list(args.presets or _default_presets())
    key_path = Path(args.ssh_key_path).expanduser().resolve()
    if not key_path.exists():
        raise SystemExit(f"SSH private key not found: {key_path}")

    client = RunpodClient(api_key)
    network_volume_id = resolve_network_volume_id(
        client,
        volume_id=args.network_volume_id,
        volume_name=args.network_volume_name,
        data_center_id=args.data_center_id,
    )

    repo_archive_path = archive_repo_snapshot(REPO_ROOT, output_path=local_run_root / "repo_snapshot.tar.gz")
    data_archive_path = None
    if args.local_study_base_path:
        data_archive_path = Path(local_run_root / "study_base_path.tar.gz")
        create_directory_archive(Path(args.local_study_base_path), output_path=data_archive_path)

    jobs: list[JobRecord] = []
    connections: dict[str, RunpodPodConnection] = {}
    failures: list[str] = []
    try:
        for preset_key in presets:
            log.info("Launching RunPod study for preset=%s", preset_key)
            try:
                job = _prepare_job(
                    args,
                    client=client,
                    key_path=key_path,
                    repo_archive_path=repo_archive_path,
                    data_archive_path=data_archive_path,
                    local_run_root=local_run_root,
                    remote_run_root=remote_run_root,
                    preset_key=preset_key,
                    network_volume_id=network_volume_id,
                    cleanup_on_error=not args.keep_pods,
                )
            except Exception as exc:
                message = f"Failed to launch preset {preset_key}: {exc}"
                log.exception(message)
                failures.append(message)
                continue
            jobs.append(job)
            connections[job.pod_id] = RunpodPodConnection(
                pod_id=job.pod_id,
                name=job.pod_name,
                public_ip=job.public_ip,
                ssh_port=job.ssh_port,
                volume_mount_path="/workspace",
                data_center_id=args.data_center_id,
            )
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
                    failures.append(f"Preset {job.preset_key} failed with return code {job.returncode}.")
                log.info("Preset %s finished with rc=%s", job.preset_key, job.returncode)
                if not args.keep_pods:
                    try:
                        client.delete_pod(pod_id)
                    except Exception as exc:  # pragma: no cover - depends on live API
                        failures.append(f"Failed to delete pod {pod_id} for preset {job.preset_key}: {exc}")
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
        "failures": failures,
        "jobs": [asdict(job) for job in jobs],
    }
    _write_json(local_run_root / "summary.json", summary)
    if failures:
        for failure in failures:
            log.error("%s", failure)
        return 1
    for job in jobs:
        if job.best_trial_summary:
            log.info(
                "Preset %s best trial=%s value=%s",
                job.preset_key,
                job.best_trial_summary.get("number"),
                job.best_trial_summary.get("value"),
            )
    log.info("RunPod embedding tuning finished successfully. Local manifest root: %s", local_run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
