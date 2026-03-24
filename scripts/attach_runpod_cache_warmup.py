#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partitioning.tuning.presets import PRESETS
from scripts.attach_runpod_embedding_tuning import load_manual_pod_specs
from scripts.launch_runpod_embedding_tuning import _create_remote_script, _remote_probe_status, _timestamp
from scripts.runpod_utils import RunpodPodConnection, archive_repo_snapshot, maybe_unlink, run_ssh, upload_file

log = logging.getLogger(__name__)


@dataclass
class CacheWarmJobRecord:
    preset_key: str
    pod_id: str
    pod_name: str
    public_ip: str
    ssh_port: int
    remote_run_dir: str
    remote_log_path: str
    remote_status_path: str
    remote_manifest_path: str
    status: str = "launched"
    returncode: int | None = None


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach existing RunPod Pods to phase-A cache warmup over SSH.")
    parser.add_argument("--ssh-key-path", default="~/.ssh/id_ed25519", type=str, help="Private SSH key for Pod access.")
    parser.add_argument("--study-base-path", required=True, type=str, help="Dataset base path as seen from inside the Pod.")
    parser.add_argument("--remote-worktree", default="/root/FDS", type=str, help="Remote checkout path inside the Pod.")
    parser.add_argument("--remote-venv-path", default="/root/.venvs/fds", type=str, help="Remote virtualenv path.")
    parser.add_argument("--cache-dir", default="/workspace/caches/fds", type=str, help="Remote shared FDS cache dir.")
    parser.add_argument("--hf-home", default="/workspace/caches/hf", type=str, help="Remote HF cache dir.")
    parser.add_argument("--torch-home", default="/workspace/caches/torch", type=str, help="Remote torch cache dir.")
    parser.add_argument("--run-stamp", default=None, type=str, help="Override the run timestamp/stamp.")
    parser.add_argument("--remote-run-root", default=None, type=str, help="Remote persistent root for cache-warm logs.")
    parser.add_argument("--local-run-root", default=None, type=str, help="Local manifest root.")
    parser.add_argument("--pod-specs-file", default=None, type=str, help="JSON file describing one or more manually created Pods.")
    parser.add_argument("--preset", default=None, type=str, help="Single preset key when launching against one pod.")
    parser.add_argument("--host", default=None, type=str, help="Single Pod public IP/hostname.")
    parser.add_argument("--ssh-port", default=None, type=int, help="Single Pod SSH port.")
    parser.add_argument("--pod-name", default="manual-runpod-pod", type=str, help="Single Pod logical name.")
    parser.add_argument("--pod-id", default=None, type=str, help="Optional Pod id for manifest purposes.")
    parser.add_argument("--study-mode-override", default=None, choices=["fixed", "discovery"], help="Optional override for the preset study mode.")
    parser.add_argument("--device", default="cuda", type=str, help="Device override inside the Pod.")
    parser.add_argument("--proxy", default=None, type=str, help="Optional proxy override.")
    parser.add_argument("--jobs-json", default=None, type=str, help="Optional explicit cache job list to run instead of the full preset matrix.")
    parser.add_argument("--embed-batch-size", default=100, type=int, help="Embedding extractor batch size.")
    parser.add_argument("--metric-batch-size", default=128, type=int, help="Scorer feature batch size.")
    parser.add_argument("--num-workers", default=4, type=int, help="Worker count for cache warmup.")
    parser.add_argument("--poll-interval-seconds", default=30, type=int, help="Status polling interval.")
    parser.add_argument("--log-level", default="INFO", type=str, help="Python logging level.")
    return parser.parse_args()


def _warm_command(args: argparse.Namespace, *, preset_key: str, remote_manifest_path: str) -> list[str]:
    preset = PRESETS[preset_key]
    command = [
        "python",
        "scripts/warm_embedding_cache_matrix.py",
        "--dataset",
        preset.dataset_name,
        "--study_mode",
        args.study_mode_override or preset.study_mode,
        "--base_path",
        args.study_base_path,
        "--cache_dir",
        args.cache_dir,
        "--device",
        args.device,
        "--embed_batch_size",
        str(int(args.embed_batch_size)),
        "--metric_batch_size",
        str(int(args.metric_batch_size)),
        "--num_workers",
        str(int(args.num_workers)),
        "--manifest_path",
        remote_manifest_path,
    ]
    if args.proxy is not None:
        command.extend(["--proxy", args.proxy])
    if args.jobs_json is not None:
        remote_jobs_path = f"{remote_manifest_path.rsplit('/', 1)[0]}/jobs.json"
        command.extend(["--jobs_json", remote_jobs_path])
    return command


def build_remote_cache_warm_script(
    *,
    preset_key: str,
    repo_archive_remote_path: str,
    remote_worktree: str,
    remote_venv_path: str,
    remote_run_dir: str,
    remote_log_path: str,
    remote_status_path: str,
    cache_dir: str,
    hf_home: str,
    torch_home: str,
    warm_command: list[str],
) -> str:
    install_target = f"{remote_worktree}[partitioning,embedding,evaluation]"
    return f"""#!/usr/bin/env bash
set -euo pipefail
RUN_DIR={_q(remote_run_dir)}
WORKTREE={_q(remote_worktree)}
VENV_PATH={_q(remote_venv_path)}
LOG_PATH={_q(remote_log_path)}
STATUS_PATH={_q(remote_status_path)}
mkdir -p "$RUN_DIR" "$(dirname "$LOG_PATH")" "$(dirname "$STATUS_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1
echo "[START] $(date -Is) preset={preset_key}"
rm -f "$STATUS_PATH"
rm -rf "$WORKTREE"
mkdir -p "$WORKTREE"
tar -xzf {_q(repo_archive_remote_path)} -C "$WORKTREE" --strip-components=1
mkdir -p "$(dirname "$VENV_PATH")"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  if ! python3 -m venv "$VENV_PATH"; then
    rm -rf "$VENV_PATH"
    if ! python3 -m pip --version >/dev/null 2>&1; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv
    fi
    if ! python3 -m venv "$VENV_PATH"; then
      python3 -m pip install --user virtualenv
      python3 -m virtualenv "$VENV_PATH"
    fi
  fi
fi
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e {_q(install_target)}
export FDS_CACHE_DIR={_q(cache_dir)}
export HF_HOME={_q(hf_home)}
export TORCH_HOME={_q(torch_home)}
export PIP_CACHE_DIR="$(dirname "$HF_HOME")/pip"
export TMPDIR="$(dirname "$HF_HOME")/tmp"
export PYTHONUNBUFFERED=1
unset FDS_HPC_PROXY || true
mkdir -p "$FDS_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
cd "$WORKTREE"
set +e
{_shell_join(warm_command)}
rc=$?
set -e
finished_at="$(date -Is)"
printf '{{\\n  "preset": "{preset_key}",\\n  "returncode": %s,\\n  "finished_at": "%s"\\n}}\\n' \\
  "$rc" "$finished_at" > "$STATUS_PATH"
echo "[END] $(date -Is) preset={preset_key} rc=$rc"
exit "$rc"
"""


def _launch_remote_script(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    remote_script_path: str,
) -> None:
    launch_command = f"chmod +x {_q(remote_script_path)} && nohup bash {_q(remote_script_path)} >/dev/null 2>&1 < /dev/null &"
    run_ssh(connection, key_path=key_path, remote_command=launch_command, check=True, capture_output=False)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    stamp = args.run_stamp or _timestamp()
    remote_run_root = args.remote_run_root or f"/workspace/runs/cache_warmup/{stamp}"
    local_run_root = (
        Path(args.local_run_root).expanduser().resolve()
        if args.local_run_root
        else (REPO_ROOT / "local_artifacts" / "runpod" / "cache_warmup" / stamp).resolve()
    )
    local_run_root.mkdir(parents=True, exist_ok=True)

    key_path = Path(args.ssh_key_path).expanduser().resolve()
    if not key_path.exists():
        raise SystemExit(f"SSH private key not found: {key_path}")

    pod_specs = load_manual_pod_specs(args)
    repo_archive_path = archive_repo_snapshot(REPO_ROOT, output_path=local_run_root / "repo_snapshot.tar.gz")

    jobs: list[CacheWarmJobRecord] = []
    try:
        for spec in pod_specs:
            if spec.preset_key not in PRESETS:
                valid = ", ".join(sorted(PRESETS))
                raise ValueError(f"Unknown embedding preset '{spec.preset_key}'. Valid presets: {valid}")
            connection = spec.to_connection()
            remote_run_dir = f"{remote_run_root.rstrip('/')}/{spec.preset_key}"
            remote_log_path = f"{remote_run_dir}/stdout.log"
            remote_status_path = f"{remote_run_dir}/status.json"
            remote_manifest_path = f"{remote_run_dir}/manifest.json"
            repo_archive_remote_path = f"/tmp/{connection.pod_id}-fds-repo.tar.gz"
            run_ssh(connection, key_path=key_path, remote_command=f"mkdir -p {_q(remote_run_dir)}", check=True, capture_output=True)
            upload_file(connection, key_path=key_path, local_path=repo_archive_path, remote_path=repo_archive_remote_path)
            if args.jobs_json is not None:
                local_jobs_path = Path(args.jobs_json).expanduser().resolve()
                remote_jobs_path = f"{remote_run_dir}/jobs.json"
                upload_file(connection, key_path=key_path, local_path=local_jobs_path, remote_path=remote_jobs_path)
            remote_script_path = f"/tmp/{connection.pod_id}-{spec.preset_key}-cache-warm.sh"
            script_text = build_remote_cache_warm_script(
                preset_key=spec.preset_key,
                repo_archive_remote_path=repo_archive_remote_path,
                remote_worktree=args.remote_worktree,
                remote_venv_path=args.remote_venv_path,
                remote_run_dir=remote_run_dir,
                remote_log_path=remote_log_path,
                remote_status_path=remote_status_path,
                cache_dir=args.cache_dir,
                hf_home=args.hf_home,
                torch_home=args.torch_home,
                warm_command=_warm_command(args, preset_key=spec.preset_key, remote_manifest_path=remote_manifest_path),
            )
            local_script_path = _create_remote_script(script_text, local_run_root=local_run_root, preset_key=spec.preset_key)
            try:
                upload_file(connection, key_path=key_path, local_path=local_script_path, remote_path=remote_script_path)
            finally:
                maybe_unlink(local_script_path)
            _launch_remote_script(connection, key_path=key_path, remote_script_path=remote_script_path)
            jobs.append(
                CacheWarmJobRecord(
                    preset_key=spec.preset_key,
                    pod_id=connection.pod_id,
                    pod_name=connection.name,
                    public_ip=connection.public_ip,
                    ssh_port=connection.ssh_port,
                    remote_run_dir=remote_run_dir,
                    remote_log_path=remote_log_path,
                    remote_status_path=remote_status_path,
                    remote_manifest_path=remote_manifest_path,
                )
            )
    finally:
        maybe_unlink(repo_archive_path)

    manifest_payload = {
        "args": {
            "study_base_path": args.study_base_path,
            "cache_dir": args.cache_dir,
            "run_stamp": stamp,
        },
        "jobs": [asdict(job) for job in jobs],
    }
    with open(local_run_root / "cache_warmup_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    pending = {job.preset_key: job for job in jobs}
    while pending:
        for preset_key, job in list(pending.items()):
            connection = RunpodPodConnection(
                pod_id=job.pod_id,
                name=job.pod_name,
                public_ip=job.public_ip,
                ssh_port=job.ssh_port,
                volume_mount_path="/workspace",
            )
            status = _remote_probe_status(connection, key_path=key_path, remote_status_path=job.remote_status_path)
            if not status:
                continue
            job.status = "completed" if int(status.get("returncode", 1)) == 0 else "failed"
            job.returncode = int(status.get("returncode", 1))
            pending.pop(preset_key, None)
        if pending:
            time.sleep(max(5, int(args.poll_interval_seconds)))

    with open(local_run_root / "cache_warmup_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": {
                    "study_base_path": args.study_base_path,
                    "cache_dir": args.cache_dir,
                    "run_stamp": stamp,
                },
                "jobs": [asdict(job) for job in jobs],
            },
            handle,
            indent=2,
        )

    failures = [job for job in jobs if job.returncode not in (0, None)]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
