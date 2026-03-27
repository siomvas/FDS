#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import smtplib
import subprocess
import time
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_runpod_embedding_tuning import _remote_probe_status
from scripts.runpod_utils import RunpodClient, RunpodPodConnection, env_or_none, pod_ssh_connection, run_ssh

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonitorJob:
    pod_id: str
    pod_name: str
    preset_key: str
    remote_run_dir: str
    remote_status_path: str
    remote_study_dir: str


def _default_alert_email() -> str | None:
    explicit = env_or_none("FDS_ALERT_EMAIL")
    if explicit:
        return explicit
    probe = subprocess.run(["git", "config", "--get", "user.email"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    email = (probe.stdout or "").strip()
    if not email or email.endswith("@users.noreply.github.com"):
        return None
    return email


def _send_mail_with_system_mail(*, to_addr: str, subject: str, body: str) -> None:
    subprocess.run(
        ["/usr/bin/mail", "-s", subject, to_addr],
        input=body,
        text=True,
        check=True,
    )


def _send_mail_with_smtp(*, to_addr: str, subject: str, body: str) -> None:
    host = env_or_none("SMTP_HOST")
    if not host:
        raise RuntimeError("SMTP_HOST is not configured.")
    port = int(env_or_none("SMTP_PORT") or "587")
    user = env_or_none("SMTP_USER")
    password = env_or_none("SMTP_PASSWORD")
    sender = env_or_none("SMTP_FROM") or user or to_addr
    message = EmailMessage()
    message["From"] = sender
    message["To"] = to_addr
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(message)


def send_alert_email(*, to_addr: str | None, subject: str, body: str) -> bool:
    if not to_addr:
        log.warning("No alert email recipient configured; skipping email alert. Subject=%s", subject)
        return False
    try:
        if env_or_none("SMTP_HOST"):
            _send_mail_with_smtp(to_addr=to_addr, subject=subject, body=body)
        elif shutil.which("mail"):
            _send_mail_with_system_mail(to_addr=to_addr, subject=subject, body=body)
        else:
            raise RuntimeError("No supported email delivery method is configured.")
    except Exception as exc:
        log.exception("Failed to send alert email to %s: %s", to_addr, exc)
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor RunPod embedding-study pods and clean up finished jobs.")
    parser.add_argument("--api-key", default=None, type=str, help="RunPod API key. Defaults to RUNPOD_API_KEY.")
    parser.add_argument("--summary-json", nargs="+", required=True, help="One or more launcher summary.json files.")
    parser.add_argument("--ssh-key-path", default="~/.ssh/id_ed25519", type=str, help="Private SSH key.")
    parser.add_argument("--poll-seconds", default=900, type=int, help="Polling interval in seconds.")
    parser.add_argument("--state-file", default=None, type=str, help="Optional monitor state file path.")
    parser.add_argument("--alert-email", default=None, type=str, help="Recipient for crash alerts.")
    parser.add_argument("--delete-finished-pods", action="store_true", help="Delete pods after successful completion.")
    parser.add_argument("--log-level", default="INFO", type=str, help="Python logging level.")
    return parser.parse_args()


def _load_jobs(summary_paths: list[str]) -> list[MonitorJob]:
    jobs: list[MonitorJob] = []
    for raw_path in summary_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("jobs", []):
            jobs.append(
                MonitorJob(
                    pod_id=str(item["pod_id"]),
                    pod_name=str(item["pod_name"]),
                    preset_key=str(item["preset_key"]),
                    remote_run_dir=str(item["remote_run_dir"]),
                    remote_status_path=str(item["remote_status_path"]),
                    remote_study_dir=str(item["remote_study_dir"]),
                )
            )
    deduped: dict[str, MonitorJob] = {}
    for job in jobs:
        deduped[job.pod_id] = job
    return list(deduped.values())


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _probe_process(connection: RunpodPodConnection, *, key_path: Path) -> bool:
    proc = run_ssh(
        connection,
        key_path=key_path,
        remote_command="pgrep -af 'python.*scripts/tune_embedding_preset.py' >/dev/null",
        check=False,
        capture_output=False,
    )
    return proc.returncode == 0


def _workspace_backed(connection: RunpodPodConnection, *, key_path: Path, remote_run_dir: str) -> bool:
    proc = run_ssh(
        connection,
        key_path=key_path,
        remote_command=f"test -d {json.dumps(remote_run_dir)}",
        check=False,
        capture_output=False,
    )
    return proc.returncode == 0 and remote_run_dir.startswith("/workspace/")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    api_key = args.api_key or env_or_none("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("Missing RunPod API key. Pass --api-key or set RUNPOD_API_KEY.")
    summary_paths = list(args.summary_json)
    jobs = _load_jobs(summary_paths)
    if not jobs:
        log.warning("No jobs found yet in the provided summary JSON files. Monitor will keep polling for manifests.")
    state_path = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else (REPO_ROOT / "local_artifacts" / "runpod" / "monitor" / "embedding_jobs_state.json").resolve()
    )
    state = _load_state(state_path)
    key_path = Path(args.ssh_key_path).expanduser().resolve()
    client = RunpodClient(api_key)
    alert_email = args.alert_email or _default_alert_email()

    while True:
        jobs = _load_jobs(summary_paths)
        pods = {str(p["id"]): p for p in client.list_pods()}
        dirty = False
        for job in jobs:
            job_state = state.setdefault(job.pod_id, {})
            if job_state.get("terminal"):
                continue
            pod = pods.get(job.pod_id)
            if pod is None:
                if not job_state.get("missing_alerted"):
                    send_alert_email(
                        to_addr=alert_email,
                        subject=f"RunPod pod missing: {job.pod_name}",
                        body=f"Pod {job.pod_name} ({job.pod_id}) is missing from the RunPod API before completion.\nPreset: {job.preset_key}\nRun dir: {job.remote_run_dir}\n",
                    )
                    job_state["missing_alerted"] = True
                    dirty = True
                continue
            connection = pod_ssh_connection(pod)
            if connection is None:
                log.info("Pod %s is not SSH-ready yet.", job.pod_name)
                continue
            status = _remote_probe_status(connection, key_path=key_path, remote_status_path=job.remote_status_path)
            if status is not None:
                rc = int(status.get("returncode", 1))
                if rc == 0:
                    if _workspace_backed(connection, key_path=key_path, remote_run_dir=job.remote_run_dir):
                        log.info("Job %s finished successfully.", job.pod_name)
                        if args.delete_finished_pods:
                            try:
                                client.delete_pod(job.pod_id)
                                log.info("Deleted finished pod %s.", job.pod_name)
                            except Exception as exc:
                                log.exception("Failed to delete finished pod %s: %s", job.pod_name, exc)
                    job_state["terminal"] = True
                    job_state["result"] = "success"
                    dirty = True
                    continue
                if not job_state.get("failure_alerted"):
                    send_alert_email(
                        to_addr=alert_email,
                        subject=f"RunPod study failed: {job.pod_name}",
                        body=(
                            f"Pod {job.pod_name} ({job.pod_id}) finished with rc={rc}.\n"
                            f"Preset: {job.preset_key}\nRun dir: {job.remote_run_dir}\nStatus file: {job.remote_status_path}\n"
                        ),
                    )
                    job_state["failure_alerted"] = True
                    dirty = True
                continue
            if _probe_process(connection, key_path=key_path):
                log.info("Job %s still running.", job.pod_name)
                continue
            if not job_state.get("crash_alerted"):
                send_alert_email(
                    to_addr=alert_email,
                    subject=f"RunPod study appears crashed: {job.pod_name}",
                    body=(
                        f"Pod {job.pod_name} ({job.pod_id}) has no live tune_embedding_preset.py process and no status.json.\n"
                        f"Preset: {job.preset_key}\nRun dir: {job.remote_run_dir}\n"
                    ),
                )
                job_state["crash_alerted"] = True
                dirty = True
        if dirty:
            _save_state(state_path, state)
        time.sleep(max(30, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
