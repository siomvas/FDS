#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from scripts.runpod_utils import RunpodApiError, RunpodClient, env_or_none

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PodSpec:
    pod_id: str
    host: str
    ssh_port: int
    name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pause Runpod pods automatically once watched Python workload processes "
            "go idle (either success or crash)."
        )
    )
    parser.add_argument(
        "--pods-json",
        required=True,
        type=str,
        help="Path to JSON with {'pods': [{'pod_id','host','ssh_port','name'}]} or a list of pod dicts.",
    )
    parser.add_argument("--api-key", default=None, type=str, help="Runpod API key. Defaults to RUNPOD_API_KEY.")
    parser.add_argument("--ssh-key-path", default="~/.ssh/id_ed25519", type=str, help="SSH private key path.")
    parser.add_argument(
        "--process-pattern",
        default="run_best_compromise|tune_embedding_preset.py|run_experiment.py|run_matrix_task.py|python -m pip install",
        type=str,
        help="egrep pattern used with pgrep -af on remote pods.",
    )
    parser.add_argument("--poll-seconds", default=60, type=int, help="Polling interval in seconds.")
    parser.add_argument(
        "--idle-confirm-polls",
        default=2,
        type=int,
        help="Number of consecutive idle polls after first observed activity before stop.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        type=str,
        help="State JSON output path. Defaults under local_artifacts/runpod/monitor.",
    )
    parser.add_argument("--log-level", default="INFO", type=str, help="Logging level.")
    return parser.parse_args()


def _load_pods(path: Path) -> list[PodSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("pods", [])
    if not isinstance(items, list):
        raise ValueError("pods-json must be a list or an object with 'pods' list.")
    out: list[PodSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid pod item: {item!r}")
        pod_id = str(item["pod_id"])
        host = str(item.get("host") or item.get("public_ip"))
        ssh_port = int(item["ssh_port"])
        name = str(item.get("name") or pod_id)
        out.append(PodSpec(pod_id=pod_id, host=host, ssh_port=ssh_port, name=name))
    if not out:
        raise ValueError("No pods found in pods-json.")
    return out


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _remote_has_workload(*, pod: PodSpec, key_path: Path, pattern: str) -> tuple[bool, str]:
    cmd = [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=12",
        "-p",
        str(pod.ssh_port),
        f"root@{pod.host}",
        f"pgrep -af {json.dumps(pattern)} || true",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return False, output
    return bool(output), output


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    api_key = args.api_key or env_or_none("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("Missing Runpod API key. Pass --api-key or set RUNPOD_API_KEY.")
    pods = _load_pods(Path(args.pods_json).expanduser().resolve())
    key_path = Path(args.ssh_key_path).expanduser().resolve()
    state_path = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else (REPO_ROOT / "local_artifacts" / "runpod" / "monitor" / "idle_stop_state.json").resolve()
    )
    client = RunpodClient(api_key)

    state: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "poll_seconds": int(args.poll_seconds),
            "idle_confirm_polls": int(args.idle_confirm_polls),
            "process_pattern": str(args.process_pattern),
        },
        "pods": {
            pod.pod_id: {
                "name": pod.name,
                "host": pod.host,
                "ssh_port": pod.ssh_port,
                "had_activity": False,
                "idle_polls": 0,
                "stopped": False,
                "last_seen_active_at": None,
                "last_seen_processes": "",
                "last_error": None,
            }
            for pod in pods
        },
    }
    _save_state(state_path, state)

    log.info("Watching %d pod(s). State file: %s", len(pods), state_path)
    while True:
        try:
            # Keep a lightweight heartbeat against the Runpod API. We don't need the
            # response content right now, but we want transient API issues to be
            # handled inside the loop instead of crashing the daemon.
            client.list_pods()
        except Exception as exc:
            log.exception("Runpod API probe failed; retrying next poll: %s", exc)
            _save_state(state_path, state)
            time.sleep(max(15, int(args.poll_seconds)))
            continue

        active_watch = 0
        for pod in pods:
            pod_state = state["pods"][pod.pod_id]
            if pod_state.get("stopped"):
                continue
            active_watch += 1

            try:
                has_workload, processes = _remote_has_workload(
                    pod=pod,
                    key_path=key_path,
                    pattern=str(args.process_pattern),
                )
                pod_state["last_seen_processes"] = processes
                if has_workload:
                    pod_state["had_activity"] = True
                    pod_state["idle_polls"] = 0
                    pod_state["last_seen_active_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    pod_state["last_error"] = None
                    log.info("Pod %s active.", pod.name)
                    continue

                if not pod_state.get("had_activity"):
                    log.info("Pod %s idle but no activity seen yet; not stopping.", pod.name)
                    continue

                pod_state["idle_polls"] = int(pod_state.get("idle_polls", 0)) + 1
                idle_polls = int(pod_state["idle_polls"])
                log.info("Pod %s idle after activity (%d/%d).", pod.name, idle_polls, int(args.idle_confirm_polls))
                if idle_polls < int(args.idle_confirm_polls):
                    continue

                client.stop_pod(pod.pod_id)
                pod_state["stopped"] = True
                pod_state["stopped_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                pod_state["stop_reason"] = "idle_after_activity"
                pod_state["last_error"] = None
                log.info("Stopped pod %s (%s).", pod.name, pod.pod_id)
            except RunpodApiError as exc:
                pod_state["last_error"] = f"stop_failed: {exc}"
                log.exception("Failed to stop pod %s: %s", pod.name, exc)
            except Exception as exc:
                pod_state["last_error"] = str(exc)
                log.exception("Pod processing failed for %s: %s", pod.name, exc)

        _save_state(state_path, state)
        if active_watch == 0:
            log.info("All watched pods are stopped. Exiting.")
            return 0
        time.sleep(max(15, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
