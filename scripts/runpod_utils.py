from __future__ import annotations

import json
import logging
import os
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib import error, request

log = logging.getLogger(__name__)

RUNPOD_API_BASE_URL = "https://rest.runpod.io/v1"
DEFAULT_REPO_EXCLUDES = (
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".DS_Store",
    ".venv",
    "benchmark_results",
    "local_artifacts",
    "local_data",
    "outputs",
    "results",
)


class RunpodApiError(RuntimeError):
    """Raised when the RunPod REST API returns an unexpected response."""


@dataclass(frozen=True)
class RunpodPodConnection:
    pod_id: str
    name: str
    public_ip: str
    ssh_port: int
    volume_mount_path: str
    data_center_id: str | None = None

    def ssh_target(self) -> str:
        return f"root@{self.public_ip}"


@dataclass(frozen=True)
class RunpodPodConfig:
    name: str
    image_name: str
    network_volume_id: str
    gpu_type_ids: tuple[str, ...] = ()
    gpu_count: int = 1
    cloud_type: str = "SECURE"
    compute_type: str = "GPU"
    cpu_flavor_ids: tuple[str, ...] = ()
    cpu_flavor_priority: str = "availability"
    vcpu_count: int | None = None
    memory_in_gb: int | None = None
    ports: tuple[str, ...] = ("22/tcp", "8888/http")
    container_disk_in_gb: int = 100
    support_public_ip: bool = True
    volume_mount_path: str = "/workspace"
    volume_in_gb: int = 0
    global_networking: bool = True
    interruptible: bool = False
    locked: bool = False
    data_center_ids: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    min_vcpu_per_gpu: int | None = None
    min_ram_per_gpu: int | None = None


def build_pod_create_payload(config: RunpodPodConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": config.name,
        "cloudType": config.cloud_type,
        "computeType": config.compute_type,
        "imageName": config.image_name,
        "networkVolumeId": config.network_volume_id,
        "containerDiskInGb": int(config.container_disk_in_gb),
        "ports": list(config.ports),
        "supportPublicIp": bool(config.support_public_ip),
        "volumeMountPath": config.volume_mount_path,
        "volumeInGb": int(config.volume_in_gb),
        "globalNetworking": bool(config.global_networking),
        "interruptible": bool(config.interruptible),
        "locked": bool(config.locked),
    }
    if config.compute_type.upper() == "CPU":
        if not config.cpu_flavor_ids:
            raise ValueError("CPU Pod creation requires at least one cpu_flavor_id.")
        payload["cpuFlavorIds"] = list(config.cpu_flavor_ids)
        payload["cpuFlavorPriority"] = str(config.cpu_flavor_priority)
        if config.vcpu_count is not None:
            payload["vcpuCount"] = int(config.vcpu_count)
    else:
        if not config.gpu_type_ids:
            raise ValueError("GPU Pod creation requires at least one gpu_type_id.")
        payload["gpuCount"] = int(config.gpu_count)
        payload["gpuTypeIds"] = list(config.gpu_type_ids)
    if config.data_center_ids:
        payload["dataCenterIds"] = list(config.data_center_ids)
        payload["dataCenterPriority"] = "availability"
    if config.env:
        payload["env"] = dict(config.env)
    if config.compute_type.upper() != "CPU" and config.min_vcpu_per_gpu is not None:
        payload["minVCPUPerGPU"] = int(config.min_vcpu_per_gpu)
    if config.compute_type.upper() != "CPU" and config.min_ram_per_gpu is not None:
        payload["minRAMPerGPU"] = int(config.min_ram_per_gpu)
    return payload


class RunpodClient:
    def __init__(self, api_key: str, *, base_url: str = RUNPOD_API_BASE_URL):
        token = api_key.strip()
        if not token:
            raise ValueError("RunPod API key must be non-empty.")
        self.api_key = token
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        expected_statuses: Iterable[int] = (200,),
    ) -> Any:
        data = None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method.upper(),
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                status = response.getcode()
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:  # pragma: no cover - exercised with live API only
            detail = exc.read().decode("utf-8", errors="replace")
            raise RunpodApiError(f"RunPod API {method} {path} failed with {exc.code}: {detail}") from exc
        except error.URLError as exc:  # pragma: no cover - exercised with live API only
            raise RunpodApiError(f"RunPod API {method} {path} failed: {exc.reason}") from exc

        if status not in set(expected_statuses):
            raise RunpodApiError(f"RunPod API {method} {path} returned unexpected status {status}: {body}")
        if not body:
            return None
        return json.loads(body)

    def list_pods(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/pods")
        return list(response)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/pods/{pod_id}"))

    def create_pod(self, config: RunpodPodConfig | Mapping[str, Any]) -> dict[str, Any]:
        payload = build_pod_create_payload(config) if isinstance(config, RunpodPodConfig) else dict(config)
        return dict(self._request("POST", "/pods", payload=payload, expected_statuses=(201,)))

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}", expected_statuses=(200, 204))

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return dict(self._request("POST", f"/pods/{pod_id}/stop", expected_statuses=(200,)))

    def start_pod(self, pod_id: str) -> dict[str, Any]:
        return dict(self._request("POST", f"/pods/{pod_id}/start", expected_statuses=(200,)))

    def list_network_volumes(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/networkvolumes")
        return list(response)

    def create_network_volume(self, *, name: str, size: int, data_center_id: str) -> dict[str, Any]:
        payload = {"name": name, "size": int(size), "dataCenterId": data_center_id}
        return dict(self._request("POST", "/networkvolumes", payload=payload, expected_statuses=(201,)))


def resolve_network_volume_id(
    client: RunpodClient,
    *,
    volume_id: str | None = None,
    volume_name: str | None = None,
    data_center_id: str | None = None,
) -> str:
    if volume_id:
        return volume_id
    if not volume_name:
        raise ValueError("Provide either a RunPod network volume id or a volume name.")
    matches = [volume for volume in client.list_network_volumes() if volume.get("name") == volume_name]
    if data_center_id:
        matches = [volume for volume in matches if volume.get("dataCenterId") == data_center_id]
    if not matches:
        raise ValueError(
            f"Could not find a RunPod network volume named '{volume_name}'"
            + (f" in data center '{data_center_id}'." if data_center_id else ".")
        )
    if len(matches) > 1:
        ids = ", ".join(str(volume.get("id")) for volume in matches)
        raise ValueError(
            f"Volume name '{volume_name}' is ambiguous. Matching ids: {ids}. Pass --network-volume-id instead."
        )
    return str(matches[0]["id"])


def pod_ssh_connection(pod: Mapping[str, Any]) -> RunpodPodConnection | None:
    public_ip = pod.get("publicIp")
    if not public_ip:
        return None
    port_mappings = pod.get("portMappings") or {}
    ssh_port = port_mappings.get("22")
    if ssh_port in (None, ""):
        return None
    machine = pod.get("machine") or {}
    return RunpodPodConnection(
        pod_id=str(pod["id"]),
        name=str(pod.get("name") or pod["id"]),
        public_ip=str(public_ip),
        ssh_port=int(ssh_port),
        volume_mount_path=str(pod.get("volumeMountPath") or "/workspace"),
        data_center_id=str(machine.get("dataCenterId")) if machine.get("dataCenterId") else None,
    )


def build_ssh_argv(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    remote_command: str,
    extra_options: Sequence[str] = (),
) -> list[str]:
    argv = [
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
        "ConnectTimeout=15",
        "-p",
        str(connection.ssh_port),
    ]
    argv.extend(extra_options)
    argv.extend([connection.ssh_target(), remote_command])
    return argv


def run_ssh(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    remote_command: str,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        build_ssh_argv(connection, key_path=key_path, remote_command=remote_command),
        check=check,
        text=True,
        capture_output=capture_output,
    )


def build_scp_argv(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    local_path: Path,
    remote_path: str,
) -> list[str]:
    return [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-P",
        str(connection.ssh_port),
        str(local_path),
        f"{connection.ssh_target()}:{remote_path}",
    ]


def upload_file(
    connection: RunpodPodConnection,
    *,
    key_path: Path,
    local_path: Path,
    remote_path: str,
) -> None:
    subprocess.run(
        build_scp_argv(connection, key_path=key_path, local_path=local_path, remote_path=remote_path),
        check=True,
        text=True,
    )


def wait_for_pod_ssh(
    client: RunpodClient,
    *,
    pod_id: str,
    key_path: Path,
    timeout_seconds: int,
    poll_interval_seconds: int = 10,
) -> RunpodPodConnection:
    deadline = time.monotonic() + timeout_seconds
    last_status = "UNKNOWN"
    while time.monotonic() < deadline:
        pod = client.get_pod(pod_id)
        last_status = str(pod.get("desiredStatus") or pod.get("lastStatusChange") or "UNKNOWN")
        connection = pod_ssh_connection(pod)
        if connection is not None:
            probe = subprocess.run(
                build_ssh_argv(connection, key_path=key_path, remote_command="true"),
                check=False,
                text=True,
                capture_output=True,
            )
            if probe.returncode == 0:
                return connection
            log.debug("SSH probe for pod %s failed: %s", pod_id, probe.stderr.strip())
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for pod {pod_id} SSH readiness. Last status: {last_status}")


def _normalize_archive_relpath(name: str) -> tuple[str, ...]:
    parts = Path(name).parts
    if not parts:
        return ()
    if parts[0] == ".":
        return tuple(parts[1:])
    return tuple(parts[1:]) if len(parts) > 1 else ()


def _archive_filter(tarinfo: tarfile.TarInfo, *, excluded_roots: set[str]) -> tarfile.TarInfo | None:
    rel_parts = _normalize_archive_relpath(tarinfo.name)
    if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in rel_parts):
        return None
    if rel_parts and rel_parts[0] in excluded_roots:
        return None
    if tarinfo.name.endswith(".pyc"):
        return None
    return tarinfo


def create_directory_archive(
    source_dir: Path,
    *,
    output_path: Path | None = None,
    excluded_roots: Iterable[str] = (),
) -> Path:
    source = source_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Archive source directory does not exist: {source}")
    excluded = set(excluded_roots)
    if output_path is None:
        handle = tempfile.NamedTemporaryFile(prefix="fds-runpod-", suffix=".tar.gz", delete=False)
        output = Path(handle.name)
        handle.close()
    else:
        output = output_path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w:gz") as archive:
        archive.add(
            source,
            arcname="source",
            filter=lambda tarinfo: _archive_filter(tarinfo, excluded_roots=excluded),
        )
    return output


def archive_repo_snapshot(repo_root: Path, *, output_path: Path | None = None) -> Path:
    return create_directory_archive(repo_root, output_path=output_path, excluded_roots=DEFAULT_REPO_EXCLUDES)


def maybe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def env_or_none(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
