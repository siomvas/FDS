from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from scripts.runpod_utils import (
    RunpodPodConfig,
    build_pod_create_payload,
    create_directory_archive,
    pod_ssh_connection,
    resolve_network_volume_id,
)


class _FakeClient:
    def __init__(self, volumes):
        self._volumes = list(volumes)

    def list_network_volumes(self):
        return list(self._volumes)


def test_build_pod_create_payload_includes_optional_fields():
    payload = build_pod_create_payload(
        RunpodPodConfig(
            name="fds-test",
            image_name="runpod/pytorch:test",
            gpu_type_ids=("NVIDIA L40S",),
            network_volume_id="vol-123",
            gpu_count=2,
            container_disk_in_gb=200,
            data_center_ids=("EU-RO-1",),
            env={"ENV_VAR": "value"},
            min_vcpu_per_gpu=8,
            min_ram_per_gpu=32,
        )
    )

    assert payload["name"] == "fds-test"
    assert payload["imageName"] == "runpod/pytorch:test"
    assert payload["gpuTypeIds"] == ["NVIDIA L40S"]
    assert payload["gpuCount"] == 2
    assert payload["networkVolumeId"] == "vol-123"
    assert payload["containerDiskInGb"] == 200
    assert payload["dataCenterIds"] == ["EU-RO-1"]
    assert payload["env"] == {"ENV_VAR": "value"}
    assert payload["minVCPUPerGPU"] == 8
    assert payload["minRAMPerGPU"] == 32


def test_resolve_network_volume_id_by_name_and_data_center():
    client = _FakeClient(
        [
            {"id": "vol-a", "name": "fds", "dataCenterId": "US-KS-1"},
            {"id": "vol-b", "name": "fds", "dataCenterId": "EU-RO-1"},
        ]
    )

    assert resolve_network_volume_id(client, volume_name="fds", data_center_id="EU-RO-1") == "vol-b"


def test_resolve_network_volume_id_rejects_ambiguous_name():
    client = _FakeClient(
        [
            {"id": "vol-a", "name": "fds", "dataCenterId": "US-KS-1"},
            {"id": "vol-b", "name": "fds", "dataCenterId": "EU-RO-1"},
        ]
    )

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_network_volume_id(client, volume_name="fds")


def test_pod_ssh_connection_requires_public_ip_and_ssh_mapping():
    assert pod_ssh_connection({"id": "pod-1", "publicIp": None, "portMappings": {"22": 12345}}) is None
    assert pod_ssh_connection({"id": "pod-1", "publicIp": "1.2.3.4", "portMappings": {}}) is None

    connection = pod_ssh_connection(
        {
            "id": "pod-1",
            "name": "fds-pod",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 12345},
            "volumeMountPath": "/workspace",
            "machine": {"dataCenterId": "EU-RO-1"},
        }
    )

    assert connection is not None
    assert connection.ssh_target() == "root@1.2.3.4"
    assert connection.ssh_port == 12345
    assert connection.data_center_id == "EU-RO-1"


def test_create_directory_archive_excludes_generated_directories(tmp_path: Path):
    source = tmp_path / "repo"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "results").mkdir()
    (source / "results" / "drop.txt").write_text("drop\n", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (source / "pkg" / "__pycache__").mkdir(parents=True)
    (source / "pkg" / "__pycache__" / "mod.pyc").write_bytes(b"pyc")

    archive_path = create_directory_archive(source, output_path=tmp_path / "repo.tar.gz", excluded_roots={"results", ".git"})

    with tarfile.open(archive_path, mode="r:gz") as archive:
        names = archive.getnames()

    assert "source/scripts/keep.py" in names
    assert "source/results/drop.txt" not in names
    assert "source/.git/config" not in names
    assert all("__pycache__" not in name for name in names)
