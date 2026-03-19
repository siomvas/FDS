from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.attach_runpod_embedding_tuning import ManualPodSpec, load_manual_pod_specs


def test_manual_pod_spec_to_connection():
    spec = ManualPodSpec(
        preset_key="food101_embedding_fixed",
        public_ip="1.2.3.4",
        ssh_port=31234,
        pod_name="food-pod",
        pod_id="pod-123",
        volume_mount_path="/workspace",
        data_center_id="EU-RO-1",
    )

    connection = spec.to_connection()

    assert connection.pod_id == "pod-123"
    assert connection.name == "food-pod"
    assert connection.public_ip == "1.2.3.4"
    assert connection.ssh_port == 31234
    assert connection.volume_mount_path == "/workspace"
    assert connection.data_center_id == "EU-RO-1"


def test_load_manual_pod_specs_from_single_pod_args():
    args = Namespace(
        pod_specs_file=None,
        host="1.2.3.4",
        ssh_port=30022,
        preset="tiny_imagenet_embedding_fixed",
        pod_name="tiny-pod",
        pod_id="pod-1",
    )

    specs = load_manual_pod_specs(args)

    assert len(specs) == 1
    assert specs[0].preset_key == "tiny_imagenet_embedding_fixed"
    assert specs[0].public_ip == "1.2.3.4"
    assert specs[0].ssh_port == 30022
    assert specs[0].pod_name == "tiny-pod"
    assert specs[0].pod_id == "pod-1"


def test_load_manual_pod_specs_from_file(tmp_path: Path):
    payload = {
        "pods": [
            {
                "preset_key": "food101_embedding_fixed",
                "host": "1.2.3.4",
                "ssh_port": 30022,
                "name": "food-pod",
                "pod_id": "pod-food",
            },
            {
                "preset": "ade20k_embedding_fixed",
                "public_ip": "5.6.7.8",
                "ssh_port": 30023,
                "pod_name": "ade-pod",
            },
        ]
    }
    path = tmp_path / "pods.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    args = Namespace(
        pod_specs_file=str(path),
        host=None,
        ssh_port=None,
        preset=None,
        pod_name=None,
        pod_id=None,
    )

    specs = load_manual_pod_specs(args)

    assert [spec.preset_key for spec in specs] == ["food101_embedding_fixed", "ade20k_embedding_fixed"]
    assert specs[0].pod_name == "food-pod"
    assert specs[1].public_ip == "5.6.7.8"
    assert specs[1].pod_name == "ade-pod"


def test_load_manual_pod_specs_requires_input():
    args = Namespace(
        pod_specs_file=None,
        host=None,
        ssh_port=None,
        preset=None,
        pod_name=None,
        pod_id=None,
    )

    with pytest.raises(ValueError, match="Provide either --pod-specs-file"):
        load_manual_pod_specs(args)
