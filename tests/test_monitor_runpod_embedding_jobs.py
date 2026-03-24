from __future__ import annotations

import json
from pathlib import Path

from scripts.monitor_runpod_embedding_jobs import _load_jobs


def test_load_jobs_from_multiple_summaries_dedupes_by_pod_id(tmp_path: Path):
    summary_a = {
        "jobs": [
            {
                "pod_id": "pod-a",
                "pod_name": "tiny-pod",
                "preset_key": "tiny_imagenet_embedding_fixed",
                "remote_run_dir": "/workspace/runs/tiny",
                "remote_status_path": "/workspace/runs/tiny/status.json",
                "remote_study_dir": "/workspace/runs/tiny/studies",
            }
        ]
    }
    summary_b = {
        "jobs": [
            {
                "pod_id": "pod-b",
                "pod_name": "food-pod",
                "preset_key": "food101_embedding_fixed",
                "remote_run_dir": "/workspace/runs/food",
                "remote_status_path": "/workspace/runs/food/status.json",
                "remote_study_dir": "/workspace/runs/food/studies",
            },
            {
                "pod_id": "pod-a",
                "pod_name": "tiny-pod-new",
                "preset_key": "tiny_imagenet_embedding_fixed",
                "remote_run_dir": "/workspace/runs/tiny2",
                "remote_status_path": "/workspace/runs/tiny2/status.json",
                "remote_study_dir": "/workspace/runs/tiny2/studies",
            },
        ]
    }
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(summary_a), encoding="utf-8")
    b_path.write_text(json.dumps(summary_b), encoding="utf-8")

    jobs = _load_jobs([str(a_path), str(b_path)])

    assert len(jobs) == 2
    by_id = {job.pod_id: job for job in jobs}
    assert by_id["pod-a"].pod_name == "tiny-pod-new"
    assert by_id["pod-b"].remote_study_dir == "/workspace/runs/food/studies"
