from __future__ import annotations

from argparse import Namespace

from scripts.launch_runpod_embedding_tuning import build_remote_study_script, _study_command


def test_study_command_includes_optional_arguments():
    args = Namespace(
        study_base_path="/workspace/data",
        n_trials=25,
        seed=7,
        num_workers=3,
        embed_batch_size=64,
        metric_batch_size=96,
        artifact_top_k=4,
        min_client_size=12,
        cache_dir="/workspace/caches/fds",
        timeout=3600,
        device="cuda",
        proxy="",
        no_cache=True,
    )

    command = _study_command(args, preset_key="tiny_imagenet_embedding_fixed", remote_study_dir="/workspace/run/studies")

    assert command[:3] == ["python", "scripts/tune_embedding_preset.py", "tiny_imagenet_embedding_fixed"]
    assert "--base_path" in command and "/workspace/data" in command
    assert "--study_dir" in command and "/workspace/run/studies" in command
    assert "--timeout" in command and "3600" in command
    assert "--device" in command and "cuda" in command
    assert "--proxy" in command
    assert "--no_cache" in command


def test_build_remote_study_script_includes_sync_block_and_status_file():
    script = build_remote_study_script(
        preset_key="food101_embedding_fixed",
        repo_archive_remote_path="/tmp/repo.tar.gz",
        data_archive_remote_path="/tmp/data.tar.gz",
        remote_worktree="/root/FDS",
        remote_venv_path="/root/.venvs/fds",
        remote_run_dir="/workspace/runs/food101",
        remote_log_path="/workspace/runs/food101/stdout.log",
        remote_status_path="/workspace/runs/food101/status.json",
        remote_study_dir="/workspace/runs/food101/studies",
        study_base_path="/workspace/data",
        cache_dir="/workspace/caches/fds",
        hf_home="/workspace/caches/hf",
        torch_home="/workspace/caches/torch",
        study_command=["python", "scripts/tune_embedding_preset.py", "food101_embedding_fixed"],
    )

    assert "tar -xzf /tmp/repo.tar.gz -C \"$WORKTREE\" --strip-components=1" in script
    assert "tar -xzf /tmp/data.tar.gz -C /workspace/data --strip-components=1" in script
    assert "READY_MARKER=/workspace/data/.fds_runpod_sync_ready" in script
    assert "python -m pip install -e '/root/FDS[partitioning,embedding,evaluation,fl,dev]'" in script
    assert "export OPENBLAS_NUM_THREADS=8" in script
    assert "export OMP_NUM_THREADS=8" in script
    assert "export MKL_NUM_THREADS=8" in script
    assert "export NUMEXPR_NUM_THREADS=8" in script
    assert "finished_at=\"$(date -Is)\"" in script
    assert "\"preset\": \"food101_embedding_fixed\"" in script
