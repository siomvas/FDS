from __future__ import annotations

from argparse import Namespace

from scripts.launch_runpod_embedding_tuning import build_remote_study_script, _study_command


def test_study_command_includes_optional_arguments():
    args = Namespace(
        study_base_path="/workspace/data",
        n_trials=25,
        sampler="brute_force",
        bruteforce_avoid_premature_stop=True,
        seed=7,
        num_workers=3,
        embed_batch_size=64,
        metric_batch_size=96,
        artifact_top_k=4,
        target_n_clients=5,
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
    assert "--sampler" in command and "brute_force" in command
    assert "--bruteforce_avoid_premature_stop" in command
    assert "--device" in command and "cuda" in command
    assert "--target_n_clients" in command and "5" in command
    assert "--proxy" in command
    assert "--no_cache" in command


def test_study_command_auto_sampler_uses_bruteforce_for_discovery_and_tpe_for_fixed():
    shared = dict(
        study_base_path="/workspace/data",
        n_trials=10,
        sampler="auto",
        bruteforce_avoid_premature_stop=False,
        seed=7,
        num_workers=2,
        embed_batch_size=64,
        metric_batch_size=96,
        artifact_top_k=4,
        target_n_clients=None,
        min_client_size=20,
        cache_dir="/workspace/caches/fds",
        timeout=None,
        device=None,
        proxy=None,
        no_cache=False,
    )

    fixed_command = _study_command(
        Namespace(**shared),
        preset_key="tiny_imagenet_embedding_fixed",
        remote_study_dir="/workspace/run/studies",
    )
    discovery_command = _study_command(
        Namespace(**shared),
        preset_key="tiny_imagenet_embedding_discovery",
        remote_study_dir="/workspace/run/studies",
    )

    assert fixed_command[fixed_command.index("--sampler") + 1] == "tpe"
    assert discovery_command[discovery_command.index("--sampler") + 1] == "brute_force"


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
    assert "python -m pip install -e '/root/FDS[partitioning,embedding,evaluation]'" in script
    assert "if ! python3 -m venv \"$VENV_PATH\"; then" in script
    assert "if ! python3 -m pip --version >/dev/null 2>&1; then" in script
    assert "apt-get update" in script
    assert "apt-get install -y python3-pip python3-venv" in script
    assert "python3 -m pip install --user virtualenv" in script
    assert "python3 -m virtualenv \"$VENV_PATH\"" in script
    assert "[RESUME] removed stale RUNNING trials" in script
    assert "SELECT trial_id FROM trials WHERE state = 'RUNNING'" in script
    assert "export OPENBLAS_NUM_THREADS=8" in script
    assert "export OMP_NUM_THREADS=8" in script
    assert "export MKL_NUM_THREADS=8" in script
    assert "export NUMEXPR_NUM_THREADS=8" in script
    assert "export PIP_CACHE_DIR=\"$(dirname \"$HF_HOME\")/pip\"" in script
    assert "export TMPDIR=\"$(dirname \"$HF_HOME\")/tmp\"" in script
    assert "finished_at=\"$(date -Is)\"" in script
    assert "\"preset\": \"food101_embedding_fixed\"" in script
