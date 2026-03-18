#!/usr/bin/env bash
set -euo pipefail

export FDS_CACHE_DIR="/path/to/gtai/FDS/embedding_cache"
export HF_HOME="/path/to/data/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/path/to/data/.cache/torch"
export FDS_HPC_PROXY="http://proxy.example.org:3128"
export https_proxy="${FDS_HPC_PROXY}"
export HTTPS_PROXY="${FDS_HPC_PROXY}"
export http_proxy="${FDS_HPC_PROXY}"
export HTTP_PROXY="${FDS_HPC_PROXY}"

export DATASETS_ROOT="/path/to/gtai/FDS"
export tiny_imagenet_data_root="${DATASETS_ROOT}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
	export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
	export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
	export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1}"
fi

mkdir -p "${FDS_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}"

echo "[hpc_session_env] FDS_CACHE_DIR=${FDS_CACHE_DIR}"
echo "[hpc_session_env] HF_HOME=${HF_HOME}"
echo "[hpc_session_env] HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "[hpc_session_env] TORCH_HOME=${TORCH_HOME}"
echo "[hpc_session_env] FDS_HPC_PROXY=${FDS_HPC_PROXY}"
echo "[hpc_session_env] DATASETS_ROOT=${DATASETS_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	echo "[hpc_session_env] HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
	echo "[hpc_session_env] TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
	echo "[hpc_session_env] HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
fi
