#!/usr/bin/env bash
set -euo pipefail

PARTITION="${1:-gtai}"
SESSION_NAME="${2:-fds-hpc}"
GPUS="${3:-1}"
CPUS_PER_TASK="${4:-8}"
MEM="${5:-64G}"
TIME_LIMIT="${6:-04:00:00}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on PATH" >&2
  exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "srun not found on PATH" >&2
  exit 1
fi

case "${PARTITION}" in
  gtai)
    GRES="gpu:a100_80g:${GPUS}"
    ;;
  preemptgpu)
    GRES="gpu:a100_80g:${GPUS}"
    ;;
  gpu-a100)
    GRES="gpu:a100:${GPUS}"
    ;;
  *)
    echo "Unsupported partition '${PARTITION}'. Use: gtai | preemptgpu | gpu-a100" >&2
    exit 1
    ;;
esac

SRUN_CMD="cd /path/to/data/FDS && source scripts/hpc_session_env.sh && srun -p ${PARTITION} --gres=${GRES} --cpus-per-task=${CPUS_PER_TASK} --mem=${MEM} --time=${TIME_LIMIT} --pty bash -l"
TMUX_CMD="${SRUN_CMD}; RC=\$?; echo; echo '[hpc_tmux_srun] srun exited with code' \"\$RC\"; echo '[hpc_tmux_srun] tmux session kept alive for inspection.'; exec bash -l"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Attaching existing tmux session '${SESSION_NAME}'"
  tmux attach-session -t "${SESSION_NAME}"
  exit 0
fi

echo "Creating tmux session '${SESSION_NAME}'"
echo "Launching interactive job: ${SRUN_CMD}"

tmux new-session -d -s "${SESSION_NAME}" "${TMUX_CMD}"
tmux attach-session -t "${SESSION_NAME}"
