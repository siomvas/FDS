#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/path/to/data/FDS"
MANIFEST_PATH="${1:-${REPO_ROOT}/local_artifacts/experiments/baselines/20260319_032519/fl_job_manifest_resubmitted_v2.csv}"
SESSION_START="${SESSION_START:-101}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
CONDA_ENV="${CONDA_ENV:-new}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_BASE="$(conda info --base)"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  if [[ -d "/path/to/.conda/new" ]]; then
    CONDA_ENV="/path/to/.conda/new"
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/local_artifacts/experiments/baselines/tmux_relaunch_${STAMP}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

OUT_MANIFEST="${RUN_DIR}/tmux_launch_manifest.csv"
echo "session,job_name,dataset,partition,gres,mem,slurm_script,log_file" > "${OUT_MANIFEST}"

infer_n_clients_from_partition_path() {
  local partition_path="$1"
  local partition_name
  partition_name="$(basename "${partition_path}" .json)"
  if [[ "${partition_name}" =~ _n([0-9]+)_ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

ensure_script_has_n_clients() {
  local slurm_script="$1"
  local n_clients="$2"
  if grep -q 'fed.n_clients=' "${slurm_script}"; then
    if [[ "${n_clients}" == "20" ]] && ! grep -q 'system.gpu_ids=' "${slurm_script}"; then
      python - "${slurm_script}" <<'PY'
import re
import sys
from pathlib import Path

script_path = Path(sys.argv[1])
text = script_path.read_text(encoding="utf-8")

pattern = re.compile(r"(\bfed\.n_clients=20\b\s+)(fed\.rounds=)")
updated, count = pattern.subn(
    r"\1system.gpu_ids=[0,0,0,0,0,0,0,0,0,0]   \2",
    text,
    count=1,
)
if count == 0:
    raise SystemExit(f"Unable to inject n20 gpu_config into {script_path}")

script_path.write_text(updated, encoding="utf-8")
print(f"[repair] injected n20 gpu_config into {script_path}")
PY
    fi
    return 0
  fi

  python - "${slurm_script}" "${n_clients}" <<'PY'
import re
import sys
from pathlib import Path

script_path = Path(sys.argv[1])
n_clients = sys.argv[2]
text = script_path.read_text(encoding="utf-8")

pattern = re.compile(r"(\bseed=[^\s]+\s+)(fed\.rounds=)")
updated, count = pattern.subn(rf"\1fed.n_clients={n_clients}   \2", text, count=1)
if count == 0:
    raise SystemExit(f"Unable to inject fed.n_clients into {script_path}")

script_path.write_text(updated, encoding="utf-8")
print(f"[repair] injected fed.n_clients={n_clients} into {script_path}")
PY

  if [[ "${n_clients}" == "20" ]] && ! grep -q 'system.gpu_ids=' "${slurm_script}"; then
    python - "${slurm_script}" <<'PY'
import re
import sys
from pathlib import Path

script_path = Path(sys.argv[1])
text = script_path.read_text(encoding="utf-8")

pattern = re.compile(r"(\bfed\.n_clients=20\b\s+)(fed\.rounds=)")
updated, count = pattern.subn(
    r"\1system.gpu_ids=[0,0,0,0,0,0,0,0,0,0]   \2",
    text,
    count=1,
)
if count == 0:
    raise SystemExit(f"Unable to inject n20 gpu_config into {script_path}")

script_path.write_text(updated, encoding="utf-8")
print(f"[repair] injected n20 gpu_config into {script_path}")
PY
  fi
}

ensure_script_has_conda_env() {
  local slurm_script="$1"
  local resolved_env="$2"
  if grep -q 'conda activate new' "${slurm_script}"; then
    python - "${slurm_script}" "${resolved_env}" <<'PY'
import sys
from pathlib import Path

script_path = Path(sys.argv[1])
resolved_env = sys.argv[2]
text = script_path.read_text(encoding="utf-8")
updated = text.replace("conda activate new", f"conda activate {resolved_env}")
script_path.write_text(updated, encoding="utf-8")
print(f"[repair] rewrote conda env in {script_path} -> {resolved_env}")
PY
  fi
}

choose_partition() {
  local dataset="$1"
  local idx="$2"
  local n_clients="$3"
  local partition
  local gres
  local mem

  case "${dataset}" in
    ade20k|pascal_voc)
      partition="preemptgpu"
      gres="gpu:a100_80g:1"
      mem="80G"
      ;;
    tiny_imagenet)
      # n10/n20 classification runs must avoid gpu-a100 (40GB VRAM).
      if (( n_clients >= 10 )); then
        if (( idx % 2 == 0 )); then
          partition="gtai"
          gres="gpu:a100_80g:1"
        else
          partition="preemptgpu"
          gres="gpu:a100_80g:1"
        fi
      else
        # n5 can still use gpu-a100 for throughput.
        if (( idx % 2 == 0 )); then
          partition="gtai"
          gres="gpu:a100_80g:1"
        else
          partition="gpu-a100"
          gres="gpu:a100:1"
        fi
      fi
      mem="64G"
      ;;
    food101)
      # Food101 repeatedly OOMs on gpu-a100; force A100-80GB for all client counts.
      if (( idx % 2 == 0 )); then
        partition="gtai"
      else
        partition="preemptgpu"
      fi
      gres="gpu:a100_80g:1"
      mem="64G"
      ;;
    *)
      partition="preemptgpu"
      gres="gpu:a100_80g:1"
      mem="64G"
      ;;
  esac

  echo "${partition},${gres},${mem}"
}

idx=0
while IFS=, read -r job_id job_name dataset splitter partition_path run_name slurm_script; do
  job_id="${job_id//$'\r'/}"
  job_name="${job_name//$'\r'/}"
  dataset="${dataset//$'\r'/}"
  splitter="${splitter//$'\r'/}"
  partition_path="${partition_path//$'\r'/}"
  run_name="${run_name//$'\r'/}"
  slurm_script="${slurm_script//$'\r'/}"
  if [[ "${job_id}" == "job_id" ]]; then
    continue
  fi
  if [[ -z "${slurm_script}" || ! -f "${slurm_script}" ]]; then
    echo "[skip] missing slurm script for ${job_name}: ${slurm_script}" >&2
    continue
  fi

  if n_clients="$(infer_n_clients_from_partition_path "${partition_path}")"; then
    ensure_script_has_n_clients "${slurm_script}" "${n_clients}"
  else
    echo "[warn] could not infer n_clients from partition path for ${job_name}: ${partition_path}" >&2
  fi
  ensure_script_has_conda_env "${slurm_script}" "${CONDA_ENV}"

  session=$((SESSION_START + idx))
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux kill-session -t "${session}" || true
  fi

  IFS=',' read -r partition gres mem <<< "$(choose_partition "${dataset}" "${idx}" "${n_clients}")"
  log_file="${LOG_DIR}/${session}_${job_name}.log"

  inner_cmd="cd ${REPO_ROOT} && source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && source scripts/hpc_session_env.sh >/dev/null && exec > >(tee -a ${log_file}) 2>&1 && echo [START] \$(date -Is) session=${session} job=${job_name} partition=${partition} && srun -p ${partition} --job-name=${job_name} --gres=${gres} --cpus-per-task=${CPUS_PER_TASK} --mem=${mem} --time=${TIME_LIMIT} --pty bash -lc 'set -e; cd ${REPO_ROOT}; source ${CONDA_BASE}/etc/profile.d/conda.sh; conda activate ${CONDA_ENV}; source scripts/hpc_session_env.sh >/dev/null; bash ${slurm_script}' ; rc=\$?; echo [END] \$(date -Is) rc=\$rc; exec bash -l"

  tmux new-session -d -s "${session}" "${inner_cmd}"
  echo "${session},${job_name},${dataset},${partition},${gres},${mem},${slurm_script},${log_file}" >> "${OUT_MANIFEST}"
  echo "[tmux] session=${session} job=${job_name} partition=${partition}"

  idx=$((idx + 1))
done < "${MANIFEST_PATH}"

echo ""
echo "Launched ${idx} tmux sessions."
echo "Run dir: ${RUN_DIR}"
echo "Launch manifest: ${OUT_MANIFEST}"
echo "Attach example: tmux a -t ${SESSION_START}"
echo "List sessions: tmux ls"
