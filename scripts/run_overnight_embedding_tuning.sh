#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/data/FDS}"
BASE_PATH="${BASE_PATH:-/path/to/gtai/FDS}"
TRIALS="${TRIALS:-200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-28800}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-100}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-128}"
PARTITION_NAME="${PARTITION_NAME:-gtai}"
GRES="${GRES:-gpu:a100_80g:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-10:00:00}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${REPO_ROOT}/local_artifacts/experiments/embedding_tuning/${STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
STUDY_DIR="${RUN_ROOT}/studies"
mkdir -p "${LOG_DIR}" "${STUDY_DIR}"

if [[ "$#" -gt 0 ]]; then
  PRESET_KEYS=("$@")
else
  PRESET_KEYS=(
    "tiny_imagenet_embedding_fixed"
    "food101_embedding_fixed"
    "ade20k_embedding_fixed"
    "PascalVOC_embedding_fixed"
  )
fi

cd "${REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate new
source scripts/hpc_session_env.sh >/dev/null

launch_job() {
  local session="$1"
  local preset_key="$2"
  local log_file="${LOG_DIR}/${preset_key}.log"
  local job_name="embed_${preset_key}"

  local inner_cmd="cd ${REPO_ROOT} && source scripts/hpc_session_env.sh >/dev/null && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate new && exec > >(tee -a ${log_file}) 2>&1 && echo [START] \$(date -Is) preset=${preset_key} && srun -p ${PARTITION_NAME} --job-name=${job_name} --gres=${GRES} --cpus-per-task=${CPUS_PER_TASK} --mem=${MEMORY} --time=${TIME_LIMIT} --pty bash -lc 'set -e; cd ${REPO_ROOT}; source scripts/hpc_session_env.sh >/dev/null; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate new; python scripts/tune_embedding_preset.py ${preset_key} --base_path ${BASE_PATH} --study_dir ${STUDY_DIR} --n_trials ${TRIALS} --timeout ${TIMEOUT_SECONDS} --num_workers ${NUM_WORKERS} --seed ${SEED} --embed_batch_size ${EMBED_BATCH_SIZE} --metric_batch_size ${METRIC_BATCH_SIZE}' ; rc=\$?; echo [END] \$(date -Is) rc=\$rc; exec bash -l"

  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" "${inner_cmd}"
}

manifest_entries=()
session=1
for preset_key in "${PRESET_KEYS[@]}"; do
  launch_job "${session}" "${preset_key}"
  manifest_entries+=("\"${preset_key}\": \"${LOG_DIR}/${preset_key}.log\"")
  session=$((session + 1))
done

{
  echo "{"
  echo "  \"timestamp\": \"${STAMP}\","
  echo "  \"base_path\": \"${BASE_PATH}\","
  echo "  \"trials\": ${TRIALS},"
  echo "  \"timeout_seconds\": ${TIMEOUT_SECONDS},"
  echo "  \"seed\": ${SEED},"
  echo "  \"num_workers\": ${NUM_WORKERS},"
  echo "  \"study_dir\": \"${STUDY_DIR}\","
  echo "  \"presets\": {"
  for idx in "${!manifest_entries[@]}"; do
    suffix=","
    if [[ "${idx}" -eq "$((${#manifest_entries[@]} - 1))" ]]; then
      suffix=""
    fi
    echo "    ${manifest_entries[$idx]}${suffix}"
  done
  echo "  }"
  echo "}"
} > "${RUN_ROOT}/run_manifest.json"

echo "Overnight embedding tuning launched."
echo "Run root: ${RUN_ROOT}"
echo "Study dir: ${STUDY_DIR}"
echo "Attach: tmux a -t 1|2|3|..."
echo "Morning summary: python scripts/check_overnight_embedding_tuning.py --run-root ${RUN_ROOT}"
