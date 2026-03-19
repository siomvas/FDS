#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/path/to/data/FDS"
BASE_PATH="${BASE_PATH:-/path/to/gtai/FDS}"
TRIALS="${TRIALS:-200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-28800}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${REPO_ROOT}/local_artifacts/experiments/tuning/${STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
STUDY_DIR="${RUN_ROOT}/studies"
mkdir -p "${LOG_DIR}" "${STUDY_DIR}"

cd "${REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate new
source scripts/hpc_session_env.sh >/dev/null

MISSING_DEPS="$(python - <<'PY'
missing = []
for name in ("numpy", "optuna"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
print(" ".join(missing))
PY
)"

if [[ -n "${MISSING_DEPS}" ]]; then
  echo "[deps] Installing missing packages in env 'new': ${MISSING_DEPS}"
  python -m pip install ${MISSING_DEPS}
fi

for session in 1 2 3 4; do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

python - <<'PY'
import timm
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

print('[warm] timm/resnet18 pretrained')
_ = timm.create_model('resnet18', pretrained=True)
print('[warm] timm/convnext_small pretrained')
_ = timm.create_model('convnext_small', pretrained=True)
print('[warm] torchvision/deeplabv3_resnet50 pretrained')
_ = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, weights_backbone=None)
print('[warm] torchvision/fasterrcnn_resnet50_fpn_v2 pretrained')
_ = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, weights_backbone=None)
print('[warm] done')
PY

launch_job() {
  local session="$1"
  local partition="$2"
  local gres="$3"
  local job_name="$4"
  local tune_script="$5"
  local study_subdir="$6"
  local log_file="${LOG_DIR}/${study_subdir}.log"

  local inner_cmd="cd ${REPO_ROOT} && source scripts/hpc_session_env.sh >/dev/null && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate new && exec > >(tee -a ${log_file}) 2>&1 && echo [START] \$(date -Is) && srun -p ${partition} --job-name=${job_name} --gres=${gres} --cpus-per-task=8 --mem=64G --time=10:00:00 --pty bash -lc 'set -e; cd ${REPO_ROOT}; source scripts/hpc_session_env.sh >/dev/null; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate new; python ${tune_script} --base_path ${BASE_PATH} --study_dir ${STUDY_DIR} --n_trials ${TRIALS} --timeout ${TIMEOUT_SECONDS} --num_workers ${NUM_WORKERS} --seed ${SEED}' ; rc=\$?; echo [END] \$(date -Is) rc=\$rc; exec bash -l"

  tmux new-session -d -s "${session}" "${inner_cmd}"
}

launch_job 1 gtai "gpu:a100_80g:1" overnight_tune_tiny scripts/tune_tiny_imagenet_resnet18.py tiny_imagenet_resnet18
launch_job 2 gtai "gpu:a100_80g:1" overnight_tune_food scripts/tune_food101_convnext_small.py food101_convnext_small

if sinfo -h -o "%P" | sed 's/*//g' | grep -qx "gpu-a100"; then
  launch_job 3 gpu-a100 "gpu:a100:1" overnight_tune_ade scripts/tune_ade20k_deeplabv3_resnet50.py ade20k_deeplabv3_resnet50
else
  launch_job 3 preemptgpu "gpu:a100_80g:1" overnight_tune_ade scripts/tune_ade20k_deeplabv3_resnet50.py ade20k_deeplabv3_resnet50
fi

launch_job 4 preemptgpu "gpu:a100_80g:1" overnight_tune_voc scripts/tune_pascal_voc_fasterrcnn_resnet50_fpn_v2.py pascal_voc_fasterrcnn_resnet50_fpn_v2

cat > "${RUN_ROOT}/run_manifest.json" <<EOF
{
  "timestamp": "${STAMP}",
  "base_path": "${BASE_PATH}",
  "trials": ${TRIALS},
  "timeout_seconds": ${TIMEOUT_SECONDS},
  "seed": ${SEED},
  "num_workers": ${NUM_WORKERS},
  "study_dir": "${STUDY_DIR}",
  "logs": {
    "tiny": "${LOG_DIR}/tiny_imagenet_resnet18.log",
    "food": "${LOG_DIR}/food101_convnext_small.log",
    "ade": "${LOG_DIR}/ade20k_deeplabv3_resnet50.log",
    "voc": "${LOG_DIR}/pascal_voc_fasterrcnn_resnet50_fpn_v2.log"
  }
}
EOF

echo "Overnight tuning launched."
echo "Run root: ${RUN_ROOT}"
echo "Study dir: ${STUDY_DIR}"
echo "Attach: tmux a -t 1|2|3|4"
echo "Morning summary: python scripts/check_overnight_tuning.py --run-root ${RUN_ROOT}"
