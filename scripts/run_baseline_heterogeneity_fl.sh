#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/path/to/data/FDS"
BASE_PATH="${BASE_PATH:-/path/to/gtai/FDS}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-100}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${REPO_ROOT}/local_artifacts/experiments/baselines/${STAMP}"
SPLIT_ROOT="${RUN_ROOT}/splits"
METRICS_ROOT="${RUN_ROOT}/metrics"
FL_ROOT="${RUN_ROOT}/fl"
SLURM_SCRIPT_ROOT="${RUN_ROOT}/slurm_scripts"
MANIFEST="${RUN_ROOT}/fl_job_manifest.csv"

mkdir -p "${SPLIT_ROOT}" "${METRICS_ROOT}" "${FL_ROOT}" "${SLURM_SCRIPT_ROOT}"

validate_script_n_clients() {
  local script_path="$1"
  local expected_n_clients="$2"
  if ! grep -q "fed.n_clients=${expected_n_clients}" "${script_path}"; then
    echo "Generated script missing expected fed.n_clients=${expected_n_clients}: ${script_path}" >&2
    return 1
  fi
}

cd "${REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_BASE="$(conda info --base)"
conda activate new
source scripts/hpc_session_env.sh >/dev/null

generate_random_only_splits() {
  local dataset="$1"
  local output_dir="$2"
  python - <<PY
import json
from pathlib import Path

from data_loading.dataset import BaseDataset
from partitioning.random_splitter import RandomSplitter

dataset_name = "${dataset}"
base_path = "${BASE_PATH}"
output_dir = Path("${output_dir}")
seed = int("${SEED}")

train_dataset = BaseDataset(dataset_name, base_path, "train")
val_dataset = BaseDataset(dataset_name, base_path, "val")
output_dir.mkdir(parents=True, exist_ok=True)

for n_clients in (5, 10, 20):
    splitter = RandomSplitter(train_dataset, val_dataset, n_clients, heterogeneity=0.0, seed=seed)
    payload = splitter._make_serializable(splitter.get_client_splits())
    out_path = output_dir / f"random_n{n_clients}_seed{seed}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved Random split to {out_path}")
PY
}

# 1) Generate required partition files
python scripts/generate_baselines.py \
  --dataset tiny_imagenet --base_path "${BASE_PATH}" --output "${SPLIT_ROOT}/tiny_imagenet" \
  --n_clients 5 10 20 --alphas 0.1 0.5 1.0 --seed "${SEED}"

python scripts/generate_baselines.py \
  --dataset food101 --base_path "${BASE_PATH}" --output "${SPLIT_ROOT}/food101" \
  --n_clients 5 10 20 --alphas 0.1 0.5 1.0 --seed "${SEED}"

generate_random_only_splits ade20k "${SPLIT_ROOT}/ade20k"
generate_random_only_splits pascal_voc "${SPLIT_ROOT}/pascal_voc"

# 2) Run heterogeneity metrics
python evaluation/evaluate_partitions.py \
  --partition_dir "${SPLIT_ROOT}/tiny_imagenet" \
  --dataset tiny_imagenet --base_path "${BASE_PATH}" --output "${METRICS_ROOT}/tiny_imagenet" \
  --feature_model inception_v3 --seed "${SEED}"

python evaluation/evaluate_partitions.py \
  --partition_dir "${SPLIT_ROOT}/food101" \
  --dataset food101 --base_path "${BASE_PATH}" --output "${METRICS_ROOT}/food101" \
  --feature_model inception_v3 --seed "${SEED}"

for partition_path in "${SPLIT_ROOT}"/ade20k/random_n*_seed"${SEED}".json; do
  python evaluation/evaluate_partitions.py \
    --partition "${partition_path}" \
    --dataset ade20k --base_path "${BASE_PATH}" --output "${METRICS_ROOT}/ade20k" \
    --feature_model inception_v3 --feature_only --seed "${SEED}"
done

for partition_path in "${SPLIT_ROOT}"/pascal_voc/random_n*_seed"${SEED}".json; do
  python evaluation/evaluate_partitions.py \
    --partition "${partition_path}" \
    --dataset pascal_voc --base_path "${BASE_PATH}" --output "${METRICS_ROOT}/pascal_voc" \
    --feature_model inception_v3 --feature_only --seed "${SEED}"
done

# 3) Submit FL jobs (30 separate jobs)
echo "job_id,job_name,dataset,splitter,partition_path,run_name,slurm_script" > "${MANIFEST}"

submit_fl_job() {
  local dataset="$1"
  local splitter="$2"
  local partition_path="$3"
  local partition_name
  partition_name="$(basename "${partition_path}" .json)"
  local n_clients=""
  if [[ "${partition_name}" =~ _n([0-9]+)_ ]]; then
    n_clients="${BASH_REMATCH[1]}"
  fi
  if [[ -z "${n_clients}" ]]; then
    echo "Unable to infer n_clients from partition name: ${partition_name}" >&2
    return 1
  fi
  local run_name="${dataset}_${partition_name}_seed${SEED}"
  local split_env_var="${dataset}_chosen_split_path"
  local gpu_overrides_line=""
  if [[ "${n_clients}" == "20" ]]; then
    gpu_overrides_line="  system.gpu_ids=[0,0,0,0,0,0,0,0,0,0] \\"
  fi
  local slurm_partition
  local slurm_gres
  local slurm_mem
  local dataset_overrides

  case "${dataset}" in
    tiny_imagenet|food101)
      slurm_partition="gtai"
      slurm_gres="gpu:a100_80g:1"
      slurm_mem="64G"
      ;;
    ade20k|pascal_voc)
      slurm_partition="preemptgpu"
      slurm_gres="gpu:a100_80g:1"
      slurm_mem="80G"
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      return 1
      ;;
  esac

  case "${dataset}" in
    tiny_imagenet)
      dataset_overrides=$(cat <<'EOS'
  model.name=resnet18 \
  model.provider=timm \
  model.args.pretrained=true \
  training.batch_size=32 \
  training.lr=0.011318289829437791 \
  training.scheduler=cosine \
  training.lr_min=0.00011318289829437791 \
  training.optimizer.class_name=SGD \
  training.optimizer.args.momentum=0.9347723825763277 \
  training.optimizer.args.weight_decay=4.501174247598817e-06
EOS
)
      ;;
    food101)
      dataset_overrides=$(cat <<'EOS'
  model.name=convnext_small \
  model.provider=timm \
  model.args.pretrained=true \
  training.batch_size=32 \
  training.lr=3.2247852759496085e-05 \
  training.scheduler=cosine \
  training.lr_min=3.224785275949608e-07 \
  training.optimizer.class_name=AdamW \
  ~training.optimizer.args.momentum \
  training.optimizer.args.weight_decay=2.629479738981522e-06
EOS
)
      ;;
    ade20k)
      dataset_overrides=$(cat <<'EOS'
  model.name=deeplabv3_resnet50 \
  model.provider=torchvision \
  +model.args.weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 \
  +model.args.weights_backbone=null \
  training.batch_size=4 \
  training.lr=4.462965193984174e-05 \
  training.scheduler=cosine \
  training.lr_min=4.462965193984174e-07 \
  training.optimizer.class_name=AdamW \
  ~training.optimizer.args.momentum \
  training.optimizer.args.weight_decay=0.00010234321271245017
EOS
)
      ;;
    pascal_voc)
      dataset_overrides=$(cat <<'EOS'
  model.name=fasterrcnn_resnet50_fpn_v2 \
  model.provider=torchvision \
  +model.args.weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 \
  +model.args.weights_backbone=null \
  training.batch_size=2 \
  training.lr=1.096580630626157e-05 \
  training.scheduler=none \
  training.optimizer.class_name=AdamW \
  ~training.optimizer.args.momentum \
  training.optimizer.args.weight_decay=0.0
EOS
)
      ;;
  esac

  local script_path="${SLURM_SCRIPT_ROOT}/${dataset}_${partition_name}.sh"
  cat > "${script_path}" <<EOS
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate new
source scripts/hpc_session_env.sh >/dev/null
export DATASETS_ROOT="${BASE_PATH}"
export ${split_env_var}="${partition_path}"
python fl_training/run_experiment.py --config-path conf --config-name config \
  dataset=${dataset} \
  seed=${SEED} \
  fed.n_clients=${n_clients} \
${gpu_overrides_line}
  fed.rounds=${ROUNDS} \
  training.local_steps=${LOCAL_STEPS} \
  training.steps_or_epochs=epochs \
  name=${run_name} \
  paths.output_dir=${FL_ROOT} \
${dataset_overrides}
EOS

  chmod +x "${script_path}"
  validate_script_n_clients "${script_path}" "${n_clients}"
  local job_name="base_${dataset}_${splitter}_$(basename "${partition_path}" .json)"
  local job_id
  job_id="$(sbatch --parsable -p "${slurm_partition}" --gres="${slurm_gres}" --cpus-per-task=8 --mem="${slurm_mem}" --time=24:00:00 --job-name="${job_name}" "${script_path}")"
  echo "${job_id},${job_name},${dataset},${splitter},${partition_path},${run_name},${script_path}" >> "${MANIFEST}"
  echo "[submitted] ${job_id} ${job_name}"
}

for n in 5 10 20; do
  submit_fl_job tiny_imagenet random "${SPLIT_ROOT}/tiny_imagenet/random_n${n}_seed${SEED}.json"
  submit_fl_job food101 random "${SPLIT_ROOT}/food101/random_n${n}_seed${SEED}.json"
  submit_fl_job ade20k random "${SPLIT_ROOT}/ade20k/random_n${n}_seed${SEED}.json"
  submit_fl_job pascal_voc random "${SPLIT_ROOT}/pascal_voc/random_n${n}_seed${SEED}.json"
done

for n in 5 10 20; do
  for alpha in 0.1 0.5 1.0; do
    submit_fl_job tiny_imagenet dirichlet "${SPLIT_ROOT}/tiny_imagenet/dirichlet_n${n}_alpha${alpha}_seed${SEED}.json"
    submit_fl_job food101 dirichlet "${SPLIT_ROOT}/food101/dirichlet_n${n}_alpha${alpha}_seed${SEED}.json"
  done
done

echo "Baseline workflow complete."
echo "RUN_ROOT=${RUN_ROOT}"
echo "SPLIT_ROOT=${SPLIT_ROOT}"
echo "METRICS_ROOT=${METRICS_ROOT}"
echo "FL_ROOT=${FL_ROOT}"
echo "MANIFEST=${MANIFEST}"
