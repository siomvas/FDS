#!/bin/bash
#SBATCH --job-name=fds_fl_voc_pure
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=47G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/fl_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/fl_%A_%a.err
#SBATCH --array=0-4

cd /path/to/FDS

export DATASETS_ROOT=/path/to/gtai/FDS
export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export http_proxy=http://proxy.example.org:3128
export HTTP_PROXY=http://proxy.example.org:3128
export https_proxy=http://proxy.example.org:3128
export HTTPS_PROXY=http://proxy.example.org:3128

CRITERIA=(color light location main_object size)
N_CLIENTS=(6 4 3 6 4)

CRITERION=${CRITERIA[$SLURM_ARRAY_TASK_ID]}
N=${N_CLIENTS[$SLURM_ARRAY_TASK_ID]}

PURE_PARTITIONS=/path/to/gtai/FDS/vlm_analysis/pure_criterion/pascal_voc
SCRATCH_OUTPUT=/path/to/data/fds
mkdir -p "$SCRATCH_OUTPUT"

export pascal_voc_chosen_split_path=$PURE_PARTITIONS/pure_${CRITERION}.json

echo "Running FL: pascal_voc | pure_${CRITERION} | n_clients=${N}"
python fl_training/run_experiment.py \
    dataset=pascal_voc \
    model=fasterrcnn_resnet50_fpn_v2 \
    fed.n_clients=${N} \
    fed.rounds=80 \
    ++dataset.args.num_workers=0 \
    "name=pascal_voc_fasterrcnn_resnet50_fpn_v2_fedavg_pure_${CRITERION}_seed42" \
    "paths.output_dir=${SCRATCH_OUTPUT}"
