#!/bin/bash
#SBATCH --job-name=fds_fl_voc
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/fl_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/fl_%A_%a.err
#SBATCH --array=0-3

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

ALPHAS=(0.1 0.3 0.5 1.0)
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

PARTITIONS=/path/to/gtai/FDS/vlm_partitions
SCRATCH_OUTPUT=/path/to/data/fds
mkdir -p "$SCRATCH_OUTPUT"

export pascal_voc_chosen_split_path=$PARTITIONS/pascal_voc/vlm_n10_alpha${ALPHA}_seed42.json

echo "Running FL: pascal_voc | vlm | alpha=$ALPHA"
python fl_training/run_experiment.py \
    dataset=pascal_voc \
    model=fasterrcnn_resnet50_fpn_v2 \
    fed.n_clients=10 \
    fed.rounds=80 \
    "name=pascal_voc_fasterrcnn_resnet50_fpn_v2_fedavg_vlm_alpha${ALPHA}_tuned_seed42" \
    "paths.output_dir=${SCRATCH_OUTPUT}"
