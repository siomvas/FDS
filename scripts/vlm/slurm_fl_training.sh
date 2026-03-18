#!/bin/bash
#SBATCH --job-name=fds_fl_vlm
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=47G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/fl_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/fl_%A_%a.err
#SBATCH --array=2-3

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

PARTITIONS=/path/to/FDS/partitions

# food101
export food101_chosen_split_path=$PARTITIONS/food101/vlm_n10_alpha${ALPHA}_seed42.json
SCRATCH_OUTPUT=/path/to/data/fds
mkdir -p "$SCRATCH_OUTPUT"

echo "Running FL: food101 | vlm | alpha=$ALPHA"
python fl_training/run_experiment.py \
    dataset=food101 \
    model=resnet18 \
    fed.n_clients=10 \
    "name=food101_resnet18_fedavg_vlm_alpha${ALPHA}_seed42" \
    "paths.output_dir=${SCRATCH_OUTPUT}"
