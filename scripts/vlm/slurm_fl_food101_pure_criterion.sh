#!/bin/bash
#SBATCH --job-name=fds_fl_food_pure
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
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

CRITERIA=(cuisine formality preparation texture composition)
N_CLIENTS=(5 3 5 4 4)

CRITERION=${CRITERIA[$SLURM_ARRAY_TASK_ID]}
N=${N_CLIENTS[$SLURM_ARRAY_TASK_ID]}

PURE_PARTITIONS=/path/to/gtai/FDS/vlm_analysis/pure_criterion/food101
SCRATCH_OUTPUT=/path/to/data/fds
mkdir -p "$SCRATCH_OUTPUT"

export food101_chosen_split_path=$PURE_PARTITIONS/pure_${CRITERION}.json

echo "Running FL: food101 | pure_${CRITERION} | n_clients=${N}"
python fl_training/run_experiment.py \
    dataset=food101 \
    model=convnext_small \
    fed.n_clients=${N} \
    fed.rounds=100 \
    training.lr_min=3.22e-07 \
    "name=food101_convnext_small_fedavg_pure_${CRITERION}_seed42" \
    "paths.output_dir=${SCRATCH_OUTPUT}"
