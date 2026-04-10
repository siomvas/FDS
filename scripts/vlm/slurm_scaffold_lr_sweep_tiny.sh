#!/bin/bash
#SBATCH --job-name=fds_scaf_lr
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/FDS/logs/fl_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/fl_%A_%a.err
#SBATCH --array=0-2

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

SPLITS_ROOT=/path/to/gtai/FDS/baselines/splits/tiny_imagenet
SCRATCH_OUTPUT=/path/to/data/fds_scaffold_lr_sweep
mkdir -p "$SCRATCH_OUTPUT"

export tiny_imagenet_chosen_split_path=$SPLITS_ROOT/random_n5_seed42.json

# Test original FedAvg LR, and two lower candidates
LRS=(0.011318 0.001 0.0002)
LR_MINS=(0.000113 0.00001 0.000002)

LR=${LRS[$SLURM_ARRAY_TASK_ID]}
LR_MIN=${LR_MINS[$SLURM_ARRAY_TASK_ID]}

NAME="tiny_imagenet_resnet18_scaffold_v2_lr${LR}"
echo "LR sweep v2: lr=$LR lr_min=$LR_MIN"

python fl_training/run_experiment.py \
    dataset=tiny_imagenet \
    model=resnet18 \
    fed.n_clients=5 \
    fed.rounds=20 \
    fed.method=scaffold \
    training.lr=${LR} \
    training.lr_min=${LR_MIN} \
    training.scheduler=cosine \
    "name=${NAME}" \
    "paths.output_dir=${SCRATCH_OUTPUT}"
