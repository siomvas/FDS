#!/bin/bash
#SBATCH --job-name=fds_fl_tiny
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=47G
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

export tiny_imagenet_chosen_split_path=$PARTITIONS/tiny_imagenet/vlm_n10_alpha${ALPHA}_seed42.json

echo "Running FL: tiny_imagenet | vlm | alpha=$ALPHA"
python fl_training/run_experiment.py \
    dataset=tiny_imagenet \
    model=resnet18 \
    fed.n_clients=10 \
    fed.rounds=100 \
    training.lr=0.011318289829437791 \
    training.batch_size=32 \
    training.scheduler=cosine \
    "training.optimizer.class_name=SGD" \
    training.optimizer.args.momentum=0.9347723825763277 \
    training.optimizer.args.weight_decay=4.501174247598817e-06 \
    "name=tiny_imagenet_resnet18_fedavg_vlm_alpha${ALPHA}_tuned_seed42" \
    "paths.output_dir=${SCRATCH_OUTPUT}"
