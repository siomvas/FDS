#!/bin/bash
#SBATCH --job-name=fds_fl_tiny_pure
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

CRITERIA=(color location main_object size time)
N_CLIENTS=(7 5 7 4 4)

CRITERION=${CRITERIA[$SLURM_ARRAY_TASK_ID]}
N=${N_CLIENTS[$SLURM_ARRAY_TASK_ID]}

PURE_PARTITIONS=/path/to/gtai/FDS/vlm_analysis/pure_criterion/tiny_imagenet
SCRATCH_OUTPUT=/path/to/data/fds
mkdir -p "$SCRATCH_OUTPUT"

export tiny_imagenet_chosen_split_path=$PURE_PARTITIONS/pure_${CRITERION}.json

echo "Running FL: tiny_imagenet | pure_${CRITERION} | n_clients=${N}"
python fl_training/run_experiment.py \
    dataset=tiny_imagenet \
    model=resnet18 \
    fed.n_clients=${N} \
    fed.rounds=100 \
    training.lr=0.011318289829437791 \
    training.batch_size=32 \
    training.scheduler=cosine \
    "training.optimizer.class_name=SGD" \
    training.optimizer.args.momentum=0.9347723825763277 \
    training.optimizer.args.weight_decay=4.501174247598817e-06 \
    "name=tiny_imagenet_resnet18_fedavg_pure_${CRITERION}_seed42" \
    "paths.output_dir=${SCRATCH_OUTPUT}"
