#!/bin/bash
#SBATCH --job-name=fds_vlm_food
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:a100_80g:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/vlm_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/vlm_%A_%a.err
#SBATCH --array=0-3

cd /path/to/FDS

export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export http_proxy=http://proxy.example.org:3128
export HTTP_PROXY=http://proxy.example.org:3128
export https_proxy=http://proxy.example.org:3128
export HTTPS_PROXY=http://proxy.example.org:3128
export PYTORCH_ALLOC_CONF=expandable_segments:True

CRITERIA=(cuisine formality preparation texture)
CRITERION=${CRITERIA[$SLURM_ARRAY_TASK_ID]}

echo "Running VLM full pipeline: food101 | ${CRITERION} | test split"
python partitioning/vlm/full_pipeline.py \
    --dataset food101 \
    --criteria ${CRITERION} \
    --exp_name test
