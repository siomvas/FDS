#!/bin/bash
#SBATCH --job-name=fds_vlm_food
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:a100_80g:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/vlm_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/vlm_%A_%a.err

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

echo "Running VLM step3: food101 | composition | test split"
python partitioning/vlm/step3.py \
    --dataset food101 \
    --criteria composition \
    --exp_name test
