#!/bin/bash
#SBATCH --job-name=fds_cache
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=47G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/cache_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/cache_%A_%a.err
#SBATCH --array=0-1

cd /path/to/FDS

export DATASETS_ROOT=/path/to/gtai/FDS
export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export FDS_CACHE_DIR=/path/to/gtai/FDS/embedding_cache
export http_proxy=http://proxy.example.org:3128
export HTTP_PROXY=http://proxy.example.org:3128
export https_proxy=http://proxy.example.org:3128
export HTTPS_PROXY=http://proxy.example.org:3128

# Tasks:
# 0 = pascal_voc + dinov2-base
# 1 = pascal_voc + inception_v3
# 2 = ade20k + inception_v3
DATASETS=(pascal_voc pascal_voc ade20k)
MODELS=(dinov2-base inception_v3 inception_v3)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Caching embeddings: dataset=$DATASET model=$MODEL"

python embedding/cache_embeddings.py \
    --dataset $DATASET \
    --base_path $DATASETS_ROOT \
    --model $MODEL \
    --cache_dir $FDS_CACHE_DIR \
    --batch_size 64 \
    --tokens cls \
    --k_last 4

echo "Done: $DATASET / $MODEL"
