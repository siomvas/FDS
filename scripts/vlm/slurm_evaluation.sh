#!/bin/bash
#SBATCH --job-name=fds_eval
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=47G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/eval_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/eval_%A_%a.err
#SBATCH --array=0-3

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

DATASETS=(food101 tiny_imagenet ade20k pascal_voc)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

BASE=/path/to/gtai/FDS
FEATURE_MODEL=${FEATURE_MODEL:-dinov2-base}
OUTPUT_BASE=${OUTPUT_BASE:-/path/to/gtai/FDS/vlm_eval/dinov2}

echo "Running eval: dataset=$DATASET feature_model=$FEATURE_MODEL"

python evaluation/evaluate_partitions.py \
    --partition_dir $DATASETS_ROOT/vlm_partitions/$DATASET \
    --dataset $DATASET \
    --base_path $BASE \
    --output $OUTPUT_BASE/$DATASET \
    --feature_model $FEATURE_MODEL \
    --cache_dir $FDS_CACHE_DIR \
    --subsample 0

echo "Generating report for $DATASET..."
python scripts/generate_report.py \
    --input $OUTPUT_BASE/$DATASET \
    --output $OUTPUT_BASE/$DATASET/report.html \
    --dataset $DATASET

echo "Done: $DATASET"
