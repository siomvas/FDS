#!/bin/bash
BASE=/path/to/hyperion1-homedir/gtai/FDS
export HF_HOME=/path/to/hyperion1-homedir/.cache/huggingface
export TORCH_HOME=/path/to/hyperion1-homedir/.cache/torch
export TRANSFORMERS_OFFLINE=1

for dataset in food101 tiny_imagenet ade20k pascal_voc; do
    python evaluation/evaluate_partitions.py \
        --partition_dir ./partitions/$dataset \
        --dataset $dataset \
        --base_path $BASE \
        --output_dir ./results/$dataset \
        --feature_model clip-base-patch16 \
        --cache_dir $BASE/embedding_cache \
        --subsample 500 \
        --force
done
