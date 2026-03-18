#!/bin/bash
BASE=/path/to/gtai/FDS
export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export http_proxy=http://proxy.example.org:3128
export HTTP_PROXY=http://proxy.example.org:3128
export https_proxy=http://proxy.example.org:3128
export HTTPS_PROXY=http://proxy.example.org:3128

cd /path/to/FDS

for dataset in ade20k pascal_voc; do
    echo "Evaluating $dataset..."
    python evaluation/evaluate_partitions.py \
        --partition_dir ./partitions/$dataset \
        --dataset $dataset \
        --base_path $BASE \
        --output ./results/$dataset \
        --feature_model clip-base-patch16 \q
        --subsample 500
done

echo "Done. Generating reports..."
for dataset in food101 tiny_imagenet ade20k pascal_voc; do
    python scripts/generate_report.py \
        --input ./results/$dataset \
        --output ./results/$dataset/report.html \
        --dataset $dataset
done
echo "Reports written to results/<dataset>/report.html"
