#!/bin/bash
export DATASETS_ROOT=/path/to/hyperion1-homedir/gtai/FDS
export HF_HOME=/path/to/hyperion1-homedir/.cache/huggingface
export TORCH_HOME=/path/to/hyperion1-homedir/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export http_proxy=http://proxy.example.org:3128
export HTTP_PROXY=http://proxy.example.org:3128
export https_proxy=http://proxy.example.org:3128
export HTTPS_PROXY=http://proxy.example.org:3128

PARTITIONS=/path/to/FDS/partitions

# --- food101 (classification) ---
for alpha in 0.1 0.3 0.5 1.0; do
    export food101_chosen_split_path=$PARTITIONS/food101/vlm_n10_alpha${alpha}_seed42.json
    echo "Running FL: food101 | vlm | alpha=$alpha"
    python fl_training/run_experiment.py \
        dataset=food101 \
        model=resnet18 \
        fed.n_clients=10 \
        "name=food101_resnet18_fedavg_vlm_alpha${alpha}_seed42"
done

# # --- tiny_imagenet (classification) ---
# for alpha in 0.1 0.3 0.5 1.0 5.0; do
#     export tiny_imagenet_chosen_split_path=$PARTITIONS/tiny_imagenet/vlm_n10_alpha${alpha}_seed42.json
#     echo "Running FL: tiny_imagenet | vlm | alpha=$alpha"
#     python fl_training/run_experiment.py \
#         dataset=tiny_imagenet \
#         model=resnet18 \
#         fed.n_clients=10 \
#         "name=tiny_imagenet_resnet18_fedavg_vlm_alpha${alpha}_seed42"
# done

# # --- ade20k (segmentation) ---
# for alpha in 0.1 0.3 0.5 1.0 5.0; do
#     export ade20k_chosen_split_path=$PARTITIONS/ade20k/vlm_n10_alpha${alpha}_seed42.json
#     echo "Running FL: ade20k | vlm | alpha=$alpha"
#     python fl_training/run_experiment.py \
#         dataset=ade20k \
#         model=deeplabv3_resnet50 \
#         fed.n_clients=10
# done

# # --- pascal_voc (detection) ---
# for alpha in 0.1 0.3 0.5 1.0 5.0; do
#     export pascal_voc_chosen_split_path=$PARTITIONS/pascal_voc/vlm_n10_alpha${alpha}_seed42.json
#     echo "Running FL: pascal_voc | vlm | alpha=$alpha"
#     python fl_training/run_experiment.py \
#         dataset=pascal_voc \
#         model=fasterrcnn_mobilenet_v3 \
#         fed.n_clients=10
# done
