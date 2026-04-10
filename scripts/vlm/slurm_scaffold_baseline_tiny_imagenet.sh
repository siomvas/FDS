#!/bin/bash
#SBATCH --job-name=fds_scaffold_tiny
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=47G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/fl_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/fl_%A_%a.err
#SBATCH --array=0-11

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
SCRATCH_OUTPUT=/path/to/data/fds
mkdir -p "$SCRATCH_OUTPUT"

PARTITIONS=(
  random_n5_seed42
  random_n10_seed42
  random_n20_seed42
  dirichlet_n5_alpha0.1_seed42
  dirichlet_n5_alpha0.5_seed42
  dirichlet_n5_alpha1.0_seed42
  dirichlet_n10_alpha0.1_seed42
  dirichlet_n10_alpha0.5_seed42
  dirichlet_n10_alpha1.0_seed42
  dirichlet_n20_alpha0.1_seed42
  dirichlet_n20_alpha0.5_seed42
  dirichlet_n20_alpha1.0_seed42
)
N_CLIENTS=(5 10 20 5 5 5 10 10 10 20 20 20)

PARTITION=${PARTITIONS[$SLURM_ARRAY_TASK_ID]}
N=${N_CLIENTS[$SLURM_ARRAY_TASK_ID]}

export tiny_imagenet_chosen_split_path=$SPLITS_ROOT/${PARTITION}.json

EXTRA_ARGS=""
[[ "${N}" == "20" ]] && EXTRA_ARGS="+dataset.args.num_workers=0"

echo "Running SCAFFOLD: tiny_imagenet | ${PARTITION} | n_clients=${N}"
python fl_training/run_experiment.py \
    dataset=tiny_imagenet \
    model=resnet18 \
    fed.n_clients=${N} \
    fed.rounds=100 \
    fed.method=scaffold \
    training.lr=0.011318289829437791 \
    "name=tiny_imagenet_resnet18_scaffold_${PARTITION}" \
    "paths.output_dir=${SCRATCH_OUTPUT}" \
    ${EXTRA_ARGS}
