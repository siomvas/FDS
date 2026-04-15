#!/bin/bash
#SBATCH --job-name=fds_scaffold_ade
#SBATCH --partition=preemptgpu
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/path/to/FDS/logs/fl_%A_%a.out
#SBATCH --error=/path/to/FDS/logs/fl_%A_%a.err
#SBATCH --array=0-2

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

SPLITS_ROOT=/path/to/gtai/FDS/baselines/splits/ade20k
SCRATCH_OUTPUT=/path/to/data/fds
mkdir -p "$SCRATCH_OUTPUT"

PARTITIONS=(random_n5_seed42 random_n10_seed42 random_n20_seed42)
N_CLIENTS=(5 10 20)

PARTITION=${PARTITIONS[$SLURM_ARRAY_TASK_ID]}
N=${N_CLIENTS[$SLURM_ARRAY_TASK_ID]}

export ade20k_chosen_split_path=$SPLITS_ROOT/${PARTITION}.json

# Switch to steps mode so all n values complete within 72h wall time.
# 1 epoch ≈ 55 min/round × 100 rounds = 92h (exceeds limit for n=5; worse for n=10/n=20).
# 400 steps/round ≈ 11 min/round for n=5, 37h total; n=10 ≈ 67h; n=20 ≈ 73h.
EXTRA_ARGS="training.steps_or_epochs=steps training.local_steps=400"
[[ "${N}" == "20" ]] && EXTRA_ARGS="${EXTRA_ARGS} +dataset.args.num_workers=0"

echo "Running SCAFFOLD: ade20k | ${PARTITION} | n_clients=${N}"
python fl_training/run_experiment.py \
    dataset=ade20k \
    model=deeplabv3_resnet50 \
    fed.n_clients=${N} \
    fed.rounds=100 \
    fed.method=scaffold \
    "name=ade20k_deeplabv3_resnet50_scaffold_${PARTITION}" \
    "paths.output_dir=${SCRATCH_OUTPUT}" \
    ${EXTRA_ARGS}
