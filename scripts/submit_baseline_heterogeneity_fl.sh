#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/path/to/data/FDS"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PARTITION="${PARTITION:-preemptgpu}"
GRES="${GRES:-gpu:a100_80g:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/local_artifacts/experiments/baselines"

cmd="cd ${REPO_ROOT} && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate new && bash scripts/run_baseline_heterogeneity_fl.sh"

job_id="$(sbatch --parsable \
  -p "${PARTITION}" \
  --gres="${GRES}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  --job-name="baseline_prep_${STAMP}" \
  --output="${REPO_ROOT}/local_artifacts/experiments/baselines/baseline_prep_${STAMP}.%j.out" \
  --wrap "${cmd}")"

echo "Submitted baseline prep job: ${job_id}"
echo "Track with: squeue -j ${job_id}"
