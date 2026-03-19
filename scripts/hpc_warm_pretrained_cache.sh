#!/usr/bin/env bash
set -euo pipefail

cd /path/to/data/FDS
source scripts/hpc_session_env.sh

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate new
fi

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

python - <<'PY'
import os
import timm

models = [
    "resnet18",
]

print("[warm-cache] HF_HOME=", os.environ.get("HF_HOME"))
for model_name in models:
    print(f"[warm-cache] warming timm/{model_name} pretrained weights")
    timm.create_model(model_name, pretrained=True)
print("[warm-cache] done")
PY
