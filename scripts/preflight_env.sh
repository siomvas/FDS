#!/usr/bin/env bash
set -u

status=0

run_check() {
  local description="$1"
  shift
  echo "[CHECK] ${description}"
  if "$@"; then
    echo "[PASS] ${description}"
  else
    echo "[FAIL] ${description}"
    status=1
  fi
  echo
}

echo "FDS environment preflight"
echo "Working directory: $(pwd)"
echo

run_check "Python interpreter available" python3 --version

run_check "Core stdlib import (csv)" \
  python3 -c "import csv"

run_check "Data loading package import" \
  python3 -c "import data_loading.dataset; import data_loading.data_modules"

run_check "Partitioning package import" \
  python3 -c "import partitioning"

run_check "Embedding package import" \
  python3 -c "import embedding"

run_check "Evaluation package import" \
  python3 -c "from evaluation import metrics"

run_check "Pytest collection preflight" \
  python3 -m pytest --collect-only -q

if [ "${status}" -ne 0 ]; then
  echo "Preflight failed. Install required extras and re-run:"
  echo "  pip install -e \".[partitioning,embedding,evaluation,fl,dev]\""
else
  echo "Preflight passed."
fi

exit "${status}"
