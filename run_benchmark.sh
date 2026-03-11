#!/usr/bin/env bash
# ==============================================================================
# FDS Benchmark Pipeline
# Links generation, evaluation, and reporting into a single workflow.
#
# Canonical usage:
#   ./run_benchmark.sh --dataset tiny_imagenet --base_path /path/to/data
#
# Backward-compatible usage:
#   ./run_benchmark.sh tiny_imagenet /path/to/embedding_partitions/tiny_imagenet
# ==============================================================================

set -euo pipefail

DATASET="tiny_imagenet"
BASE_DATA_PATH="${FDS_BASE_PATH:-}"
CACHE_DIR="${FDS_CACHE_DIR:-}"
EMBEDDING_DIR=""
OUTPUT_ROOT=""
SEED=42
DRY_RUN="false"

# Backward-compatible positional args.
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    DATASET="$1"
    shift
fi
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    EMBEDDING_DIR="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --base_path)
            BASE_DATA_PATH="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --embedding_dir)
            EMBEDDING_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage:
  ./run_benchmark.sh [dataset] [embedding_dir] [options]

Options:
  --dataset <name>         Dataset name (default: tiny_imagenet)
  --base_path <path>       Dataset root path (or set FDS_BASE_PATH)
  --cache_dir <path>       Embedding cache path (default: <base_path>/embedding_cache)
  --embedding_dir <path>   Embedding partitions directory
  --output <path>          Output root directory (default: ./benchmark_results/<dataset>)
  --seed <int>             Seed for baseline generation/evaluation (default: 42)
  --dry-run                Print planned commands without executing
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${BASE_DATA_PATH}" ]]; then
    echo "Error: --base_path (or FDS_BASE_PATH) is required." >&2
    exit 1
fi

if [[ -z "${CACHE_DIR}" ]]; then
    CACHE_DIR="${BASE_DATA_PATH}/embedding_cache"
fi

if [[ -z "${EMBEDDING_DIR}" ]]; then
    EMBEDDING_DIR="${BASE_DATA_PATH}/embedding_partitions/${DATASET}"
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="./benchmark_results/${DATASET}"
fi

BASELINE_DIR="${OUTPUT_ROOT}/partitions/baselines"
RESULTS_DIR="${OUTPUT_ROOT}/evaluation"
REPORT_FILE="${OUTPUT_ROOT}/report.html"

# Evaluation Settings
FEATURE_MODEL="inception_v3"
SUBSAMPLE=0
BATCH_SIZE=100

# VLM Settings
VLM_LABELS_DIR="partitioning/vlm/data/${DATASET}"

# Script locations
GENERATE_BASELINES="scripts/generate_baselines.py"
EVALUATE_PARTITIONS="evaluation/evaluate_partitions.py"
GENERATE_REPORT="scripts/generate_report.py"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[x]${NC} $1"; }

run_cmd() {
    log_info "CMD: $*"
    if [[ "${DRY_RUN}" == "true" ]]; then
        return 0
    fi
    "$@"
}

check_python_script() {
    if [[ ! -f "$1" ]]; then
        log_error "Script not found: $1"
        exit 1
    fi
}

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}FDS Benchmark Pipeline${NC}"
echo -e "${BLUE}====================================================================${NC}"
log_info "Dataset:          ${DATASET}"
log_info "Base Data Path:   ${BASE_DATA_PATH}"
log_info "Cache Dir:        ${CACHE_DIR}"
log_info "Embeddings From:  ${EMBEDDING_DIR}"
log_info "Output Directory: ${OUTPUT_ROOT}"
log_info "Seed:             ${SEED}"
if [[ "${DRY_RUN}" == "true" ]]; then
    log_warn "Dry-run enabled: commands will not execute."
fi
echo ""

check_python_script "${GENERATE_BASELINES}"
check_python_script "${EVALUATE_PARTITIONS}"
check_python_script "${GENERATE_REPORT}"

if [[ "${DRY_RUN}" != "true" ]]; then
    mkdir -p "${BASELINE_DIR}" "${RESULTS_DIR}"
fi

echo -e "${GREEN}[1/5] Generating baselines...${NC}"
VLM_ARGS=()
if [[ -d "${VLM_LABELS_DIR}" ]]; then
    VLM_ARGS=("--cluster_labels_dir" "${VLM_LABELS_DIR}")
    log_info "VLM labels found."
else
    log_warn "VLM labels not found at ${VLM_LABELS_DIR} (skipping VLM)."
fi

run_cmd python "${GENERATE_BASELINES}" \
    --dataset "${DATASET}" \
    --base_path "${BASE_DATA_PATH}" \
    --output "${BASELINE_DIR}" \
    --n_clients 5 10 \
    --alphas 0.1 0.5 \
    --seed "${SEED}" \
    "${VLM_ARGS[@]}"
log_success "Baselines generated"

echo -e "${GREEN}[2/5] Evaluating baselines...${NC}"
run_cmd python "${EVALUATE_PARTITIONS}" \
    --partition_dir "${BASELINE_DIR}" \
    --dataset "${DATASET}" \
    --base_path "${BASE_DATA_PATH}" \
    --output "${RESULTS_DIR}" \
    --feature_model "${FEATURE_MODEL}" \
    --subsample "${SUBSAMPLE}" \
    --cache_dir "${CACHE_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}"
log_success "Baselines evaluated"

echo -e "${GREEN}[3/5] Evaluating embedding partitions...${NC}"
if [[ -d "${EMBEDDING_DIR}" ]]; then
    run_cmd python "${EVALUATE_PARTITIONS}" \
        --partition_dir "${EMBEDDING_DIR}" \
        --dataset "${DATASET}" \
        --base_path "${BASE_DATA_PATH}" \
        --output "${RESULTS_DIR}" \
        --feature_model "${FEATURE_MODEL}" \
        --subsample "${SUBSAMPLE}" \
        --cache_dir "${CACHE_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}"
    log_success "Embedding partitions evaluated"
else
    log_warn "Embedding directory not found (skipping)"
fi

echo -e "${GREEN}[4/5] Building canonical results index...${NC}"
if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "CMD: python - ${RESULTS_DIR}  # write canonical results_index.*"
else
    python - "${RESULTS_DIR}" <<'PY'
from pathlib import Path
import sys

from research import write_results_index

results_dir = Path(sys.argv[1])
rows = write_results_index(results_dir, result_type="partition_eval")
if not rows:
    raise SystemExit(f"No canonical partition_eval artifacts found in {results_dir}")
print(f"Indexed {len(rows)} canonical runs in {results_dir}")
PY
fi

echo -e "${GREEN}[5/5] Generating report...${NC}"
if [[ "${DRY_RUN}" == "true" ]]; then
    log_info "CMD: python ${GENERATE_REPORT} --input ${RESULTS_DIR}/results_index.csv --output ${REPORT_FILE} --dataset ${DATASET}"
elif [[ -f "${RESULTS_DIR}/results_index.csv" ]]; then
    run_cmd python "${GENERATE_REPORT}" \
        --input "${RESULTS_DIR}/results_index.csv" \
        --output "${REPORT_FILE}" \
        --dataset "${DATASET}"
    log_success "Report generated: ${REPORT_FILE}"
else
    log_error "No canonical results_index.csv found. Report generation skipped."
    exit 1
fi
