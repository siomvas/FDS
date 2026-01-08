#!/bin/bash
# ==============================================================================
# FDS Benchmark Pipeline
# Links generation, evaluation, and reporting into a single workflow.
#
# Usage:
#   ./run_benchmark.sh <dataset_name> [embedding_partitions_dir]
#
# Example:
#   ./run_benchmark.sh tiny_imagenet ./embedding_partitions/tiny_imagenet
# ==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# --- Configuration -----------------------------------------------------------
DATASET="${1:-tiny_imagenet}"
EMBEDDING_DIR="${2:-./embedding_partitions/$DATASET}"

# Paths - Use environment variables with sensible defaults
BASE_DATA_PATH="${FDS_BASE_PATH:-/path/to/gtai/FDS}"
CACHE_DIR="${FDS_CACHE_DIR:-/path/to/gtai/FDS/embedding_cache}"

OUTPUT_ROOT="./benchmark_results/$DATASET"

# Derived Paths
BASELINE_DIR="$OUTPUT_ROOT/partitions/baselines"
RESULTS_DIR="$OUTPUT_ROOT/evaluation"
REPORT_FILE="$OUTPUT_ROOT/report.html"

# Evaluation Settings
FEATURE_MODEL="inception_v3"
SUBSAMPLE=1000
BATCH_SIZE=100

# VLM Settings
VLM_LABELS_DIR="partitioning/vlm/data/$DATASET"

# Script Locations
GENERATE_BASELINES="scripts/generate_baselines.py"
EVALUATE_PARTITIONS="evaluation/evaluate_partitions.py"
GENERATE_REPORT="scripts/generate_report.py"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# ==============================================================================
# Helper Functions
# ==============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

check_python_script() {
    if [[ ! -f "$1" ]]; then
        log_error "Script not found: $1"
        exit 1
    fi
}

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

echo -e "${BLUE}====================================================================${NC}"
echo -e "${BLUE}FDS Benchmark Pipeline${NC}"
echo -e "${BLUE}====================================================================${NC}"
echo ""
log_info "Dataset:          $DATASET"
log_info "Base Data Path:   $BASE_DATA_PATH"
log_info "Embeddings From:  $EMBEDDING_DIR"
log_info "Output Directory: $OUTPUT_ROOT"
echo ""

check_python_script "$GENERATE_BASELINES"
check_python_script "$EVALUATE_PARTITIONS"
check_python_script "$GENERATE_REPORT"

mkdir -p "$BASELINE_DIR"
mkdir -p "$RESULTS_DIR"

# ==============================================================================
# Step 1: Generate Baselines
# ==============================================================================
echo ""
echo -e "${GREEN}[1/5] Generating Baselines...${NC}"

VLM_ARGS=()
if [[ -d "$VLM_LABELS_DIR" ]]; then
    VLM_ARGS=("--cluster_labels_dir" "$VLM_LABELS_DIR")
    log_info "VLM labels found."
else
    log_warn "VLM labels not found at $VLM_LABELS_DIR (skipping VLM)."
fi

python "$GENERATE_BASELINES" \
    --dataset "$DATASET" \
    --base_path "$BASE_DATA_PATH" \
    --output_dir "$BASELINE_DIR" \
    --n_clients 5 \
    --alphas 0.5 \
    --seeds 42 \
    "${VLM_ARGS[@]+"${VLM_ARGS[@]}"}"

log_success "Baselines generated"

# ==============================================================================
# Step 2: Evaluate Baselines
# ==============================================================================
echo ""
echo -e "${GREEN}[2/5] Evaluating Baselines...${NC}"

python "$EVALUATE_PARTITIONS" \
    --partition_dir "$BASELINE_DIR" \
    --dataset "$DATASET" \
    --base_path "$BASE_DATA_PATH" \
    --output_dir "$RESULTS_DIR" \
    --feature_model "$FEATURE_MODEL" \
    --subsample "$SUBSAMPLE" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$BATCH_SIZE"

log_success "Baselines evaluated"

# ==============================================================================
# Step 3: Evaluate Embeddings
# ==============================================================================
echo ""
echo -e "${GREEN}[3/5] Evaluating Embedding Partitions...${NC}"

if [[ -d "$EMBEDDING_DIR" ]]; then
    python "$EVALUATE_PARTITIONS" \
        --partition_dir "$EMBEDDING_DIR" \
        --dataset "$DATASET" \
        --base_path "$BASE_DATA_PATH" \
        --output_dir "$RESULTS_DIR" \
        --feature_model "$FEATURE_MODEL" \
        --subsample "$SUBSAMPLE" \
        --cache_dir "$CACHE_DIR" \
        --batch_size "$BATCH_SIZE"
    
    log_success "Embedding partitions evaluated"
else
    log_warn "Embedding directory not found (skipping)"
fi

# ==============================================================================
# Step 4: Aggregate Results
# ==============================================================================
echo ""
echo -e "${GREEN}[4/5] Aggregating Results...${NC}"

# FIX: Python script updated to handle FLAT json structure
python -c "
import json
import pandas as pd
from pathlib import Path
import sys

results_dir = Path('$RESULTS_DIR')
json_files = list(results_dir.glob('*_eval.json'))

if not json_files:
    print('No evaluation results found!')
    sys.exit(0) # Not an error, just empty

rows = []
for jf in json_files:
    try:
        with open(jf) as f:
            data = json.load(f)
            # FIX: Data is already flat, just append it
            rows.append(data)
    except Exception as e:
        print(f'Skipping corrupt file {jf}: {e}')

if rows:
    df = pd.DataFrame(rows)
    output_path = results_dir / 'all_results.csv'
    df.to_csv(output_path, index=False)
    print(f'Aggregated {len(rows)} results into {output_path}')
else:
    print('No valid results to aggregate.')
"

# ==============================================================================
# Step 5: Generate Report
# ==============================================================================
echo ""
echo -e "${GREEN}[5/5] Generating Interactive Report...${NC}"

if [[ -f "$RESULTS_DIR/all_results.csv" ]]; then
    python "$GENERATE_REPORT" \
        --input "$RESULTS_DIR/all_results.csv" \
        --output "$REPORT_FILE" \
        --dataset "$DATASET"
    
    log_success "Report generated"
    echo ""
    echo -e "${BLUE}====================================================================${NC}"
    echo "Report available at: $REPORT_FILE"
else
    log_error "No results CSV found. Report generation skipped."
    exit 1
fi