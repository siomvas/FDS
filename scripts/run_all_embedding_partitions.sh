#!/bin/bash
# Run all embedding partition combinations
# This script generates embedding-based partitions for all combinations of:
# - Extractors (foundation models)
# - Pooling strategies
# - Clustering methods

# =============================================================================
# CONFIGURATION - Edit these variables
# =============================================================================

# Global settings
MOCK=true                          # Set to true for quick testing with 1k samples
DATASET="tiny_imagenet"
BASE_PATH="/path/to/gtai/FDS/"           # UPDATE THIS PATH
OUTPUT_DIR="./embedding_partitions"
SEED=42
VALSET_TYPE="global"                  # global, local, or none

# N_CLUSTERS for kmeans/spectral clustering
N_CLUSTERS=10

# Component lists - Comment out items to exclude them
# Available extractors (supported by ModelRegistry):
#   dinov2: small, base, large, giant
#   clip: base-patch32, base-patch16, large-patch14
#   siglip: base-patch16-224, base-patch16-384, large-patch16-256
#   Others: inception_v3, mae-base, mae-large, mae-huge
EXTRACTORS=(
    "dinov2-small"
    "dinov2-base"
    # "dinov2-large"
    # "dinov2-giant"
    "clip-base-patch16"
    # "clip-base-patch32"
    # "clip-large-patch14"
    "siglip-base-patch16-224"
    # "siglip-base-patch16-384"
    # "siglip-large-patch16-256"
    # "inception_v3"
    "mae-base"
)

# Available pooling strategies:
#   cls: CLS token only (fastest)
#   mean: Average all tokens
#   max: Max pooling across tokens
#   gem: Generalized mean pooling
#   lastk: Aggregate last K layers
#   spectral: Spectral sharpening
POOLING_STRATEGIES=(
    "cls"
    "mean"
    # "max"
    # "gem"
    # "lastk"
    # "spectral"
)

# Available clustering methods:
#   hdbscan: Density-based (auto-determines n_clusters)
#   kmeans: K-means (requires n_clusters)
#   knn: k-NN graph connected components
#   spectral: Spectral clustering (requires n_clusters)
CLUSTERING_METHODS=(
    "hdbscan"
    "kmeans"
    # "knn"
    # "spectral"
)

# =============================================================================
# SCRIPT - Usually no need to edit below this line
# =============================================================================

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Count total combinations
total=$((${#EXTRACTORS[@]} * ${#POOLING_STRATEGIES[@]} * ${#CLUSTERING_METHODS[@]}))
current=0
success=0
failed=0
skipped=0

# Start time
start_time=$(date +%s)

echo "========================================================================"
echo "Embedding Partition Batch Generation"
echo "========================================================================"
echo "Dataset: $DATASET"
echo "Base path: $BASE_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Mock mode: $MOCK"
echo "Validation type: $VALSET_TYPE"
echo "Seed: $SEED"
echo "N_clusters (for kmeans/spectral): $N_CLUSTERS"
echo ""
echo "Total combinations: $total"
echo "  Extractors: ${#EXTRACTORS[@]}"
echo "  Pooling strategies: ${#POOLING_STRATEGIES[@]}"
echo "  Clustering methods: ${#CLUSTERING_METHODS[@]}"
echo "========================================================================"
echo ""

# Main loop
for extractor in "${EXTRACTORS[@]}"; do
    for pooling in "${POOLING_STRATEGIES[@]}"; do
        for clustering in "${CLUSTERING_METHODS[@]}"; do
            ((current++))

            # Build output filename
            output_name="${extractor}_${pooling}_${clustering}.json"
            output_path="${OUTPUT_DIR}/${output_name}"
            log_path="${OUTPUT_DIR}/logs/${extractor}_${pooling}_${clustering}.log"

            # Skip if already exists
            if [[ -f "$output_path" ]]; then
                echo "[$current/$total] SKIP (exists): $output_name"
                ((success++))
                ((skipped++))
                continue
            fi

            echo "[$current/$total] Running: $extractor + $pooling + $clustering"

            # Build command
            cmd="python generate_embedding_partitions.py"
            cmd="$cmd --dataset $DATASET"
            cmd="$cmd --base_path $BASE_PATH"
            cmd="$cmd --extractor $extractor"
            cmd="$cmd --pooling $pooling"
            cmd="$cmd --clustering $clustering"
            cmd="$cmd --valset_type $VALSET_TYPE"
            cmd="$cmd --seed $SEED"
            cmd="$cmd --output $output_path"

            # Add n_clusters for kmeans/spectral
            if [[ "$clustering" == "kmeans" || "$clustering" == "spectral" ]]; then
                cmd="$cmd --n_clusters $N_CLUSTERS"
            fi

            # Add mock flag if enabled
            if [[ "$MOCK" == true ]]; then
                cmd="$cmd --mock"
            fi

            # Run and capture output
            echo "  Command: $cmd" >> "$log_path"
            if $cmd >> "$log_path" 2>&1; then
                echo "  ✓ Success"
                ((success++))
            else
                echo "  ✗ Failed (see $log_path)"
                ((failed++))
            fi

            echo ""
        done
    done
done

# End time and duration
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo "Total combinations: $total"
echo "  Successful: $success"
echo "  Failed: $failed"
echo "  Skipped (already existed): $skipped"
echo ""
echo "Time elapsed: ${hours}h ${minutes}m ${seconds}s"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $OUTPUT_DIR/logs"
echo ""

if [[ $failed -gt 0 ]]; then
    echo "⚠️  Some runs failed. Check log files in $OUTPUT_DIR/logs for details."
    exit 1
else
    echo "✓ All runs completed successfully!"
    exit 0
fi
