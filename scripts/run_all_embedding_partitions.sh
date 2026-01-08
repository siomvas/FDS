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
MOCK=false                            # Set to true for quick testing with 1k samples
BASE_PATH="/path/to/gtai/FDS/" # UPDATE THIS PATH
SEED=42
VALSET_TYPE="global" # global, local, or none
LAST_K_POOLING=mean  # Token aggregation method for lastk pooling
TOKEN_LAST_K=4       # K value for last_k token strategies
N_CLUSTERS=10        # Default number of clusters for kmeans/spectral
GEM_P=2.0            # Gem pooling parameter
NORMALIZE=true       # Whether to L2 normalize embeddings before clustering
DIM_REDUCTION="umap" # Dimensionality reduction method: umap or none

DATASETS=(
    "tiny_imagenet"
    "food101"
    "ade20k"
)

EXTRACTORS=(
    "dinov2-base"
    "clip-base-patch16"
    # "clip-large-patch14"
    # "siglip-base-patch16-224"
    "mae-base"
)

TOKEN_STRATEGIES=(
    "cls"
    # "all_last"
    # "k_last_all"
    # "k_last_cls"
)

TOKEN_LAYER_AGGREGATION=(
    "mean"
    # "concat"
    # "weighted"
)

POOLING_STRATEGIES=(
    "mean"
    # "max"
    # "gem"
    # "lastk"
    # "spectral"
)
CLUSTERING_METHODS=(
    "hdbscan"
    # "kmeans"
    # "knn"
    # "spectral"
)

HDBSCAN_METRICS=(
    "euclidean"
    # "cosine"
)
HDBSCAN_METHODS=(
    "eom"
    # "leaf"
)

# =============================================================================
# SCRIPT - Usually no need to edit below this line
# =============================================================================

# Count total combinations
total=$((${#EXTRACTORS[@]} * ${#POOLING_STRATEGIES[@]} * ${#TOKEN_LAYER_AGGREGATION[@]} * ${#CLUSTERING_METHODS[@]} * ${#TOKEN_STRATEGIES[@]} * ${#HDBSCAN_METRICS[@]} * ${#HDBSCAN_METHODS[@]} * ${#DATASETS[@]}))
current=0
success=0
failed=0

# Start time
start_time=$(date +%s)

# Main loop
for extractor in "${EXTRACTORS[@]}"; do
    for clustering in "${CLUSTERING_METHODS[@]}"; do
        for hdbscan_metric in "${HDBSCAN_METRICS[@]}"; do
            for hdbscan_method in "${HDBSCAN_METHODS[@]}"; do
                for token_strategy in "${TOKEN_STRATEGIES[@]}"; do
                    for layer_agg in "${TOKEN_LAYER_AGGREGATION[@]}"; do
                        for pooling in "${POOLING_STRATEGIES[@]}"; do
                            for dataset in "${DATASETS[@]}"; do
                                ((current++))

                                # Skip incompatible combinations 1: hdbscan_metric and method only for hdbscan
                                if [[ "$clustering" != "hdbscan" ]]; then
                                    if [[ "$hdbscan_metric" != "${HDBSCAN_METRICS[0]}" ]] || [[ "$hdbscan_method" != "${HDBSCAN_METHODS[0]}" ]]; then
                                        continue
                                    fi
                                fi

                                # Skip incompatible combinations 2: cls tokens don't require layer aggregation and pooling
                                if [[ "$token_strategy" == "cls" ]] && [[ "$layer_agg" != "mean" || "$pooling" != "mean" ]]; then
                                    continue
                                fi

                                # Skip incompatible combinations 3: only k_last_* token strategies use layer aggregation
                                if [[ "$token_strategy" != "k_last_all" && "$token_strategy" != "k_last_cls" ]] && [[ "$layer_agg" != "mean" ]]; then
                                    continue
                                fi

                                # Skip incompatible combinations 4: k_last_* token strategies require 'lastk' pooling
                                if [[ "$token_strategy" == "k_last_all" || "$token_strategy" == "k_last_cls" ]] && [[ "$pooling" != "lastk" ]]; then
                                    continue
                                fi

                                # Skip incompatible combinations 5: only k_last_* token strategies use lastk pooling
                                if [[ "$pooling" == "lastk" ]] && [[ "$token_strategy" != "k_last_all" && "$token_strategy" != "k_last_cls" ]]; then
                                    continue
                                fi

                                OUTPUT_DIR="${BASE_PATH}/embedding_partitions/${dataset}/${extractor}" # Output directory for partition JSONs

                                # append mock to output dir if enabled
                                if [[ "$MOCK" == true ]]; then
                                    OUTPUT_DIR="${OUTPUT_DIR}/mock"
                                fi
                                # Create output directories
                                mkdir -p "$OUTPUT_DIR"
                                mkdir -p "$OUTPUT_DIR/logs"

                                # Build output filename
                                output_name="${token_strategy}_${layer_agg}_k${TOKEN_LAST_K}_${pooling}_norm-${NORMALIZE}_dr-${DIM_REDUCTION}_${clustering}_${hdbscan_metric}_${hdbscan_method}_valset-${VALSET_TYPE}_seed${SEED}.json"
                                output_path="${OUTPUT_DIR}/${output_name}"
                                log_path="${OUTPUT_DIR}/logs/${output_name%.json}.log"
                                echo "[$current/$total] Model: $extractor | Tokens: ($token_strategy extracted, $layer_agg layer agg of last $TOKEN_LAST_K, $pooling pooling) | Preprocessing: (normalize=$NORMALIZE, dim_reduction=$DIM_REDUCTION)"
                                echo "                            | Clustering: ($clustering, $N_CLUSTERS, $hdbscan_metric, $hdbscan_method) | Dataset: $dataset | Valset: $VALSET_TYPE | Seed: $SEED"
                                # Build command
                                cmd="python -u generate_embedding_partitions.py"
                                cmd="$cmd --dataset $dataset"
                                cmd="$cmd --base_path $BASE_PATH"
                                cmd="$cmd --extractor $extractor"
                                cmd="$cmd --tokens $token_strategy"
                                cmd="$cmd --lastk_k $TOKEN_LAST_K"
                                cmd="$cmd --layer_aggregation $layer_agg"
                                cmd="$cmd --pooling $pooling"
                                cmd="$cmd --clustering $clustering"
                                cmd="$cmd --hdbscan_metric $hdbscan_metric"
                                cmd="$cmd --hdbscan_method $hdbscan_method"
                                cmd="$cmd --valset_type $VALSET_TYPE"
                                cmd="$cmd --seed $SEED"
                                cmd="$cmd --output $output_path"

                                # Add n_clusters for kmeans/spectral
                                if [[ "$clustering" == "kmeans" || "$clustering" == "spectral" ]]; then
                                    cmd="$cmd --n_clusters $N_CLUSTERS"
                                    cmd="$cmd --spectral_k 50" # Default spectral k
                                fi

                                # add token aggregation for lastk pooling
                                if [[ "$pooling" == "lastk" ]]; then
                                    cmd="$cmd --lastk_pooling $LAST_K_POOLING"
                                fi

                                # Add gem_p for gem pooling
                                if [[ "$pooling" == "gem" ]]; then
                                    cmd="$cmd --gem_p $GEM_P"
                                fi

                                # Add mock flag if enabled
                                if [[ "$MOCK" == true ]]; then
                                    cmd="$cmd --mock"
                                fi

                                # Add normalization flag
                                if [[ "$NORMALIZE" == true ]]; then
                                    cmd="$cmd --normalize"
                                fi

                                # Add dimensionality reduction method
                                if [[ "$DIM_REDUCTION" == "umap" ]]; then
                                    cmd="$cmd --dim_reduction umap"
                                    cmd="$cmd --dr_dims 30"
                                    cmd="$cmd --umap_neighbors 20"
                                    cmd="$cmd --umap_min_dist 0.0"
                                fi

                                # Add HDBSCAN specific parameters
                                if [[ "$clustering" == "hdbscan" ]]; then
                                    cmd="$cmd --min_cluster_size 500"
                                    cmd="$cmd --min_samples 15"
                                fi

                                # Run and capture output
                                echo "  Command: $cmd" >"$log_path"
                                if $cmd >>"$log_path" 2>&1; then
                                    echo "  ✓ Success"
                                    ((success++))
                                    # Optionally, remove log file on success
                                    # rm "$log_path"
                                else
                                    echo "  ✗ Failed (see $log_path)"
                                    ((failed++))
                                    # append FAIL to log file and output file names
                                    mv "$log_path" "${log_path%.log}_FAIL.log"
                                    if [[ -f "$output_path" ]]; then
                                        mv "$output_path" "${output_path%.json}_FAIL.json"
                                    fi
                                fi
                            done
                        done
                    done
                done
            done
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
echo ""
echo "Time elapsed: ${hours}h ${minutes}m ${seconds}s"
echo "Output base directory: $OUTPUT_DIR"
echo "Log directory: $OUTPUT_DIR/logs"
echo ""

if [[ $failed -gt 0 ]]; then
    echo "⚠️  Some runs failed. Check log files in $OUTPUT_DIR/logs for details."
    exit 1
else
    echo "✓ All runs completed successfully!"
    exit 0
fi
