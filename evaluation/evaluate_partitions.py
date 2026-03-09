"""
evaluate_partitions.py - Partition Evaluation Suite

Evaluates federated dataset partitions using label and feature-based metrics.
Supports single partition or batch evaluation with simple checkpointing.

Usage:
    # Evaluate all partitions in a directory
    python evaluate_partitions.py \
        --partition_dir ./partitions/tiny_imagenet \
        --dataset tiny_imagenet \
        --base_path /path/to/datasets \
        --output_dir ./results/tiny_imagenet

    # Evaluate a single partition
    python evaluate_partitions.py \
        --partition ./partitions/tiny_imagenet/dirichlet_n10_alpha0.5_seed42.json \
        --dataset tiny_imagenet \
        --base_path /path/to/datasets \
        --output_dir ./results/tiny_imagenet

    # Force re-evaluation (ignore existing results)
    python evaluate_partitions.py --partition_dir ./partitions --force ...
"""

import os

# Forces matrix operations to use a sane number of threads.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import argparse
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation.eval_config import PartitionConfig, MetricResults, EvaluationResult
from data_loading.dataset import BaseDataset
from evaluation import metrics
from utils.cli_common import add_standard_cli_args

log = logging.getLogger(__name__)

PARTITION_LOAD_ERRORS = (
    FileNotFoundError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
)

PARTITION_EVAL_ERRORS = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    FileNotFoundError,
    OSError,
)


# ============================================================================
# Helpers: Label Mapping (Strings -> Integers)
# ============================================================================

def get_or_create_label_mapping(
    dataset_name: str,
    dataset: BaseDataset,
    mapping_dir: Optional[Path] = None,
) -> Optional[Dict[str, int]]:
    """
    Handle datasets where labels are strings (e.g., Food101 "apple_pie", TinyImageNet "n021...").
    Metrics require integer labels, so we map strings to sorted integers 0..N-1.
    """
    if len(dataset) == 0: return None
    
    # Check the first item to see if mapping is needed
    # BaseDataset.__getitem__ returns (image_path, label)
    first_label = dataset[0][1]

    if isinstance(first_label, int):
        return None  # Already integers, no mapping needed

    if isinstance(first_label, str):
        mapping_file = None
        if mapping_dir is not None:
            mapping_file = mapping_dir / f"{dataset_name}_label_mapping.json"
            if mapping_file.exists():
                log.info("Loading label mapping from %s...", mapping_file)
                with open(mapping_file, "r") as f:
                    return json.load(f)

        log.info("Creating new integer mapping for string labels (%s)...", dataset_name)
        # Scan entire dataset to find unique labels
        # Note: We sort them to ensure deterministic assignment (0=Apple, 1=Banana...)
        all_labels = sorted(set(dataset[i][1] for i in range(len(dataset))))
        mapping = {label: idx for idx, label in enumerate(all_labels)}

        # Save for future runs only when an explicit persistence directory is configured.
        if mapping_file is not None:
            mapping_file.parent.mkdir(parents=True, exist_ok=True)
            with open(mapping_file, "w") as f:
                json.dump(mapping, f, indent=2)
        else:
            log.info("Label mapping persistence disabled; using in-memory mapping for this run.")
        
        return mapping

    return None


def create_dataset_map(dataset: BaseDataset, label_map: Optional[Dict[str, int]]) -> Dict[str, Dict]:
    """
    Builds O(1) lookup dictionary: path -> {image: path, label: int_label}
    Applies string->int mapping if provided.
    """
    dataset_map = {}
    log.info("Building O(1) dataset lookup map...")
    
    for path, raw_label in tqdm(dataset, desc="Indexing Dataset"):
        # Apply mapping if it exists (Food101/TinyIN), otherwise use raw integer (CIFAR)
        final_label = label_map[raw_label] if label_map else raw_label
        
        # Store in the format expected by metrics
        dataset_map[path] = {
            "image": path,
            "label": final_label
        }
        
    return dataset_map


# ============================================================================
# Data Loading
# ============================================================================

def load_partition_from_json(
    json_path: Path,
    dataset_map: Dict[str, Dict]
) -> List[List[Dict]]:
    """
    Load a partition JSON and reconstruct client datasets with labels.
    
    Args:
        json_path: Path to partition JSON with structure:
            {"clients": {"client_0": {"train": [img_path, ...], "val": [...]}, ...}}
        dataset_map: Dictionary {image_path: label} for O(1) lookups.
        
    Returns:
        List of client datasets, each a list of {"image": path, "label": label} dicts.
   """
    with open(json_path, 'r') as f:
        partition_data = json.load(f)
    
    # Sort clients to ensure deterministic order (client_0, client_1...)
    client_ids = sorted(
        partition_data["clients"].keys(), 
        key=lambda x: int(x.split('_')[-1]) if '_' in x and x.split('_')[-1].isdigit() else x
    )

    client_datasets = []
    for client_id in client_ids:
        client_split = partition_data["clients"][client_id]
        train_paths = client_split["train"]
        
        client_data = []
        for img_path in train_paths:

            # For embedding-based partitionings, the partitioning json is already using
            # a dictionary format with 'image' and 'label' keys. 
            if isinstance(img_path, dict):
                assert "image" in img_path and "label" in img_path, \
                    "Partition JSON dict entries must have 'image' and 'label' keys."
                item = dataset_map.get(img_path["image"])
                client_data.append(item)
                continue

            # O(1) Lookup - This connects the partition path to the mapped label
            item = dataset_map.get(img_path)

            if item is not None:
                client_data.append(item)
            else:
                log.warning("%s in partition but not in dataset map", img_path)
        
        client_datasets.append(client_data)
    
    return client_datasets


def subsample_datasets(
    client_datasets: List[List[Dict]],
    max_samples: int,
    seed: int
) -> List[List[Dict]]:
    """Randomly subsample each client dataset for expensive metrics."""
    if max_samples <= 0:
        return client_datasets
    
    rng = np.random.default_rng(seed)
    subsampled = []
    for dataset in client_datasets:
        if len(dataset) > max_samples:
            indices = rng.choice(len(dataset), size=max_samples, replace=False)
            subsampled.append([dataset[i] for i in indices])
        else:
            subsampled.append(dataset)
    return subsampled


def detect_has_labels(dataset_name: str, dataset_map: Dict[str, Any]) -> bool:
    """Detect whether the dataset has meaningful class labels."""
    NO_LABEL_DATASETS = {"ade20k", "ade20k_segmentation"}
    
    if dataset_name.lower() in NO_LABEL_DATASETS:
        return False
    
    try:
        first_item = next(iter(dataset_map.values()))
        return first_item["label"] is not None
    except StopIteration:
        return False


def get_num_classes(dataset_map: Dict[str, Any]) -> int:
    """Count unique classes in the dataset map."""
    labels = set(item["label"] for item in dataset_map.values())
    return len(labels)


# ============================================================================
# Metric Computation
# ============================================================================

def compute_balance_metrics(client_datasets: List[List[Dict]]) -> Dict[str, float]:
    """Compute data balance metrics across clients."""
    sizes = [len(d) for d in client_datasets]
    if not sizes:
        return {
            "samples_per_client_mean": 0.0, "samples_per_client_std": 0.0,
            "samples_per_client_min": 0, "samples_per_client_max": 0, "balance_cv": 0.0
        }

    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    
    return {
        "samples_per_client_mean": float(mean_size),
        "samples_per_client_std": float(std_size),
        "samples_per_client_min": int(np.min(sizes)),
        "samples_per_client_max": int(np.max(sizes)),
        "balance_cv": float(std_size / (mean_size + 1e-10)),
        "min_size": int(np.min(sizes)),
        "max_size": int(np.max(sizes)),
    }


def compute_label_metrics(
    client_datasets: List[List[Dict]],
    num_classes: int
) -> Dict[str, Optional[float]]:
    """Compute label distribution heterogeneity metrics."""
    results = {}
    
    # JS Divergence
    jsd_all = metrics.JS_divergence(client_datasets, num_classes, return_all=True)
    results["JSD_mean"] = float(np.mean(jsd_all))
    results["JSD_std"] = float(np.std(jsd_all))
    
    # JSD to Global
    jsd_global_all = metrics.JSD_client_to_global(client_datasets, num_classes, return_all=True)
    results["JSD_global_mean"] = float(np.mean(jsd_global_all))
    results["JSD_global_std"] = float(np.std(jsd_global_all))
    
    # Total Variation
    tv_all = metrics.TV_label_distributions(client_datasets, num_classes, return_all=True)
    results["TV_mean"] = float(np.mean(tv_all))
    results["TV_std"] = float(np.std(tv_all))
    
    return results


def compute_feature_metrics(
    dataset_name: str,
    base_path: str,
    client_datasets: List[List[Dict]],
    feature_model: str = "dinov2-small",
    batch_size: int = 64,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    subsample: int = 0,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Compute feature-based distribution metrics using subsampled data."""
    results = {}
    
    ext_kwargs = {
        "dataset_name": dataset_name,
        "base_path": base_path,
        "model_name": feature_model,
        "batch_size": batch_size,
        "use_cache": use_cache,
        "cache_dir": cache_dir
    }
    
    # --- Subsampling ---
    if subsample > 0:
        client_datasets_sub = subsample_datasets(client_datasets, subsample, seed=seed)
    else:
        client_datasets_sub = client_datasets

    # --- Distribution Distance Metrics ---
    
    log.info("    Computing FID...")
    fid_all = metrics.FID_distance(client_datasets_sub, return_all=True, **ext_kwargs)
    results["fid_mean"] = float(np.mean(fid_all))
    results["fid_std"] = float(np.std(fid_all))
    
    log.info("    Computing MMD (%s)...", feature_model)
    mmd_all = metrics.MMD(client_datasets_sub, return_all=True, **ext_kwargs)
    results["mmd_features_mean"] = float(np.mean(mmd_all))
    results["mmd_features_std"] = float(np.std(mmd_all))
    
    log.info("    Computing Sliced Wasserstein (%s)...", feature_model)
    wass_all = metrics.sliced_wasserstein_distance(client_datasets_sub, return_all=True, **ext_kwargs)
    results["wasserstein_mean"] = float(np.mean(wass_all))
    results["wasserstein_std"] = float(np.std(wass_all))
    
    log.info("    Computing Energy Distance (%s)...", feature_model)
    ed_all = metrics.energy_distance_features(client_datasets_sub, return_all=True, **ext_kwargs)
    results["energy_dist_mean"] = float(np.mean(ed_all))
    results["energy_dist_std"] = float(np.std(ed_all))
    
    # logger.debug(f"    Computing KS Test ({feature_model})...")
    # ks_all = metrics.KS_multivariate_test(
    #     client_datasets_sub, pca_variance=0.95, return_all=True, **ext_kwargs
    # )
    # results["ks_multivariate_reject_rate"] = float(np.mean([r["reject_null"] for r in ks_all]))
    # min_pvals = [r["min_pvalue"] for r in ks_all]
    # results["ks_multivariate_min_pvalue"] = float(np.min(min_pvals)) if min_pvals else None
    
    # --- Heterogeneity Metrics ---
    
    log.info("    Computing k-NN Separability (%s)...", feature_model)
    knn_all = metrics.knn_separability(client_datasets_sub, k=5, return_all=True, **ext_kwargs)
    results["knn_sep_mean"] = float(np.mean(knn_all))
    results["knn_sep_std"] = float(np.std(knn_all))
    
    log.info("    Computing Centroid Distance (%s)...", feature_model)
    cd_all = metrics.centroid_distance(client_datasets_sub, return_all=True, **ext_kwargs)
    results["centroid_dist_mean"] = float(np.mean(cd_all))
    results["centroid_dist_std"] = float(np.std(cd_all))
    
    # --- Diversity Metrics ---
    
    log.info("    Computing Vendi Score (%s)...", feature_model)
    vendi_all = metrics.vendi_score(client_datasets_sub, return_all=True, **ext_kwargs)
    results["vendi_score_mean"] = float(np.mean(vendi_all))
    results["vendi_score_std"] = float(np.std(vendi_all))
    
    log.info("    Computing Intrinsic Dimensionality (%s)...", feature_model)
    id_all = metrics.intrinsic_dimensionality(
        client_datasets_sub, method="two_nn", return_all=True, **ext_kwargs
    )
    id_clean = [x for x in id_all if not np.isnan(x)]
    if id_clean:
        results["id_mean"] = float(np.mean(id_clean))
        results["id_std"] = float(np.std(id_clean))
    else:
        results["id_mean"] = None
        results["id_std"] = None
    
    log.info("    Computing Feature Variance (%s)...", feature_model)
    fv_all = metrics.feature_variance(client_datasets_sub, return_all=True, **ext_kwargs)
    results["feature_variance_mean"] = float(np.mean(fv_all))
    results["feature_variance_std"] = float(np.std(fv_all))
    
    return results


def compute_all_metrics(
    dataset_name: str,
    base_path: str,
    client_datasets: List[List[Dict]],
    has_labels: bool,
    num_classes: int,
    feature_model: str,
    batch_size: int,
    use_cache: bool,
    cache_dir: Optional[str],
    subsample: int = 0,
    seed: int = 42,
) -> MetricResults:
    """Combine all metric groups into final results."""
    
    # 1. Balance
    log.info("  Computing balance metrics...")
    balance = compute_balance_metrics(client_datasets)
    
    # 2. Label
    label_results = {}
    if has_labels:
        log.info("  Computing label metrics...")
        label_results = compute_label_metrics(client_datasets, num_classes)
    else:
        log.info("  Skipping label metrics (no labels detected)")
    
    # 3. Feature
    log.info("  Computing feature metrics...")
    feature_results = compute_feature_metrics(
        dataset_name=dataset_name,
        base_path=base_path,
        client_datasets=client_datasets,
        feature_model=feature_model,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_dir=cache_dir,
        subsample=subsample,
        seed=seed,
    )
    
    # Flatten everything into the Dataclass
    return MetricResults(
        **balance,
        **label_results,
        **feature_results
    )


# ============================================================================
# Entry Points
# ============================================================================

def get_output_path(partition_path: Path, output_dir: Path) -> Path:
    """Generate output path for evaluation results."""
    return output_dir / f"{partition_path.stem}_eval.json"


def evaluate_partition(
    partition_path: Path,
    dataset_map: Dict[str, Dict],
    dataset_name: str,
    base_path: str,
    output_dir: Path,
    feature_model: str,
    batch_size: int,
    use_cache: bool,
    cache_dir: Optional[str],
    subsample: int,
    force: bool,
    seed: int,
) -> Optional[EvaluationResult]:
    
    # output_path = get_output_path(partition_path, output_dir)

    # Embedding-based partitions have the embedding model in the parent directory name
    # e.g., tiny_imagenet/mae-base/cls_mean_k4_mean_norm-true_dr-umap_hdbscan_euclidean_eom_valset-global_seed42.json
    # and tiny_imagenet/dinov2-base/cls_mean_k4_mean_norm-true_dr-umap_hdbscan_euclidean_eom_valset-global_seed42.json
    # should produce different output files (not overwrite each other) and not cause skipping

    parent_dir = partition_path.parent.name
    if parent_dir != "baselines":
        output_path = output_dir / f"{parent_dir}_{partition_path.stem}_eval.json"
    else:
        output_path = output_dir / f"{partition_path.stem}_eval.json"

    if output_path.exists() and not force:
        log.info("Skipping %s (results exist at %s)", partition_path.name, output_path.name)
        return None
    
    log.info("%s", "=" * 60)
    log.info("Evaluating: %s", partition_path.name)
    
    start_time = time.time()
    
    # Config Parsing
    config = PartitionConfig.from_path(str(partition_path))
    config.dataset_name = dataset_name
    
    # Load Data
    try:
        client_datasets = load_partition_from_json(partition_path, dataset_map)
    except PARTITION_LOAD_ERRORS as e:
        log.error("Failed to load partition %s: %s", partition_path, e)
        return None

    # Update dynamic client count (HDBSCAN)
    if config.n_clients == 0:
        config.n_clients = len(client_datasets)
    
    log.info("Loaded %s clients, %s samples", len(client_datasets), sum(len(c) for c in client_datasets))
    
    # Check Labels
    has_labels = detect_has_labels(dataset_name, dataset_map)
    num_classes = get_num_classes(dataset_map) if has_labels else 0
    
    # Compute
    metrics_obj = compute_all_metrics(
        dataset_name=dataset_name,
        base_path=base_path,
        client_datasets=client_datasets,
        has_labels=has_labels,
        num_classes=num_classes,
        feature_model=feature_model,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_dir=cache_dir,
        subsample=subsample,
        seed=seed,
    )
    
    # Save
    result = EvaluationResult(
        config=config,
        metrics=metrics_obj,
        timestamp=datetime.now().isoformat()
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    log.info("Saved %s (%.1fs)", output_path.name, time.time() - start_time)
    
    return result


def evaluate_all(
    partition_dir: Path,
    dataset_map: Dict[str, Dict],
    dataset_name: str,
    base_path: str,
    output_dir: Path,
    feature_model: str,
    batch_size: int,
    use_cache: bool,
    cache_dir: Optional[str],
    force: bool,
    subsample: int,
    seed: int,
) -> List[EvaluationResult]:
    
    partition_files = sorted(partition_dir.rglob("*.json"))
    log.info("Found %s partitions.", len(partition_files))
    
    results = []
    for p_path in tqdm(partition_files, desc="Evaluating"):
        try:
            res = evaluate_partition(
                p_path, dataset_map, dataset_name, base_path, output_dir,
                feature_model, batch_size, use_cache, cache_dir, subsample, force, seed
            )
            if res: results.append(res)
        except PARTITION_EVAL_ERRORS:
            log.exception("Error evaluating %s", p_path.name)
            
    # Aggregate CSV
    if results:
        # Convert results to DataFrame
        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)
        csv_path = output_dir / "all_results.csv"
        df.to_csv(csv_path, index=False)
        log.info("Aggregate CSV saved to %s", csv_path)
        
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser()
    # Inputs
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--partition", type=str)
    input_group.add_argument("--partition_dir", type=str)
    
    # Data & Output
    add_standard_cli_args(
        parser,
        include_dataset=True,
        dataset_required=True,
        include_base_path=True,
        base_path_required=True,
        include_output=True,
        output_required=True,
        output_help="Directory where evaluation JSON/CSV outputs are written.",
        include_seed=True,
        seed_default=42,
        include_dry_run=True,
    )
    
    # Config
    parser.add_argument("--feature_model", type=str, default="dinov2-small")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--subsample", type=int, default=0, 
                        help="Max samples per client for feature metrics (0=all)")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("FDS_CACHE_DIR", str(Path.cwd() / "embedding_cache")),
    )
    parser.add_argument(
        "--label_map_dir",
        type=str,
        default=os.environ.get("FDS_LABEL_MAP_DIR"),
        help="Optional directory for persisted string-label mappings.",
    )
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.dry_run:
        target = args.partition if args.partition else args.partition_dir
        log.info("[dry-run] Evaluation target: %s", target)
        log.info("[dry-run] Dataset: %s", args.dataset)
        log.info("[dry-run] Base path: %s", args.base_path)
        log.info("[dry-run] Output directory: %s", output_dir)
        return 0
    
    # 1. Load Dataset
    log.info("Loading %s from %s...", args.dataset, args.base_path)
    dataset = BaseDataset(args.dataset, args.base_path, "train")
    
    # 2. Label Mapping (String -> Int)
    use_cache = not args.no_cache
    label_map_dir = Path(args.label_map_dir) if args.label_map_dir else None
    if label_map_dir is None and use_cache and args.cache_dir:
        label_map_dir = Path(args.cache_dir) / "label_mappings"
    label_map = get_or_create_label_mapping(args.dataset, dataset, mapping_dir=label_map_dir)
    
    # 3. Create O(1) Map
    dataset_map = create_dataset_map(dataset, label_map)
    log.info("Map created with %s images.", len(dataset_map))
    
    # 4. Run
    if args.partition:
        evaluate_partition(
            Path(args.partition), dataset_map, args.dataset, args.base_path, output_dir,
            args.feature_model, args.batch_size, use_cache, args.cache_dir, 
            args.subsample, args.force, args.seed
        )
    else:
        evaluate_all(
            Path(args.partition_dir), dataset_map, args.dataset, args.base_path, output_dir,
            args.feature_model, args.batch_size, use_cache, args.cache_dir,
            args.force, args.subsample, args.seed
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
