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

# --- CRITICAL CPU CONSTRAINT ---
# Must be set BEFORE importing numpy/torch to prevent 5000% CPU spikes.
# This forces matrix operations to use a sane number of threads.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import argparse
import json
import time
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project imports - adjust path as needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_config import PartitionConfig, MetricResults, EvaluationResult
from data_loading.dataset import BaseDataset
from evaluation import metrics


# ============================================================================
# Helpers: Label Mapping (Strings -> Integers)
# ============================================================================

def get_or_create_label_mapping(dataset_name: str, dataset: BaseDataset) -> Optional[Dict[str, int]]:
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
        # Naming convention: {dataset}_label_mapping.json
        mapping_file = Path("evaluation") / f"{dataset_name}_label_mapping.json"
        
        if mapping_file.exists():
            print(f"  Loading label mapping from {mapping_file}...")
            with open(mapping_file, 'r') as f:
                return json.load(f)

        print(f"  Creating new integer mapping for string labels ({dataset_name})...")
        # Scan entire dataset to find unique labels
        # Note: We sort them to ensure deterministic assignment (0=Apple, 1=Banana...)
        all_labels = sorted(set(dataset[i][1] for i in range(len(dataset))))
        mapping = {label: idx for idx, label in enumerate(all_labels)}

        # Save for future runs so consistency is maintained
        mapping_file.parent.mkdir(exist_ok=True)
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        return mapping

    return None


def create_dataset_map(dataset: BaseDataset, label_map: Optional[Dict[str, int]]) -> Dict[str, Dict]:
    """
    Builds O(1) lookup dictionary: path -> {image: path, label: int_label}
    Applies string->int mapping if provided.
    """
    dataset_map = {}
    print("  Building O(1) dataset lookup map...")
    
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
            # O(1) Lookup - This connects the partition path to the mapped label
            item = dataset_map.get(img_path)
            
            if item is not None:
                client_data.append(item)
            else:
                # Silent skip or warning? Warning is better for debugging.
                # print(f"Warning: {img_path} in partition but not in dataset map")
                pass
        
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
    client_datasets: List[List[Dict]],
    feature_model: str = "dinov2-small",
    batch_size: int = 64,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    subsample: int = 0
) -> Dict[str, Optional[float]]:
    """Compute feature-based distribution metrics using subsampled data."""
    results = {}
    
    ext_kwargs = {
        "model_name": feature_model,
        "batch_size": batch_size,
        "use_cache": use_cache,
        "cache_dir": cache_dir
    }
    
    # --- Subsampling ---
    if subsample > 0:
        client_datasets_sub = subsample_datasets(client_datasets, subsample, seed=42)
    else:
        client_datasets_sub = client_datasets

    # --- Distribution Distance Metrics ---
    
    print("    Computing FID...")
    fid_all = metrics.FID_distance(
        client_datasets_sub, batch_size=batch_size, return_all=True,
        use_cache=use_cache, cache_dir=cache_dir
    )
    results["fid_mean"] = float(np.mean(fid_all))
    results["fid_std"] = float(np.std(fid_all))
    
    print(f"    Computing MMD ({feature_model})...")
    mmd_all = metrics.MMD(client_datasets_sub, return_all=True, **ext_kwargs)
    results["mmd_features_mean"] = float(np.mean(mmd_all))
    results["mmd_features_std"] = float(np.std(mmd_all))
    
    print(f"    Computing Sliced Wasserstein ({feature_model})...")
    wass_all = metrics.sliced_wasserstein_distance(client_datasets_sub, return_all=True, **ext_kwargs)
    results["wasserstein_mean"] = float(np.mean(wass_all))
    results["wasserstein_std"] = float(np.std(wass_all))
    
    print(f"    Computing Energy Distance ({feature_model})...")
    ed_all = metrics.energy_distance_features(client_datasets_sub, return_all=True, **ext_kwargs)
    results["energy_dist_mean"] = float(np.mean(ed_all))
    results["energy_dist_std"] = float(np.std(ed_all))
    
    print(f"    Computing KS Test ({feature_model})...")
    ks_all = metrics.KS_multivariate_test(
        client_datasets_sub, pca_variance=0.95, return_all=True, **ext_kwargs
    )
    results["ks_multivariate_reject_rate"] = float(np.mean([r["reject_null"] for r in ks_all]))
    min_pvals = [r["min_pvalue"] for r in ks_all]
    results["ks_multivariate_min_pvalue"] = float(np.min(min_pvals)) if min_pvals else None
    
    # --- Heterogeneity Metrics ---
    
    print(f"    Computing k-NN Separability ({feature_model})...")
    knn_all = metrics.knn_separability(client_datasets_sub, k=5, return_all=True, **ext_kwargs)
    results["knn_sep_mean"] = float(np.mean(knn_all))
    results["knn_sep_std"] = float(np.std(knn_all))
    
    print(f"    Computing Centroid Distance ({feature_model})...")
    cd_all = metrics.centroid_distance(client_datasets_sub, return_all=True, **ext_kwargs)
    results["centroid_dist_mean"] = float(np.mean(cd_all))
    results["centroid_dist_std"] = float(np.std(cd_all))
    
    # --- Diversity Metrics ---
    
    print(f"    Computing Vendi Score ({feature_model})...")
    vendi_all = metrics.vendi_score(client_datasets_sub, return_all=True, **ext_kwargs)
    results["vendi_score_mean"] = float(np.mean(vendi_all))
    results["vendi_score_std"] = float(np.std(vendi_all))
    
    print(f"    Computing Intrinsic Dimensionality ({feature_model})...")
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
    
    print(f"    Computing Feature Variance ({feature_model})...")
    fv_all = metrics.feature_variance(client_datasets_sub, return_all=True, **ext_kwargs)
    results["feature_variance_mean"] = float(np.mean(fv_all))
    results["feature_variance_std"] = float(np.std(fv_all))
    
    return results


def compute_all_metrics(
    client_datasets: List[List[Dict]],
    has_labels: bool,
    num_classes: int,
    feature_model: str,
    batch_size: int,
    use_cache: bool,
    cache_dir: Optional[str],
    subsample: int = 0
) -> MetricResults:
    """Combine all metric groups into final results."""
    
    # 1. Balance
    print("  Computing balance metrics...")
    balance = compute_balance_metrics(client_datasets)
    
    # 2. Label
    label_results = {}
    if has_labels:
        print("  Computing label metrics...")
        label_results = compute_label_metrics(client_datasets, num_classes)
    else:
        print("  Skipping label metrics (no labels detected)")
    
    # 3. Feature
    print("  Computing feature metrics...")
    feature_results = compute_feature_metrics(
        client_datasets,
        feature_model=feature_model,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_dir=cache_dir,
        subsample=subsample
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
    output_dir: Path,
    feature_model: str,
    batch_size: int,
    use_cache: bool,
    cache_dir: Optional[str],
    subsample: int,
    force: bool
) -> Optional[EvaluationResult]:
    
    output_path = get_output_path(partition_path, output_dir)
    
    if output_path.exists() and not force:
        print(f"  Skipping {partition_path.name} (results exist)")
        return None
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {partition_path.name}")
    
    start_time = time.time()
    
    # Config Parsing
    config = PartitionConfig.from_path(str(partition_path))
    config.dataset_name = dataset_name
    
    # Load Data
    try:
        client_datasets = load_partition_from_json(partition_path, dataset_map)
    except Exception as e:
        print(f"  Failed to load partition: {e}")
        return None

    # Update dynamic client count (HDBSCAN)
    if config.n_clients == 0:
        config.n_clients = len(client_datasets)
    
    print(f"  Loaded {len(client_datasets)} clients, {sum(len(c) for c in client_datasets)} samples")
    
    # Check Labels
    has_labels = detect_has_labels(dataset_name, dataset_map)
    num_classes = get_num_classes(dataset_map) if has_labels else 0
    
    # Compute
    metrics_obj = compute_all_metrics(
        client_datasets=client_datasets,
        has_labels=has_labels,
        num_classes=num_classes,
        feature_model=feature_model,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_dir=cache_dir,
        subsample=subsample
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
    print(f"  Saved: {output_path.name} ({time.time() - start_time:.1f}s)")
    
    return result


def evaluate_all(
    partition_dir: Path,
    dataset_map: Dict[str, Dict],
    dataset_name: str,
    output_dir: Path,
    feature_model: str,
    batch_size: int,
    use_cache: bool,
    cache_dir: Optional[str],
    force: bool,
    subsample: int
) -> List[EvaluationResult]:
    
    partition_files = sorted(partition_dir.rglob("*.json"))
    print(f"Found {len(partition_files)} partitions.")
    
    results = []
    for p_path in tqdm(partition_files, desc="Evaluating"):
        try:
            res = evaluate_partition(
                p_path, dataset_map, dataset_name, output_dir,
                feature_model, batch_size, use_cache, cache_dir, subsample, force
            )
            if res: results.append(res)
        except Exception as e:
            print(f"ERROR: {p_path.name}: {e}")
            traceback.print_exc()
            
    # Aggregate CSV
    if results:
        # Convert results to DataFrame
        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)
        csv_path = output_dir / "all_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Aggregate CSV saved to {csv_path}")
        
    return results


def main():
    parser = argparse.ArgumentParser()
    # Inputs
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--partition", type=str)
    input_group.add_argument("--partition_dir", type=str)
    
    # Data & Output
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Config
    parser.add_argument("--feature_model", type=str, default="dinov2-small")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--subsample", type=int, default=0, 
                        help="Max samples per client for feature metrics (0=all)")
    parser.add_argument("--cache_dir", type=str, default="/path/to/gtai/FDS/embedding_cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    
    # 1. Load Dataset
    print(f"Loading {args.dataset} from {args.base_path}...")
    dataset = BaseDataset(args.dataset, args.base_path, "train")
    
    # 2. Label Mapping (String -> Int)
    label_map = get_or_create_label_mapping(args.dataset, dataset)
    
    # 3. Create O(1) Map
    dataset_map = create_dataset_map(dataset, label_map)
    print(f"Map created with {len(dataset_map)} images.")
    
    output_dir = Path(args.output_dir)
    use_cache = not args.no_cache
    
    # 4. Run
    if args.partition:
        evaluate_partition(
            Path(args.partition), dataset_map, args.dataset, output_dir,
            args.feature_model, args.batch_size, use_cache, args.cache_dir, 
            args.subsample, args.force
        )
    else:
        evaluate_all(
            Path(args.partition_dir), dataset_map, args.dataset, output_dir,
            args.feature_model, args.batch_size, use_cache, args.cache_dir,
            args.force, args.subsample
        )

if __name__ == "__main__":
    main()