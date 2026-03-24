from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from data_loading.dataset import BaseDataset
from evaluation import metrics
from evaluation.eval_config import MetricResults
from research import create_manifest, create_result, file_sha256, write_manifest, write_result


def get_or_create_label_mapping(train_dataset: BaseDataset) -> Optional[Dict[str, int]]:
    if len(train_dataset) == 0:
        return None
    first_label = train_dataset[0][1]
    if isinstance(first_label, int):
        return None
    if isinstance(first_label, str):
        labels = sorted(set(train_dataset[i][1] for i in range(len(train_dataset))))
        return {label: idx for idx, label in enumerate(labels)}
    return None


def extract_client_datasets(
    splits: Dict[str, Any],
    train_dataset: Optional[BaseDataset] = None,
    use_original_labels: bool = False,
) -> List[List[Dict[str, Any]]]:
    if use_original_labels and train_dataset is not None:
        original_label_map = {str(train_dataset[i][0]): train_dataset[i][1] for i in range(len(train_dataset))}

    client_datasets: List[List[Dict[str, Any]]] = []
    for client_name in sorted(splits["clients"].keys()):
        client_split = splits["clients"][client_name]
        if use_original_labels and train_dataset is not None:
            client_data = [
                {"image": item["image"], "label": original_label_map[str(item["image"])]}
                for item in client_split["train"]
            ]
        else:
            client_data = list(client_split["train"])
        client_datasets.append(client_data)
    return client_datasets


def apply_label_mapping(dataset: List[Dict[str, Any]], mapping: Optional[Dict[str, int]]) -> List[Dict[str, Any]]:
    if mapping is None:
        return dataset
    return [{"image": item["image"], "label": mapping[item["label"]]} for item in dataset]


def subsample_datasets(
    client_datasets: List[List[Dict[str, Any]]],
    max_samples: int,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    if max_samples <= 0:
        return client_datasets
    rng = np.random.default_rng(seed)
    subsampled: List[List[Dict[str, Any]]] = []
    for dataset in client_datasets:
        if len(dataset) <= max_samples:
            subsampled.append(dataset)
            continue
        indices = rng.choice(len(dataset), size=max_samples, replace=False)
        subsampled.append([dataset[i] for i in indices])
    return subsampled


def compute_balance_metrics(client_datasets: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    sizes = [len(dataset) for dataset in client_datasets]
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


def compute_label_metrics(client_datasets: List[List[Dict[str, Any]]], num_classes: int) -> Dict[str, Any]:
    jsd_all = metrics.JS_divergence(client_datasets, num_classes, return_all=True)
    jsd_global_all = metrics.JSD_client_to_global(client_datasets, num_classes, return_all=True)
    tv_all = metrics.TV_label_distributions(client_datasets, num_classes, return_all=True)
    return {
        "JSD_mean": float(np.mean(jsd_all)),
        "JSD_std": float(np.std(jsd_all)),
        "JSD_global_mean": float(np.mean(jsd_global_all)),
        "JSD_global_std": float(np.std(jsd_global_all)),
        "TV_mean": float(np.mean(tv_all)),
        "TV_std": float(np.std(tv_all)),
    }


def compute_feature_metrics(
    client_datasets: List[List[Dict[str, Any]]],
    *,
    dataset_name: str,
    base_path: str,
    feature_model: str,
    batch_size: int,
    use_cache: bool,
    cache_dir: str,
) -> Dict[str, Any]:
    ext_kwargs = {
        "dataset_name": dataset_name,
        "base_path": base_path,
        "model_name": feature_model,
        "batch_size": batch_size,
        "use_cache": use_cache,
        "cache_dir": cache_dir,
    }
    fid_all = metrics.FID_distance(client_datasets, return_all=True, **ext_kwargs)
    mmd_all = metrics.MMD(client_datasets, return_all=True, **ext_kwargs)
    wass_all = metrics.sliced_wasserstein_distance(client_datasets, return_all=True, **ext_kwargs)
    energy_all = metrics.energy_distance_features(client_datasets, return_all=True, **ext_kwargs)
    id_all = metrics.intrinsic_dimensionality(client_datasets, method="two_nn", return_all=True, **ext_kwargs)
    id_clean = [value for value in id_all if not np.isnan(value)]
    knn_all = metrics.knn_separability(client_datasets, k=5, return_all=True, **ext_kwargs)
    centroid_all = metrics.centroid_distance(client_datasets, return_all=True, **ext_kwargs)
    vendi_all = metrics.vendi_score(client_datasets, return_all=True, **ext_kwargs)
    variance_all = metrics.feature_variance(client_datasets, return_all=True, **ext_kwargs)
    return {
        "fid_mean": float(np.mean(fid_all)),
        "fid_std": float(np.std(fid_all)),
        "mmd_features_mean": float(np.mean(mmd_all)),
        "mmd_features_std": float(np.std(mmd_all)),
        "wasserstein_mean": float(np.mean(wass_all)),
        "wasserstein_std": float(np.std(wass_all)),
        "energy_dist_mean": float(np.mean(energy_all)),
        "energy_dist_std": float(np.std(energy_all)),
        "id_mean": float(np.mean(id_clean)) if id_clean else None,
        "id_std": float(np.std(id_clean)) if id_clean else None,
        "knn_sep_mean": float(np.mean(knn_all)),
        "knn_sep_std": float(np.std(knn_all)),
        "centroid_dist_mean": float(np.mean(centroid_all)),
        "centroid_dist_std": float(np.std(centroid_all)),
        "vendi_score_mean": float(np.mean(vendi_all)),
        "vendi_score_std": float(np.std(vendi_all)),
        "feature_variance_mean": float(np.mean(variance_all)),
        "feature_variance_std": float(np.std(variance_all)),
    }


def compute_metrics_bundle(
    *,
    client_datasets: List[List[Dict[str, Any]]],
    num_classes: int,
    metrics_mode: str,
    feature_input: Optional[List[List[Dict[str, Any]]]] = None,
    dataset_name: str,
    base_path: str,
    feature_model: str,
    batch_size: int,
    use_cache: bool,
    cache_dir: str,
) -> Dict[str, Any]:
    metrics_obj = MetricResults(**compute_balance_metrics(client_datasets))
    if metrics_mode in {"label", "both"}:
        for key, value in compute_label_metrics(client_datasets, num_classes).items():
            setattr(metrics_obj, key, value)
    if metrics_mode in {"feature", "both"}:
        for key, value in compute_feature_metrics(
            feature_input or client_datasets,
            dataset_name=dataset_name,
            base_path=base_path,
            feature_model=feature_model,
            batch_size=batch_size,
            use_cache=use_cache,
            cache_dir=cache_dir,
        ).items():
            setattr(metrics_obj, key, value)
    return asdict(metrics_obj)


def persist_partition_eval_result(
    *,
    output_dir: Path,
    run_id: str,
    script_name: str,
    resolved_config: Dict[str, Any],
    dataset_name: str,
    base_path: str,
    partition_path: Path,
    method: str,
    n_clients: int,
    heterogeneity: Optional[float],
    seed: int,
    feature_model: str,
    status: str,
    summary: Dict[str, Any],
    metrics_payload: Dict[str, Any],
    failure: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = create_manifest(
        run_id=run_id,
        script_name=script_name,
        command=list(sys.argv),
        resolved_config=resolved_config,
        dataset={
            "name": dataset_name,
            "base_path": base_path,
            "split": "train",
        },
        partition={
            "path": str(partition_path),
            "filename": partition_path.name,
            "sha256": file_sha256(partition_path) if partition_path.exists() else None,
            "method": method,
            "n_clients": n_clients,
            "heterogeneity": heterogeneity,
        },
        model={
            "name": feature_model,
            "task_type": "embedding_feature_evaluation",
        },
        seed=seed,
        artifacts={
            "run_dir": str(run_dir),
            "manifest_path": str(run_dir / "run_manifest.json"),
            "result_path": str(run_dir / "result.json"),
            "split_path": str(partition_path),
        },
    )
    write_manifest(run_dir, manifest)
    write_result(
        run_dir,
        create_result(
            run_id=run_id,
            result_type="partition_eval",
            status=status,
            summary=summary,
            metrics=metrics_payload,
            artifacts={"split_path": str(partition_path), "run_dir": str(run_dir)},
            failure=failure,
        ),
    )
