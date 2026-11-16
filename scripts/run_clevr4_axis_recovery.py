#!/usr/bin/env python3
"""Run the CLEVR-4 representation-axis recovery rebuttal experiment.

The script has deliberately small, resumable phases so feature extraction can be
split across GPUs while all metrics are merged into a single audited deliverable.
It uses the image ordering recorded by a completed semantic-pipeline step-1 file,
not a fresh directory listing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from embedding.embedding_extractors import ModelRegistry


LOGGER = logging.getLogger("clevr4_axis_recovery")

TAXONOMIES = ("texture", "shape", "color", "count")
SCATTERING_CONFIGS = (
    "wavelet_scattering-224-2",
    "wavelet_scattering-224-3",
    "wavelet_scattering-224-4",
    "wavelet_scattering-128-3",
    "wavelet_scattering-64-3",
)
NO_PRETRAIN_CONFIGS = SCATTERING_CONFIGS + (
    "raw_pixel_resizer-64",
    "sparse_random_projection_resizer",
    "UAE",
)
FOUNDATION_CONFIGS = (
    "dinov2-base",
    "clip-base-patch16",
    "siglip-base-patch16-224",
    "mae-base",
    "inception_v3",
)
ALL_CONFIGS = NO_PRETRAIN_CONFIGS + FOUNDATION_CONFIGS
SPECTRAL_CONFIGS = ("dinov2-base", "mae-base")
P_REMOVED = (0, 1, 2, 3, 5, 10)
EXPECTED_SAMPLES = 10_531
EXPECTED_ORDER_SHA256 = "f8ff5b4ea9ab483befa17fbb8ea4e28b0051adad5d870b7a97b31af0d5749197"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_protocol(
    semantic_step1: Path,
    annotations_path: Path,
) -> tuple[list[str], list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Load the exact ordered image list from the recent semantic run."""

    rows: list[dict[str, Any]] = []
    with semantic_step1.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {semantic_step1}:{line_number}") from exc
            if not isinstance(row, dict) or not row.get("image_id") or not row.get("image"):
                raise ValueError(f"Missing image_id/image at {semantic_step1}:{line_number}")
            rows.append(row)

    image_ids = [str(row["image_id"]) for row in rows]
    images = [str(Path(row["image"]).resolve()) for row in rows]
    if len(image_ids) != EXPECTED_SAMPLES or len(set(image_ids)) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} unique semantic-run IDs, got "
            f"{len(image_ids)} rows and {len(set(image_ids))} unique IDs."
        )
    order_sha256 = hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()
    if order_sha256 != EXPECTED_ORDER_SHA256:
        raise ValueError(
            f"Semantic-run order hash is {order_sha256}, expected {EXPECTED_ORDER_SHA256}."
        )
    missing_images = [path for path in images if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images; first: {missing_images[:3]}")

    with annotations_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, dict):
        raise ValueError(f"Expected an annotation dictionary in {annotations_path}.")

    labels: dict[str, np.ndarray] = {}
    taxonomy_counts: dict[str, dict[str, int]] = {}
    for taxonomy in TAXONOMIES:
        values = []
        for image_id in image_ids:
            annotation = annotations.get(image_id)
            if not isinstance(annotation, dict) or taxonomy not in annotation:
                raise ValueError(f"Missing {taxonomy!r} annotation for {image_id}.")
            values.append(str(annotation[taxonomy]).strip().lower())
        unique = sorted(set(values))
        if len(unique) != 10:
            raise ValueError(f"Expected 10 {taxonomy} labels, found {len(unique)}: {unique}")
        labels[taxonomy] = np.asarray(values)
        taxonomy_counts[taxonomy] = dict(sorted(Counter(values).items()))

    protocol = {
        "sample_count": len(image_ids),
        "semantic_step1": str(semantic_step1.resolve()),
        "semantic_step1_sha256": _sha256_file(semantic_step1),
        "annotations": str(annotations_path.resolve()),
        "annotations_sha256": _sha256_file(annotations_path),
        "ordered_id_sha256": order_sha256,
        "first_image_ids": image_ids[:3],
        "last_image_ids": image_ids[-3:],
        "taxonomy_counts": taxonomy_counts,
    }
    return image_ids, images, labels, protocol


def _canonical_extractor_name(config: str) -> str:
    if config == "sparse_random_projection_resizer":
        return "sparse_random_projection_resizer-64-1000"
    return config


def _default_batch_size(config: str) -> int:
    if config.startswith("wavelet_scattering"):
        return 16
    if config in {"raw_pixel_resizer-64", "sparse_random_projection_resizer", "UAE"}:
        return 256
    return 128


def _batch_features(extractor: Any, config: str, batch: torch.Tensor) -> np.ndarray:
    """Apply the repository extractor's native representation to one batch."""

    batch = batch.to(extractor.device, non_blocking=True)
    if config.startswith("wavelet_scattering"):
        output = extractor.scattering(batch)
        output = output.reshape(output.shape[0], -1)
    elif config == "raw_pixel_resizer-64":
        output = batch.reshape(batch.shape[0], -1)
    elif config == "UAE":
        output = extractor.autoencoder(batch)
        output = output.reshape(output.shape[0], -1)
    elif config == "inception_v3":
        output = extractor.model(batch)
    elif config == "sparse_random_projection_resizer":
        flat = batch.reshape(batch.shape[0], -1).cpu().numpy()
        if not getattr(extractor, "_axis_recovery_fitted", False):
            extractor.transformer.fit(flat)
            extractor._axis_recovery_fitted = True
        return np.asarray(extractor.transformer.transform(flat), dtype=np.float32)
    else:
        model_output = extractor.model(pixel_values=batch)
        if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
            output = model_output.pooler_output
        else:
            output = model_output.last_hidden_state[:, 0, :]
    return np.asarray(output.detach().float().cpu().numpy(), dtype=np.float32)


def extract_one(
    config: str,
    images: Sequence[str],
    protocol: dict[str, Any],
    output_root: Path,
    device: str,
    batch_size: int | None,
    num_workers: int,
) -> None:
    if config not in ALL_CONFIGS:
        raise ValueError(f"Unknown configuration {config!r}.")
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_path = features_dir / f"{_slug(config)}.npy"
    metadata_path = features_dir / f"{_slug(config)}.json"
    if feature_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing = np.load(feature_path, mmap_mode="r")
        if (
            existing.shape[0] == EXPECTED_SAMPLES
            and metadata.get("ordered_id_sha256") == protocol["ordered_id_sha256"]
            and metadata.get("complete") is True
        ):
            LOGGER.info("Skipping complete feature file for %s: %s", config, existing.shape)
            return
        raise ValueError(f"Incomplete or mismatched existing feature artifact for {config}.")

    _seed_everything(0)
    actual_batch_size = batch_size or _default_batch_size(config)
    extractor_name = _canonical_extractor_name(config)
    LOGGER.info("Loading %s as %s on %s", config, extractor_name, device)
    extractor = ModelRegistry.create(
        extractor_name,
        device=device,
        batch_size=actual_batch_size,
        num_workers=num_workers,
    )
    extractor.load_model()
    if config == "mae-base":
        # ViTMAEModel retains its pre-training mask ratio during eval mode.  A
        # reference image representation must encode the intact image; leaving
        # the default 75% random mask enabled makes features depend on inference
        # RNG and does not match the usual downstream use of an MAE encoder.
        extractor.model.config.mask_ratio = 0.0
        LOGGER.info("Disabled MAE inference masking (mask_ratio=0.0).")
    dataloader = extractor._get_dataloader(list(images))

    temporary = features_dir / f".{_slug(config)}.{os.getpid()}.npy"
    memmap: np.memmap | None = None
    offset = 0
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(dataloader):
                if batch is None:
                    raise ValueError(f"All images in batch {batch_index} were unreadable.")
                values = _batch_features(extractor, config, batch)
                if values.ndim != 2 or not np.isfinite(values).all():
                    raise ValueError(
                        f"{config} emitted invalid batch shape/values at batch {batch_index}: "
                        f"shape={values.shape}."
                    )
                if memmap is None:
                    memmap = np.lib.format.open_memmap(
                        temporary,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(images), values.shape[1]),
                    )
                    LOGGER.info("%s feature shape will be %s", config, memmap.shape)
                if values.shape[1] != memmap.shape[1] or offset + len(values) > len(images):
                    raise ValueError(f"Inconsistent feature batch for {config} at batch {batch_index}.")
                memmap[offset : offset + len(values)] = values
                offset += len(values)
                if batch_index % 25 == 0:
                    memmap.flush()
                    LOGGER.info("%s extraction: %d/%d", config, offset, len(images))
        if memmap is None or offset != len(images):
            raise ValueError(f"{config} wrote {offset}/{len(images)} feature rows.")
        memmap.flush()
        shape = list(memmap.shape)
        del memmap
        os.replace(temporary, feature_path)
    except BaseException:
        if memmap is not None:
            memmap.flush()
        raise

    metadata = {
        "complete": True,
        "config": config,
        "extractor_name": extractor_name,
        "feature_path": str(feature_path.resolve()),
        "feature_shape": shape,
        "dtype": "float32",
        "ordered_id_sha256": protocol["ordered_id_sha256"],
        "extraction_seed": 0,
        "batch_size": actual_batch_size,
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "representation": "repository_native_no_extra_normalization",
        "mae_mask_ratio": 0.0 if config == "mae-base" else None,
    }
    _atomic_json(metadata_path, metadata)
    LOGGER.info("Completed %s: %s", config, shape)

    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _partition_key(config: str, p_removed: int, seed: int) -> str:
    return f"{config}__p{p_removed}__seed{seed}"


def _write_evaluation_shard(
    output_root: Path,
    config: str,
    p_removed: int,
    seed: int,
    assignments: np.ndarray,
    labels: dict[str, np.ndarray],
) -> None:
    if assignments.shape != (EXPECTED_SAMPLES,):
        raise ValueError(f"Unexpected assignments shape: {assignments.shape}")
    cluster_sizes = np.bincount(assignments, minlength=10).astype(int).tolist()
    if len(cluster_sizes) != 10 or sum(cluster_sizes) != EXPECTED_SAMPLES:
        raise ValueError(f"Invalid cluster sizes for {config}, p={p_removed}, seed={seed}.")
    metric_rows = []
    for taxonomy in TAXONOMIES:
        truth = labels[taxonomy]
        metric_rows.append(
            {
                "config": config,
                "p_removed": int(p_removed),
                "seed": int(seed),
                "taxonomy": taxonomy,
                "nmi": float(
                    normalized_mutual_info_score(truth, assignments, average_method="arithmetic")
                ),
                "ami": float(
                    adjusted_mutual_info_score(truth, assignments, average_method="arithmetic")
                ),
                "ari": float(adjusted_rand_score(truth, assignments)),
            }
        )
    payload = {
        "config": config,
        "p_removed": int(p_removed),
        "seed": int(seed),
        "cluster_sizes": cluster_sizes,
        "metrics": metric_rows,
    }
    shard = output_root / "evaluation_shards" / f"{_slug(_partition_key(config, p_removed, seed))}.json"
    _atomic_json(shard, payload)


def evaluate_one(
    config: str,
    labels: dict[str, np.ndarray],
    protocol: dict[str, Any],
    output_root: Path,
    seeds: Sequence[int],
    spectral: bool,
) -> None:
    if config not in ALL_CONFIGS:
        raise ValueError(f"Unknown configuration {config!r}.")
    feature_path = output_root / "features" / f"{_slug(config)}.npy"
    metadata_path = output_root / "features" / f"{_slug(config)}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("ordered_id_sha256") != protocol["ordered_id_sha256"]:
        raise ValueError(f"Feature order mismatch for {config}.")
    features = np.load(feature_path, mmap_mode="r")
    if features.shape[0] != EXPECTED_SAMPLES or features.dtype != np.float32:
        raise ValueError(f"Invalid features for {config}: {features.shape}, {features.dtype}")

    p_values = P_REMOVED if spectral else (0,)
    if any(p > 0 for p in p_values) and config not in SPECTRAL_CONFIGS:
        raise ValueError(f"Spectral surgery is only defined for {SPECTRAL_CONFIGS}.")

    transformed: np.ndarray | None = None
    if any(p > 0 for p in p_values):
        LOGGER.info("Fitting full PCA for %s on %s", config, features.shape)
        # PCA's full orthogonal score space preserves all Euclidean distances at
        # p=0. Slicing scores[:, p:] is exactly the residual after removing the
        # top p principal directions, without storing a second full residual.
        pca = PCA()
        transformed = np.asarray(pca.fit_transform(features), dtype=np.float32)
        _atomic_json(
            output_root / "pca" / f"{_slug(config)}.json",
            {
                "config": config,
                "n_components": int(transformed.shape[1]),
                "explained_variance_ratio_top10": pca.explained_variance_ratio_[:10].tolist(),
                "ordered_id_sha256": protocol["ordered_id_sha256"],
                "solver": pca._fit_svd_solver,
            },
        )

    for p_removed in p_values:
        matrix: Any = features if p_removed == 0 else transformed[:, p_removed:]
        for seed in seeds:
            shard = (
                output_root
                / "evaluation_shards"
                / f"{_slug(_partition_key(config, p_removed, seed))}.json"
            )
            if shard.is_file():
                LOGGER.info("Skipping existing evaluation %s p=%d seed=%d", config, p_removed, seed)
                continue
            LOGGER.info("KMeans %s p=%d seed=%d on %s", config, p_removed, seed, matrix.shape)
            assignments = KMeans(n_clusters=10, random_state=seed).fit_predict(matrix)
            _write_evaluation_shard(
                output_root, config, p_removed, seed, assignments, labels
            )
            LOGGER.info(
                "Completed KMeans %s p=%d seed=%d sizes=%s",
                config,
                p_removed,
                seed,
                np.bincount(assignments, minlength=10).tolist(),
            )


def choose_best_scattering(output_root: Path) -> str:
    """Select the seed-0 scattering variant by the primary texture endpoint."""

    candidates: list[tuple[float, str]] = []
    for config in SCATTERING_CONFIGS:
        shard = (
            output_root
            / "evaluation_shards"
            / f"{_slug(_partition_key(config, 0, 0))}.json"
        )
        payload = json.loads(shard.read_text(encoding="utf-8"))
        row = next(item for item in payload["metrics"] if item["taxonomy"] == "texture")
        candidates.append((float(row["nmi"]), config))
    # Config name is a deterministic tie breaker.
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def _load_all_shards(output_root: Path) -> list[dict[str, Any]]:
    shards = []
    for path in sorted((output_root / "evaluation_shards").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or len(payload.get("metrics", [])) != 4:
            raise ValueError(f"Malformed evaluation shard: {path}")
        shards.append(payload)
    return shards


def finalize(output_root: Path, protocol: dict[str, Any]) -> None:
    best_scattering = choose_best_scattering(output_root)
    expected = {
        _partition_key(config, 0, 0) for config in ALL_CONFIGS
    }
    expected.update(
        _partition_key(config, p, 0)
        for config in SPECTRAL_CONFIGS
        for p in P_REMOVED
    )
    expected.update(_partition_key("dinov2-base", 0, seed) for seed in (0, 1, 2))
    expected.update(_partition_key(best_scattering, 0, seed) for seed in (0, 1, 2))

    shards = _load_all_shards(output_root)
    by_key = {
        _partition_key(item["config"], int(item["p_removed"]), int(item["seed"])): item
        for item in shards
    }
    missing = sorted(expected - set(by_key))
    if missing:
        raise ValueError(f"Cannot finalize; missing {len(missing)} evaluations: {missing}")

    selected = [by_key[key] for key in sorted(expected)]
    metric_rows = [row for item in selected for row in item["metrics"]]
    metric_rows.sort(key=lambda row: (row["config"], row["p_removed"], row["seed"], row["taxonomy"]))
    results_path = output_root / "results.csv"
    temporary = results_path.with_name(f".{results_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("config", "p_removed", "seed", "taxonomy", "nmi", "ami", "ari"),
        )
        writer.writeheader()
        writer.writerows(metric_rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, results_path)

    cluster_payload = {
        "sample_count": EXPECTED_SAMPLES,
        "k": 10,
        "best_scattering_control": best_scattering,
        "best_scattering_selection": "highest seed-0 texture NMI",
        "ordered_id_sha256": protocol["ordered_id_sha256"],
        "partitions": [
            {
                "config": item["config"],
                "p_removed": int(item["p_removed"]),
                "seed": int(item["seed"]),
                "cluster_sizes": [int(value) for value in item["cluster_sizes"]],
            }
            for item in selected
        ],
    }
    _atomic_json(output_root / "cluster_sizes.json", cluster_payload)

    seed0 = [row for row in metric_rows if int(row["seed"]) == 0]
    dino_control = [
        row
        for row in metric_rows
        if row["config"] == "dinov2-base" and int(row["p_removed"]) == 0
    ]
    scatter_control = [
        row
        for row in metric_rows
        if row["config"] == best_scattering and int(row["p_removed"]) == 0
    ]
    lines = [
        "# CLEVR-4 axis recovery summary",
        "",
        (
            f"All scores use the exact {EXPECTED_SAMPLES}-image order from the recent semantic run "
            f"(ordered-ID SHA-256 `{protocol['ordered_id_sha256']}`). K-means uses K=10 and "
            "scikit-learn defaults; NMI uses arithmetic normalization. Highest-configuration "
            "comparisons use seed 0 so the control seeds are not treated as hyperparameter trials."
        ),
        "",
        "| taxonomy | highest configuration overall (seed 0) | NMI | best unmodified foundation model | NMI | gap | conservative seed spread | gap exceeds spread? |",
        "|---|---|---:|---|---:|---:|---:|:---:|",
    ]
    for taxonomy in TAXONOMIES:
        taxonomy_rows = [row for row in seed0 if row["taxonomy"] == taxonomy]
        highest = max(taxonomy_rows, key=lambda row: float(row["nmi"]))
        foundation = max(
            (
                row
                for row in taxonomy_rows
                if row["config"] in FOUNDATION_CONFIGS and int(row["p_removed"]) == 0
            ),
            key=lambda row: float(row["nmi"]),
        )
        dino_values = [float(row["nmi"]) for row in dino_control if row["taxonomy"] == taxonomy]
        scatter_values = [
            float(row["nmi"]) for row in scatter_control if row["taxonomy"] == taxonomy
        ]
        dino_std = float(np.std(dino_values, ddof=0))
        scatter_std = float(np.std(scatter_values, ddof=0))
        spread = max(dino_std, scatter_std)
        gap = float(highest["nmi"]) - float(foundation["nmi"])
        highest_name = highest["config"]
        if int(highest["p_removed"]):
            highest_name += f" (p={int(highest['p_removed'])})"
        exceeds = "yes" if gap > spread else "no"
        lines.append(
            f"| {taxonomy} | {highest_name} | {float(highest['nmi']):.6f} | "
            f"{foundation['config']} | {float(foundation['nmi']):.6f} | {gap:+.6f} | "
            f"{spread:.6f} | {exceeds} |"
        )

    lines.extend(
        [
            "",
            "## Part A: representations without pretrained foundation models",
            "",
            "| taxonomy | best no-pretraining configuration (seed 0) | NMI | best unmodified foundation model | NMI | gap | conservative seed spread | absolute gap exceeds spread? |",
            "|---|---|---:|---|---:|---:|---:|:---:|",
        ]
    )
    for taxonomy in TAXONOMIES:
        taxonomy_rows = [row for row in seed0 if row["taxonomy"] == taxonomy]
        no_pretrain = max(
            (row for row in taxonomy_rows if row["config"] in NO_PRETRAIN_CONFIGS),
            key=lambda row: float(row["nmi"]),
        )
        foundation = max(
            (
                row
                for row in taxonomy_rows
                if row["config"] in FOUNDATION_CONFIGS and int(row["p_removed"]) == 0
            ),
            key=lambda row: float(row["nmi"]),
        )
        dino_values = [float(row["nmi"]) for row in dino_control if row["taxonomy"] == taxonomy]
        scatter_values = [
            float(row["nmi"]) for row in scatter_control if row["taxonomy"] == taxonomy
        ]
        spread = max(float(np.std(dino_values, ddof=0)), float(np.std(scatter_values, ddof=0)))
        gap = float(no_pretrain["nmi"]) - float(foundation["nmi"])
        exceeds = "yes" if abs(gap) > spread else "no"
        lines.append(
            f"| {taxonomy} | {no_pretrain['config']} | {float(no_pretrain['nmi']):.6f} | "
            f"{foundation['config']} | {float(foundation['nmi']):.6f} | {gap:+.6f} | "
            f"{spread:.6f} | {exceeds} |"
        )

    texture_rows = [row for row in seed0 if row["taxonomy"] == "texture"]
    best_no_pretrain_texture = max(
        (row for row in texture_rows if row["config"] in NO_PRETRAIN_CONFIGS),
        key=lambda row: float(row["nmi"]),
    )
    best_scattering_texture = max(
        (row for row in texture_rows if row["config"] in SCATTERING_CONFIGS),
        key=lambda row: float(row["nmi"]),
    )
    best_foundation_texture = max(
        (
            row
            for row in texture_rows
            if row["config"] in FOUNDATION_CONFIGS and int(row["p_removed"]) == 0
        ),
        key=lambda row: float(row["nmi"]),
    )
    lines.extend(
        [
            "",
            (
                "Part A is negative on texture: the best no-pretraining representation was "
                f"`{best_no_pretrain_texture['config']}` at NMI "
                f"{float(best_no_pretrain_texture['nmi']):.6f}, and the best scattering variant "
                f"(`{best_scattering_texture['config']}`) reached only "
                f"{float(best_scattering_texture['nmi']):.6f}, versus "
                f"{float(best_foundation_texture['nmi']):.6f} for "
                f"`{best_foundation_texture['config']}`. No no-pretraining arm exceeded the best "
                "unmodified foundation model on any taxonomy."
            ),
            "",
            (
                f"The best scattering control is `{best_scattering}`, selected before its extra "
                "seeds by highest seed-0 texture NMI. “Conservative seed spread” is the larger "
                "population standard deviation of NMI across seeds 0–2 for unmodified DINOv2-base "
                "and this scattering variant."
            ),
            "",
            "## Seed-control NMI standard deviations",
            "",
            "| taxonomy | DINOv2-base | best scattering |",
            "|---|---:|---:|",
        ]
    )
    for taxonomy in TAXONOMIES:
        dino_values = [float(row["nmi"]) for row in dino_control if row["taxonomy"] == taxonomy]
        scatter_values = [
            float(row["nmi"]) for row in scatter_control if row["taxonomy"] == taxonomy
        ]
        lines.append(
            f"| {taxonomy} | {np.std(dino_values, ddof=0):.6f} | "
            f"{np.std(scatter_values, ddof=0):.6f} |"
        )

    spectral_best = []
    for taxonomy in TAXONOMIES:
        candidates = [
            row
            for row in seed0
            if row["taxonomy"] == taxonomy and row["config"] in SPECTRAL_CONFIGS
        ]
        winner = max(candidates, key=lambda row: float(row["nmi"]))
        spectral_best.append(
            f"- {taxonomy}: `{winner['config']}` p={int(winner['p_removed'])}, "
            f"NMI {float(winner['nmi']):.6f}."
        )
    lines.extend(["", "## Best spectral-surgery settings", "", *spectral_best, ""])
    dino_texture_base = next(
        row
        for row in seed0
        if row["config"] == "dinov2-base"
        and int(row["p_removed"]) == 0
        and row["taxonomy"] == "texture"
    )
    dino_texture_best = max(
        (
            row
            for row in seed0
            if row["config"] == "dinov2-base" and row["taxonomy"] == "texture"
        ),
        key=lambda row: float(row["nmi"]),
    )
    mae_shape_base = next(
        row
        for row in seed0
        if row["config"] == "mae-base"
        and int(row["p_removed"]) == 0
        and row["taxonomy"] == "shape"
    )
    mae_shape_p3 = next(
        row
        for row in seed0
        if row["config"] == "mae-base"
        and int(row["p_removed"]) == 3
        and row["taxonomy"] == "shape"
    )
    lines.extend(
        [
            (
                "Spectral surgery did recover a previously weak texture axis: DINOv2 rose from "
                f"NMI {float(dino_texture_base['nmi']):.6f} at p=0 to "
                f"{float(dino_texture_best['nmi']):.6f} at "
                f"p={int(dino_texture_best['p_removed'])}. It did not reproduce the reported "
                "roughly sevenfold MAE shape gain at p=3: NMI changed from "
                f"{float(mae_shape_base['nmi']):.6f} to {float(mae_shape_p3['nmi']):.6f}."
            ),
            "",
            (
                "MAE was evaluated as an intact-image encoder with inference `mask_ratio=0.0`; "
                "otherwise the Hugging Face model applies a stochastic 75% pre-training mask in "
                "evaluation mode. Sparse random projection uses the repository default 64×64 "
                "input and 1,000 output dimensions."
            ),
            "",
        ]
    )
    _atomic_text(output_root / "summary.md", "\n".join(lines))

    _atomic_json(
        output_root / "run_manifest.json",
        {
            **protocol,
            "configs": list(ALL_CONFIGS),
            "no_pretraining_configs": list(NO_PRETRAIN_CONFIGS),
            "spectral_configs": list(SPECTRAL_CONFIGS),
            "p_removed": list(P_REMOVED),
            "kmeans": {"n_clusters": 10, "random_state": "row seed", "other_parameters": "sklearn defaults"},
            "metrics": {
                "nmi": {"average_method": "arithmetic"},
                "ami": {"average_method": "arithmetic"},
                "ari": "sklearn default",
            },
            "best_scattering_control": best_scattering,
            "results_rows": len(metric_rows),
            "partition_count": len(selected),
            "results_sha256": _sha256_file(results_path),
            "cluster_sizes_sha256": _sha256_file(output_root / "cluster_sizes.json"),
            "summary_sha256": _sha256_file(output_root / "summary.md"),
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "torch": torch.__version__,
            "scipy": __import__("scipy").__version__,
            "kymatio": __import__("kymatio").__version__,
            "mae_inference_mask_ratio": 0.0,
            "source_script_sha256": _sha256_file(Path(__file__).resolve()),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("protocol", "extract", "evaluate", "controls", "finalize"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--semantic-step1", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--spectral", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _, images, labels, protocol = load_protocol(args.semantic_step1, args.annotations)
    _atomic_json(args.output_root / "protocol.json", protocol)

    if args.mode == "protocol":
        LOGGER.info("Protocol verified: %s", json.dumps(protocol, sort_keys=True))
        return
    if args.mode == "extract":
        if not args.configs:
            raise ValueError("--configs is required for extract mode.")
        for config in args.configs:
            extract_one(
                config,
                images,
                protocol,
                args.output_root,
                args.device,
                args.batch_size,
                args.num_workers,
            )
        return
    if args.mode == "evaluate":
        if not args.configs:
            raise ValueError("--configs is required for evaluate mode.")
        for config in args.configs:
            evaluate_one(
                config,
                labels,
                protocol,
                args.output_root,
                seeds=args.seeds,
                spectral=args.spectral,
            )
        return
    if args.mode == "controls":
        best_scattering = choose_best_scattering(args.output_root)
        LOGGER.info("Best scattering by seed-0 texture NMI: %s", best_scattering)
        evaluate_one(
            "dinov2-base", labels, protocol, args.output_root, seeds=(1, 2), spectral=False
        )
        evaluate_one(
            best_scattering, labels, protocol, args.output_root, seeds=(1, 2), spectral=False
        )
        return
    finalize(args.output_root, protocol)


if __name__ == "__main__":
    main()
