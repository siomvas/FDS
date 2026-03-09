#!/usr/bin/env python3
"""Run Tiny-ImageNet embedding + multivariate partitioning audit.

This script implements the end-to-end audit workflow described in
`audit/embedding_partitioning/matrix_manifest.yaml`:
1. Baseline checks
2. Embedding matrix sweeps (default families + variant smoke + tiered confirmations)
3. Reproducibility and cache checks
4. Multivariate (VLM) partition generation and validation
5. Downstream FL smoke checks
6. Report and backlog generation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from data_loading.data_modules import get_datamodule
from data_loading.dataset import BaseDataset
from embedding.cache_manager import CacheManager
from embedding.embedding_extractors import ModelRegistry
from fl_training.engines.classification import ClassificationEngine
from partitioning.embedding_splitter import EmbeddingSplitter


log = logging.getLogger("embedding_matrix_audit")


CRITERION_TO_DIR = {
    "color": "output_color_train",
    "main_object": "output_main_object_train",
    "size": "output_size_train",
    "location": "output_location_train",
    "time": "output_time_train",
}

DEFAULT_FAMILY_MATRIX = [
    "inception_v3",
    "mae-base",
    "dinov2-base",
    "clip-base-patch16",
    "siglip-base-patch16-224",
    "raw_pixel_resizer-64",
    "sparse_random_projection_resizer-64-1000",
    "UAE",
    "wavelet_scattering-224-3",
]

VARIANT_SMOKE_MATRIX = [
    "mae-large",
    "mae-huge",
    "dinov2-small",
    "dinov2-large",
    "dinov2-giant",
    "clip-base-patch32",
    "clip-large-patch14",
    "siglip-base-patch16-384",
    "siglip-large-patch16-256",
    "raw_pixel_resizer-128",
    "sparse_random_projection_resizer-64-512",
    "wavelet_scattering-224-2",
]

CLUSTERING_PRESETS = {
    "hdbscan": {
        "min_cluster_size": 30,
        "hdbscan_metric": "euclidean",
        "hdbscan_method": "eom",
    },
    "kmeans": {
        "n_clusters": 6,
    },
    "knn": {
        "n_neighbors": 15,
        "metric": "euclidean",
    },
    "spectral": {
        "n_clusters": 6,
        "affinity": "nearest_neighbors",
    },
}

TIER3_SHORTLIST = [
    ("dinov2-base", "hdbscan"),
    ("dinov2-base", "kmeans"),
    ("clip-base-patch16", "kmeans"),
    ("raw_pixel_resizer-64", "knn"),
]

FL_SMOKE_TARGETS = [
    {"n_clients": 2, "alpha": 0.1, "seed": 42},
    {"n_clients": 2, "alpha": 5.0, "seed": 42},
    {"n_clients": 10, "alpha": 0.5, "seed": 42},
]

RUN_RESULTS_COLUMNS = [
    "run_id",
    "phase",
    "tier",
    "extractor",
    "clustering",
    "seed",
    "subset_train",
    "subset_val",
    "cap_reason",
    "status",
    "duration_sec",
    "n_clients",
    "cluster_count",
    "train_samples",
    "val_samples",
    "split_path",
    "split_hash",
    "error_type",
    "error_message",
]

BACKLOG_COLUMNS = [
    "id",
    "severity",
    "confidence",
    "combo",
    "evidence",
    "root_cause",
    "fix",
    "test_delta",
    "status",
]


class DatasetSubset:
    """Simple deterministic subset wrapper preserving dataset identity."""

    def __init__(self, dataset: BaseDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
        self.dataset_name = getattr(dataset, "dataset_name", None)
        self.requested_dataset_name = getattr(dataset, "requested_dataset_name", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


@dataclass
class Finding:
    finding_id: str
    severity: str
    confidence: float
    combo: str
    evidence: str
    root_cause: str
    fix: str
    test_delta: str
    status: str = "open"

    def to_backlog_row(self) -> Dict[str, str]:
        return {
            "id": self.finding_id,
            "severity": self.severity,
            "confidence": f"{self.confidence:.2f}",
            "combo": self.combo,
            "evidence": self.evidence,
            "root_cause": self.root_cause,
            "fix": self.fix,
            "test_delta": self.test_delta,
            "status": self.status,
        }


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, num_classes))

    def forward(self, x):
        return self.net(x)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_last5(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-5:])


def deterministic_indices(length: int, size: Optional[int], seed: int) -> List[int]:
    if size is None or size >= length:
        return list(range(length))
    if size <= 0:
        raise ValueError("Subset size must be > 0 when provided.")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(length, size=size, replace=False).tolist())


def subset_dataset(dataset: BaseDataset, size: Optional[int], seed: int) -> BaseDataset:
    if size is None or size >= len(dataset):
        return dataset
    return DatasetSubset(dataset, deterministic_indices(len(dataset), size, seed))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def load_vlm_maps(vlm_root: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    label_maps: Dict[str, Dict[str, str]] = {}
    line_counts: Dict[str, int] = {}

    for criterion, rel_dir in CRITERION_TO_DIR.items():
        path = vlm_root / rel_dir / "step3_result.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing VLM annotation file: {path}")

        mapping: Dict[str, str] = {}
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                mapping[normalize_last5(entry["image"])] = str(entry["class"])
                count += 1
        label_maps[criterion] = mapping
        line_counts[criterion] = count
    return label_maps, line_counts


def validate_split_invariants(
    splits: Dict[str, Any], expected_train_count: int
) -> Dict[str, Any]:
    clients = splits.get("clients", {})
    train_paths: List[str] = []
    for block in clients.values():
        for sample in block.get("train", []):
            train_paths.append(str(sample["image"]))

    total_train = len(train_paths)
    unique_train = len(set(train_paths))
    duplicate_count = total_train - unique_train
    missing_count = expected_train_count - unique_train

    return {
        "total_train": total_train,
        "unique_train": unique_train,
        "duplicate_count": duplicate_count,
        "missing_count": missing_count,
        "no_overlap": duplicate_count == 0,
        "full_coverage": unique_train == expected_train_count,
    }


def build_clustering_params(method: str, seed: int) -> Dict[str, Any]:
    params = dict(CLUSTERING_PRESETS[method])
    if method in {"kmeans", "spectral"}:
        params["random_state"] = seed
    return params


def apply_runtime_caps(
    extractor: str,
    clustering: str,
    subset_train: Optional[int],
    subset_val: Optional[int],
) -> Tuple[Optional[int], Optional[int], str]:
    train_cap = subset_train
    val_cap = subset_val
    reasons: List[str] = []

    # Wavelet embeddings are very high-dimensional (~500k for 224-3) and become
    # prohibitively expensive with large sample sizes in density/spectral methods.
    if extractor.startswith("wavelet_scattering"):
        max_train = 300
        max_val = 60
        if train_cap is None or train_cap > max_train:
            train_cap = max_train
            reasons.append("wavelet_train_cap")
        if val_cap is None or val_cap > max_val:
            val_cap = max_val
            reasons.append("wavelet_val_cap")

    # Full-size HDBSCAN on high-dimensional raw pixels can be extremely slow.
    if extractor.startswith("raw_pixel_resizer") and clustering == "hdbscan":
        max_train = 5000
        max_val = 1000
        if train_cap is None or train_cap > max_train:
            train_cap = max_train
            reasons.append("raw_hdbscan_train_cap")
        if val_cap is None or val_cap > max_val:
            val_cap = max_val
            reasons.append("raw_hdbscan_val_cap")

    # Spectral scaling is O(n^2) memory/time; enforce practical cap.
    if clustering == "spectral":
        max_train = 2000
        max_val = 400
        if train_cap is None or train_cap > max_train:
            train_cap = max_train
            reasons.append("spectral_train_cap")
        if val_cap is None or val_cap > max_val:
            val_cap = max_val
            reasons.append("spectral_val_cap")

    return train_cap, val_cap, ",".join(reasons)


def run_embedding_combo(
    *,
    run_id: str,
    phase: str,
    tier: str,
    extractor: str,
    clustering: str,
    seed: int,
    train_full: BaseDataset,
    val_full: BaseDataset,
    subset_train: Optional[int],
    subset_val: Optional[int],
    output_dir: Path,
    cache_dir: Path,
    dataset_name: str,
    dataset_root: str,
    device: str,
    batch_size: int,
    num_workers: int,
    proxy: Optional[str],
    findings: List[Finding],
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    split_path = output_dir / "splits" / tier / f"{run_id}.json"
    ensure_dir(split_path.parent)

    row: Dict[str, Any] = {
        "run_id": run_id,
        "phase": phase,
        "tier": tier,
        "extractor": extractor,
        "clustering": clustering,
        "seed": seed,
        "subset_train": subset_train if subset_train is not None else "full",
        "subset_val": subset_val if subset_val is not None else "full",
        "cap_reason": "",
        "status": "failed",
        "duration_sec": "",
        "n_clients": "",
        "cluster_count": "",
        "train_samples": "",
        "val_samples": "",
        "split_path": str(split_path),
        "split_hash": "",
        "error_type": "",
        "error_message": "",
    }

    try:
        effective_train, effective_val, cap_reason = apply_runtime_caps(
            extractor=extractor,
            clustering=clustering,
            subset_train=subset_train,
            subset_val=subset_val,
        )
        if cap_reason:
            log.warning(
                "Applying runtime caps for %s + %s: train=%s val=%s (%s)",
                extractor,
                clustering,
                effective_train,
                effective_val,
                cap_reason,
            )
            row["cap_reason"] = cap_reason

        row["subset_train"] = effective_train if effective_train is not None else "full"
        row["subset_val"] = effective_val if effective_val is not None else "full"
        train_ds = subset_dataset(train_full, effective_train, seed)
        val_ds = subset_dataset(val_full, effective_val, seed + 11)
        clustering_params = build_clustering_params(clustering, seed)

        splitter = EmbeddingSplitter(
            train_dataset=train_ds,
            val_dataset=val_ds,
            extractor_name=extractor,
            clustering_method=clustering,
            clustering_params=clustering_params,
            token_extraction_strategy="cls",
            token_params={},
            pooling_strategy="mean",
            pooling_params={},
            preprocessing_params={},
            valset_type="global",
            seed=seed,
            cache_dir=str(cache_dir),
            dataset_name=dataset_name,
            base_path=dataset_root,
            device=device,
            batch_size=batch_size,
            use_cache=True,
            proxy=proxy,
            num_workers=num_workers,
        )

        splits = splitter.get_client_splits()
        splitter.validate_splits(splits)

        invariants = validate_split_invariants(splits, expected_train_count=len(train_ds))
        if not invariants["no_overlap"] or not invariants["full_coverage"]:
            raise ValueError(
                "Split invariants failed "
                f"(no_overlap={invariants['no_overlap']}, full_coverage={invariants['full_coverage']})."
            )

        if clustering in {"kmeans", "spectral"} and splitter.n_clients != CLUSTERING_PRESETS[clustering]["n_clusters"]:
            raise ValueError(
                f"Expected {CLUSTERING_PRESETS[clustering]['n_clusters']} clients for {clustering}, "
                f"got {splitter.n_clients}."
            )
        if clustering in {"hdbscan", "knn"} and splitter.n_clients <= 0:
            raise ValueError(f"{clustering} returned non-positive client count: {splitter.n_clients}.")

        splitter.store_client_splits(str(split_path), format="json")
        split_hash = file_sha256(split_path)

        summary = splitter.get_split_summary()
        row["status"] = "passed"
        row["n_clients"] = summary["n_clients"]
        row["cluster_count"] = splitter.cluster_info_.get("num_clusters", "")
        row["train_samples"] = summary.get("total_train_samples", "")
        row["val_samples"] = summary.get("global_val_samples", summary.get("total_val_samples", ""))
        row["split_hash"] = split_hash

    except Exception as exc:  # noqa: BLE001 - keep matrix execution alive
        row["status"] = "failed"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc).replace("\n", " ")[:500]

        findings.append(
            Finding(
                finding_id=f"EMB-{run_id}",
                severity="P1",
                confidence=0.95,
                combo=f"{extractor}+{clustering}@{tier}",
                evidence=f"{split_path} | {type(exc).__name__}: {exc}",
                root_cause="Embedding partitioning combo failed at runtime.",
                fix="Inspect traceback/logs, then adjust extractor/clustering/proxy/dependency wiring.",
                test_delta="Add targeted regression for this combo in matrix harness.",
                status="open",
            )
        )
        trace_path = output_dir / "logs" / f"{run_id}.trace.txt"
        ensure_dir(trace_path.parent)
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    row["duration_sec"] = f"{elapsed:.2f}"
    return row


def split_file_name(n_clients: int, alpha: float, seed: int) -> str:
    return f"vlm_n{n_clients}_alpha{alpha}_seed{seed}.json"


def compute_specialization_score(
    split_payload: Dict[str, Any],
    vlm_maps: Dict[str, Dict[str, str]],
) -> float:
    client_scores: List[float] = []
    for client_block in split_payload.get("clients", {}).values():
        samples = client_block.get("train", [])
        norm_paths = [normalize_last5(str(sample["image"])) for sample in samples]
        if not norm_paths:
            continue

        criterion_scores: List[float] = []
        for criterion, label_map in vlm_maps.items():
            labels = [label_map[path] for path in norm_paths if path in label_map]
            if not labels:
                continue
            counts = np.array(list(Counter(labels).values()), dtype=np.float64)
            probs = counts / counts.sum()
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            max_entropy = float(np.log(len(counts))) if len(counts) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            criterion_scores.append(1.0 - normalized_entropy)
        if criterion_scores:
            client_scores.append(float(np.mean(criterion_scores)))

    if not client_scores:
        return 0.0
    return float(np.mean(client_scores))


def run_multivariate_generation(
    *,
    dataset_name: str,
    dataset_root: str,
    output_dir: Path,
    vlm_labels_dir: Path,
    findings: List[Finding],
) -> Tuple[int, str]:
    ensure_dir(output_dir)
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/generate_baselines.py",
        "--dataset",
        dataset_name,
        "--base_path",
        dataset_root,
        "--output",
        str(output_dir),
        "--n_clients",
        "2",
        "6",
        "10",
        "--alphas",
        "0.1",
        "0.5",
        "1.0",
        "5.0",
        "--seeds",
        "42",
        "99",
        "--cluster_labels_dir",
        str(vlm_labels_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(repo_root))
    if proc.returncode != 0:
        findings.append(
            Finding(
                finding_id="VLM-GEN-FAIL",
                severity="P0",
                confidence=0.99,
                combo="multivariate_generation",
                evidence=f"returncode={proc.returncode}",
                root_cause="VLM baseline generation command failed.",
                fix="Inspect command stderr and dependency/data path assumptions.",
                test_delta="Add command-level smoke regression in CI.",
                status="open",
            )
        )
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def run_fl_smoke(
    *,
    split_file: Path,
    dataset_root: str,
    num_classes: int = 200,
) -> Dict[str, Any]:
    cfg = {
        "training": {
            "steps_or_epochs": "steps",
            "loss": "cross_entropy",
            "optim_resume": "resume",
            "amp_enabled": False,
            "grad_clipping_type": "none",
            "grad_clipping_value": 0.0,
            "local_steps": 1,
            "lr": 1e-3,
            "lr_min": 0.0,
            "scheduler": "none",
        },
        "optimizer": {"class_name": "Adam", "args": {}},
        "fed": {"rounds": 1},
    }

    dm = get_datamodule(
        dataset_name="tiny_imagenet",
        data_dir=dataset_root,
        split_file=str(split_file),
        client_idx=0,
        batch_size=64,
        num_workers=0,
        seed=42,
    )
    dm.setup()

    engine = ClassificationEngine(
        model=TinyClassifier(num_classes=num_classes),
        config=cfg,
        device=torch.device("cpu"),
    )
    train_metrics = engine.train_one_round(dm.train_dataloader)
    val_metrics = engine.evaluate(dm.val_dataloader)

    return {
        "train_loss": float(train_metrics["train_loss"]),
        "num_steps_current_round": float(train_metrics["num_steps_current_round"]),
        "val_loss": float(val_metrics["val_loss"]),
        "accuracy": float(val_metrics.get("accuracy", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding and multivariate partitioning audit.")
    parser.add_argument("--dataset_name", type=str, default="tiny_imagenet")
    parser.add_argument("--dataset_root", type=str, default=str(Path("local_data").resolve()))
    parser.add_argument(
        "--vlm_labels_dir",
        type=str,
        default=str(Path("partitioning/vlm/data/tiny_imagenet").resolve()),
    )
    parser.add_argument(
        "--audit_dir",
        type=str,
        default=str(Path("audit/embedding_partitioning").resolve()),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(Path("local_artifacts/embedding_cache_audit").resolve()),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed_primary", type=int, default=42)
    parser.add_argument("--seed_secondary", type=int, default=99)
    parser.add_argument("--tier1_train", type=int, default=1000)
    parser.add_argument("--tier1_val", type=int, default=200)
    parser.add_argument("--tier2_train", type=int, default=10000)
    parser.add_argument("--tier2_val", type=int, default=2000)
    parser.add_argument("--tier2_spectral_train", type=int, default=2000)
    parser.add_argument("--tier2_spectral_val", type=int, default=400)
    parser.add_argument(
        "--proxy",
        type=str,
        default="",
        help="Proxy forwarded to extractors. Empty string disables proxy usage contexts.",
    )
    parser.add_argument("--skip_variant", action="store_true", help="Skip Phase 2 variant smoke.")
    parser.add_argument("--skip_tier1", action="store_true", help="Skip Phase 1 default family matrix.")
    parser.add_argument("--skip_tier2", action="store_true", help="Skip Tier 2 confirmation runs.")
    parser.add_argument("--skip_tier3", action="store_true", help="Skip Tier 3 full-data shortlist runs.")
    parser.add_argument("--skip_repro", action="store_true", help="Skip reproducibility checks.")
    parser.add_argument("--skip_cache", action="store_true", help="Skip embedding cache checks.")
    parser.add_argument("--skip_multivariate", action="store_true", help="Skip Phase 4 multivariate generation/validation.")
    parser.add_argument("--skip_fl_smoke", action="store_true", help="Skip downstream FL smoke runs.")
    parser.add_argument(
        "--only_run_id",
        action="append",
        default=[],
        help="Run only the specified run_id (repeatable).",
    )
    parser.add_argument(
        "--run_ids_file",
        type=str,
        default="",
        help=(
            "Optional file with run IDs to execute. Supports plain text "
            "(one run_id per line) or CSV with a 'run_id' column."
        ),
    )
    return parser.parse_args()


def load_selected_run_ids(args: argparse.Namespace) -> Optional[set[str]]:
    selected: set[str] = set(args.only_run_id or [])

    if args.run_ids_file:
        path = Path(args.run_ids_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"run_ids_file does not exist: {path}")

        if path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if "run_id" in (reader.fieldnames or []):
                    for row in reader:
                        rid = (row.get("run_id") or "").strip()
                        if rid:
                            selected.add(rid)
                else:
                    # Fallback: first CSV column if no named run_id column.
                    f.seek(0)
                    raw_reader = csv.reader(f)
                    for raw in raw_reader:
                        if not raw:
                            continue
                        rid = raw[0].strip()
                        if rid and rid.lower() != "run_id":
                            selected.add(rid)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                rid = line.strip()
                if not rid or rid.startswith("#"):
                    continue
                selected.add(rid)

    return selected if selected else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    audit_dir = Path(args.audit_dir)
    cache_dir = Path(args.cache_dir)
    dataset_root = str(Path(args.dataset_root).resolve())
    vlm_labels_dir = Path(args.vlm_labels_dir).resolve()
    proxy = args.proxy

    ensure_dir(audit_dir)
    ensure_dir(cache_dir)
    ensure_dir(audit_dir / "splits")
    ensure_dir(audit_dir / "logs")

    findings: List[Finding] = []
    run_rows: List[Dict[str, Any]] = []
    selected_run_ids = load_selected_run_ids(args)

    if selected_run_ids is not None:
        log.info("Selective run mode enabled: %d run_ids requested", len(selected_run_ids))

    def should_run(run_id: str) -> bool:
        if selected_run_ids is None:
            return True
        return run_id in selected_run_ids

    manifest = {
        "created_at_utc": now_utc(),
        "dataset_name": args.dataset_name,
        "dataset_root": dataset_root,
        "vlm_labels_dir": str(vlm_labels_dir),
        "default_family_matrix": DEFAULT_FAMILY_MATRIX,
        "variant_smoke_matrix": VARIANT_SMOKE_MATRIX,
        "clustering_presets": CLUSTERING_PRESETS,
        "tier_sizes": {
            "tier1": {"train": args.tier1_train, "val": args.tier1_val},
            "tier2": {"train": args.tier2_train, "val": args.tier2_val},
            "tier2_spectral_cap": {"train": args.tier2_spectral_train, "val": args.tier2_spectral_val},
            "tier3": {"train": "full", "val": "full"},
        },
        "seeds": [args.seed_primary, args.seed_secondary],
        "proxy": proxy,
    }
    (audit_dir / "matrix_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    # ---------------------------------------------------------------------
    # Phase 0: baseline
    # ---------------------------------------------------------------------
    log.info("Phase 0: baseline checks")
    train_full = BaseDataset(args.dataset_name, dataset_root, "train")
    val_full = BaseDataset(args.dataset_name, dataset_root, "val")
    vlm_maps, vlm_line_counts = load_vlm_maps(vlm_labels_dir)

    train_norm_paths = {normalize_last5(str(train_full[i][0])) for i in range(len(train_full))}
    overlap_by_criterion = {}
    for criterion, mapping in vlm_maps.items():
        overlap = len(train_norm_paths.intersection(mapping.keys()))
        overlap_by_criterion[criterion] = {
            "overlap_count": overlap,
            "train_total": len(train_norm_paths),
            "coverage_ratio": overlap / len(train_norm_paths) if train_norm_paths else 0.0,
        }
        if overlap < len(train_norm_paths):
            findings.append(
                Finding(
                    finding_id=f"BASE-{criterion.upper()}-COVERAGE",
                    severity="P1",
                    confidence=0.95,
                    combo=f"baseline/{criterion}",
                    evidence=f"coverage={overlap}/{len(train_norm_paths)}",
                    root_cause="VLM annotation map does not fully cover train samples after path normalization.",
                    fix="Regenerate or repair criterion labels so every train sample is present.",
                    test_delta="Add strict coverage assertion in baseline preflight.",
                )
            )

    baseline_lines = [
        "# Baseline State",
        "",
        f"- Timestamp (UTC): {now_utc()}",
        f"- Dataset root: `{dataset_root}`",
        f"- Train samples: {len(train_full)}",
        f"- Val samples: {len(val_full)}",
        f"- VLM labels root: `{vlm_labels_dir}`",
        "",
        "## VLM Line Counts",
    ]
    for criterion, count in vlm_line_counts.items():
        baseline_lines.append(f"- {criterion}: {count}")
    baseline_lines.append("")
    baseline_lines.append("## Normalized Path Overlap")
    for criterion, stats in overlap_by_criterion.items():
        baseline_lines.append(
            f"- {criterion}: {stats['overlap_count']}/{stats['train_total']} "
            f"({stats['coverage_ratio']:.4f})"
        )
    (audit_dir / "baseline.md").write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------------
    # Phase 1: tier1 matrix (default families x all clustering)
    # ---------------------------------------------------------------------
    if not args.skip_tier1:
        log.info("Phase 1: tier1 matrix")
        for extractor in DEFAULT_FAMILY_MATRIX:
            for clustering in CLUSTERING_PRESETS.keys():
                run_id = f"tier1_{extractor.replace('-', '_')}_{clustering}_s{args.seed_primary}"
                if not should_run(run_id):
                    continue
                row = run_embedding_combo(
                    run_id=run_id,
                    phase="embedding_matrix",
                    tier="tier1",
                    extractor=extractor,
                    clustering=clustering,
                    seed=args.seed_primary,
                    train_full=train_full,
                    val_full=val_full,
                    subset_train=args.tier1_train,
                    subset_val=args.tier1_val,
                    output_dir=audit_dir,
                    cache_dir=cache_dir,
                    dataset_name=args.dataset_name,
                    dataset_root=dataset_root,
                    device=args.device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    proxy=proxy,
                    findings=findings,
                )
                run_rows.append(row)

    # ---------------------------------------------------------------------
    # Phase 2: variant smoke (kmeans only, tier1 scale)
    # ---------------------------------------------------------------------
    if not args.skip_variant:
        log.info("Phase 2: variant smoke")
        for extractor in VARIANT_SMOKE_MATRIX:
            run_id = f"variant_{extractor.replace('-', '_')}_kmeans_s{args.seed_primary}"
            if not should_run(run_id):
                continue
            row = run_embedding_combo(
                run_id=run_id,
                phase="variant_smoke",
                tier="tier1_variant",
                extractor=extractor,
                clustering="kmeans",
                seed=args.seed_primary,
                train_full=train_full,
                val_full=val_full,
                subset_train=args.tier1_train,
                subset_val=args.tier1_val,
                output_dir=audit_dir,
                cache_dir=cache_dir,
                dataset_name=args.dataset_name,
                dataset_root=dataset_root,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                proxy=proxy,
                findings=findings,
            )
            run_rows.append(row)

    # ---------------------------------------------------------------------
    # Phase 3: tiered confirmation
    # ---------------------------------------------------------------------
    log.info("Phase 3: tiered confirmation")
    if not args.skip_tier2:
        if args.skip_tier1:
            passed_tier1 = [
                {
                    "tier": "tier1",
                    "status": "passed",
                    "extractor": extractor,
                    "clustering": clustering,
                }
                for extractor in DEFAULT_FAMILY_MATRIX
                for clustering in CLUSTERING_PRESETS.keys()
            ]
        else:
            passed_tier1 = [
                row for row in run_rows
                if row["tier"] == "tier1" and row["status"] == "passed"
            ]
        for prior in passed_tier1:
            clustering = prior["clustering"]
            extractor = prior["extractor"]
            if clustering == "spectral":
                subset_train = args.tier2_spectral_train
                subset_val = args.tier2_spectral_val
                tier = "tier2_spectral_cap"
            else:
                subset_train = args.tier2_train
                subset_val = args.tier2_val
                tier = "tier2"
            run_id = f"{tier}_{extractor.replace('-', '_')}_{clustering}_s{args.seed_primary}"
            if not should_run(run_id):
                continue
            run_rows.append(
                run_embedding_combo(
                    run_id=run_id,
                    phase="tiered_confirmation",
                    tier=tier,
                    extractor=extractor,
                    clustering=clustering,
                    seed=args.seed_primary,
                    train_full=train_full,
                    val_full=val_full,
                    subset_train=subset_train,
                    subset_val=subset_val,
                    output_dir=audit_dir,
                    cache_dir=cache_dir,
                    dataset_name=args.dataset_name,
                    dataset_root=dataset_root,
                    device=args.device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    proxy=proxy,
                    findings=findings,
                )
            )

    if not args.skip_tier3:
        for extractor, clustering in TIER3_SHORTLIST:
            run_id = f"tier3_{extractor.replace('-', '_')}_{clustering}_s{args.seed_primary}"
            if not should_run(run_id):
                continue
            run_rows.append(
                run_embedding_combo(
                    run_id=run_id,
                    phase="tiered_confirmation",
                    tier="tier3_full",
                    extractor=extractor,
                    clustering=clustering,
                    seed=args.seed_primary,
                    train_full=train_full,
                    val_full=val_full,
                    subset_train=None,
                    subset_val=None,
                    output_dir=audit_dir,
                    cache_dir=cache_dir,
                    dataset_name=args.dataset_name,
                    dataset_root=dataset_root,
                    device=args.device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    proxy=proxy,
                    findings=findings,
                )
            )

    # Reproducibility checks
    log.info("Phase 3b: reproducibility checks")
    repro_checks: List[Dict[str, Any]] = []
    if not args.skip_repro:
        same_seed_a = None
        same_seed_b = None
        diff_seed = None

        if should_run("repro_same_a"):
            same_seed_a = run_embedding_combo(
                run_id="repro_same_a",
                phase="repro_check",
                tier="repro",
                extractor="raw_pixel_resizer-64",
                clustering="kmeans",
                seed=args.seed_primary,
                train_full=train_full,
                val_full=val_full,
                subset_train=args.tier1_train,
                subset_val=args.tier1_val,
                output_dir=audit_dir,
                cache_dir=cache_dir,
                dataset_name=args.dataset_name,
                dataset_root=dataset_root,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                proxy=proxy,
                findings=findings,
            )
            run_rows.append(same_seed_a)

        if should_run("repro_same_b"):
            same_seed_b = run_embedding_combo(
                run_id="repro_same_b",
                phase="repro_check",
                tier="repro",
                extractor="raw_pixel_resizer-64",
                clustering="kmeans",
                seed=args.seed_primary,
                train_full=train_full,
                val_full=val_full,
                subset_train=args.tier1_train,
                subset_val=args.tier1_val,
                output_dir=audit_dir,
                cache_dir=cache_dir,
                dataset_name=args.dataset_name,
                dataset_root=dataset_root,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                proxy=proxy,
                findings=findings,
            )
            run_rows.append(same_seed_b)

        if should_run("repro_diff_seed"):
            diff_seed = run_embedding_combo(
                run_id="repro_diff_seed",
                phase="repro_check",
                tier="repro",
                extractor="raw_pixel_resizer-64",
                clustering="kmeans",
                seed=args.seed_secondary,
                train_full=train_full,
                val_full=val_full,
                subset_train=args.tier1_train,
                subset_val=args.tier1_val,
                output_dir=audit_dir,
                cache_dir=cache_dir,
                dataset_name=args.dataset_name,
                dataset_root=dataset_root,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                proxy=proxy,
                findings=findings,
            )
            run_rows.append(diff_seed)

        if same_seed_a is not None and same_seed_b is not None and same_seed_a["status"] == "passed" and same_seed_b["status"] == "passed":
            equal_same_seed = same_seed_a["split_hash"] == same_seed_b["split_hash"]
            repro_checks.append(
                {"check": "same_seed_equal", "result": equal_same_seed, "a": same_seed_a["split_hash"], "b": same_seed_b["split_hash"]}
            )
            if not equal_same_seed:
                findings.append(
                    Finding(
                        finding_id="REPRO-SAME-SEED",
                        severity="P1",
                        confidence=0.90,
                        combo="raw_pixel_resizer-64+kmeans",
                        evidence="same seed produced different split hashes",
                        root_cause="Non-deterministic behavior under fixed seed.",
                        fix="Audit random state propagation in embedding extraction and clustering path.",
                        test_delta="Add deterministic hash assertion test for fixed-seed combo.",
                    )
                )

        if same_seed_a is not None and diff_seed is not None and same_seed_a["status"] == "passed" and diff_seed["status"] == "passed":
            changed_diff_seed = same_seed_a["split_hash"] != diff_seed["split_hash"]
            repro_checks.append(
                {"check": "diff_seed_changes", "result": changed_diff_seed, "a": same_seed_a["split_hash"], "b": diff_seed["split_hash"]}
            )
            if not changed_diff_seed:
                findings.append(
                    Finding(
                        finding_id="REPRO-DIFF-SEED",
                        severity="P2",
                        confidence=0.80,
                        combo="raw_pixel_resizer-64+kmeans",
                        evidence="different seeds produced identical split hashes",
                        root_cause="Seed may not be affecting the expected stochastic path.",
                        fix="Verify random_state usage in clusterer construction and subset sampling.",
                        test_delta="Add explicit different-seed divergence check for stochastic methods.",
                    )
                )

    # Cache checks
    log.info("Phase 3c: cache checks")
    cache_checks = {
        "cache_hit_before": "skipped",
        "cache_hit_after": "skipped",
        "shape_equal": "skipped",
        "values_equal": "skipped",
        "first_duration_sec": "skipped",
        "second_duration_sec": "skipped",
    }
    if not args.skip_cache:
        cache_probe_dir = cache_dir / "cache_probe"
        ensure_dir(cache_probe_dir)
        cache_mgr = CacheManager(cache_dir=str(cache_probe_dir), enabled=True)
        images = [str(train_full[i][0]) for i in deterministic_indices(len(train_full), 256, args.seed_primary)]
        extractor = ModelRegistry.create(
            "raw_pixel_resizer-64",
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_manager=cache_mgr,
            proxy=proxy,
        )
        cache_hit_before = cache_mgr.get(
            model_name=extractor.model_name,
            image_paths=images,
            dataset_name=args.dataset_name,
            base_path=dataset_root,
            target_size=64,
        )
        t0 = datetime.now(timezone.utc)
        emb_first = extractor.extract_embeddings(
            images=images,
            use_cache=True,
            dataset_name=args.dataset_name,
            base_path=dataset_root,
        )
        first_duration = (datetime.now(timezone.utc) - t0).total_seconds()

        cache_hit_after = cache_mgr.get(
            model_name=extractor.model_name,
            image_paths=images,
            dataset_name=args.dataset_name,
            base_path=dataset_root,
            target_size=64,
        )

        t1 = datetime.now(timezone.utc)
        emb_second = extractor.extract_embeddings(
            images=images,
            use_cache=True,
            dataset_name=args.dataset_name,
            base_path=dataset_root,
        )
        second_duration = (datetime.now(timezone.utc) - t1).total_seconds()

        cache_checks = {
            "cache_hit_before": cache_hit_before is not None,
            "cache_hit_after": cache_hit_after is not None,
            "shape_equal": emb_first.shape == emb_second.shape,
            "values_equal": bool(np.allclose(emb_first, emb_second)),
            "first_duration_sec": round(first_duration, 4),
            "second_duration_sec": round(second_duration, 4),
        }
        if not cache_checks["cache_hit_after"] or not cache_checks["values_equal"]:
            findings.append(
                Finding(
                    finding_id="CACHE-EMB-CHECK",
                    severity="P1",
                    confidence=0.90,
                    combo="raw_pixel_resizer-64 cache probe",
                    evidence=json.dumps(cache_checks),
                    root_cause="Cache keying/retrieval failed for repeated extraction.",
                    fix="Audit CacheManager key construction and extractor cache params.",
                    test_delta="Add direct cache probe regression to automated audit.",
                )
            )

    # ---------------------------------------------------------------------
    # Phase 4: multivariate VLM audit
    # ---------------------------------------------------------------------
    multivariate_rows: List[Dict[str, Any]] = []
    fl_rows: List[Dict[str, Any]] = []
    low_alpha_mean = 0.0
    high_alpha_mean = 0.0
    heterogeneity_order_ok = True
    rc: Any = "skipped"

    if not args.skip_multivariate:
        log.info("Phase 4: multivariate VLM audit")
        multivariate_dir = audit_dir / "multivariate_splits"
        rc, cmd_output = run_multivariate_generation(
            dataset_name=args.dataset_name,
            dataset_root=dataset_root,
            output_dir=multivariate_dir,
            vlm_labels_dir=vlm_labels_dir,
            findings=findings,
        )
        (audit_dir / "logs" / "multivariate_generation.log").write_text(cmd_output, encoding="utf-8")

        specialization_by_alpha: Dict[float, List[float]] = defaultdict(list)
        for n_clients in [2, 6, 10]:
            for alpha in [0.1, 0.5, 1.0, 5.0]:
                for seed in [42, 99]:
                    path = multivariate_dir / split_file_name(n_clients, alpha, seed)
                    row = {
                        "file": str(path),
                        "exists": path.exists(),
                        "n_clients": n_clients,
                        "alpha": alpha,
                        "seed": seed,
                        "status": "missing",
                        "no_overlap": False,
                        "full_coverage": False,
                        "min_samples_ok": False,
                        "specialization_score": "",
                    }
                    if path.exists():
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        inv = validate_split_invariants(payload, expected_train_count=len(train_full))
                        min_samples_ok = all(
                            len(block.get("train", [])) >= 10
                            for block in payload.get("clients", {}).values()
                        )
                        score = compute_specialization_score(payload, vlm_maps)
                        specialization_by_alpha[alpha].append(score)
                        row.update(
                            {
                                "status": "validated",
                                "no_overlap": inv["no_overlap"],
                                "full_coverage": inv["full_coverage"],
                                "min_samples_ok": min_samples_ok,
                                "specialization_score": f"{score:.6f}",
                            }
                        )
                        if not inv["no_overlap"] or not inv["full_coverage"] or not min_samples_ok:
                            findings.append(
                                Finding(
                                    finding_id=f"VLM-INV-n{n_clients}-a{alpha}-s{seed}",
                                    severity="P1",
                                    confidence=0.95,
                                    combo=f"vlm_n{n_clients}_alpha{alpha}_seed{seed}",
                                    evidence=json.dumps(
                                        {
                                            "no_overlap": inv["no_overlap"],
                                            "full_coverage": inv["full_coverage"],
                                            "min_samples_ok": min_samples_ok,
                                        }
                                    ),
                                    root_cause="Generated VLM split violates core split invariants.",
                                    fix="Inspect multivariate assignment/retry logic and serialization path.",
                                    test_delta="Add invariant assertions for this VLM configuration.",
                                )
                            )
                    multivariate_rows.append(row)

        low_alpha_mean = float(np.mean(specialization_by_alpha[0.1])) if specialization_by_alpha[0.1] else 0.0
        high_alpha_mean = float(np.mean(specialization_by_alpha[5.0])) if specialization_by_alpha[5.0] else 0.0
        heterogeneity_order_ok = low_alpha_mean > high_alpha_mean
        if not heterogeneity_order_ok:
            findings.append(
                Finding(
                    finding_id="VLM-HETEROGENEITY-ORDER",
                    severity="P2",
                    confidence=0.75,
                    combo="multivariate alpha sweep",
                    evidence=f"low_alpha_mean={low_alpha_mean:.6f}, high_alpha_mean={high_alpha_mean:.6f}",
                    root_cause="Observed specialization does not decrease from low alpha to high alpha.",
                    fix="Inspect preference sampling and assignment temperature effects in multivariate splitter.",
                    test_delta="Add alpha monotonicity guardrail test over fixed seeds.",
                )
            )

        # FL smoke on representative VLM splits
        if not args.skip_fl_smoke:
            log.info("Phase 4b: downstream FL smoke on representative VLM splits")
            for target in FL_SMOKE_TARGETS:
                split_path = multivariate_dir / split_file_name(target["n_clients"], target["alpha"], target["seed"])
                row = {
                    "split_file": str(split_path),
                    "status": "failed",
                    "train_loss": "",
                    "val_loss": "",
                    "accuracy": "",
                    "num_steps_current_round": "",
                    "error": "",
                }
                try:
                    metrics = run_fl_smoke(split_file=split_path, dataset_root=dataset_root)
                    row.update(
                        {
                            "status": "passed",
                            "train_loss": f"{metrics['train_loss']:.6f}",
                            "val_loss": f"{metrics['val_loss']:.6f}",
                            "accuracy": f"{metrics['accuracy']:.6f}",
                            "num_steps_current_round": f"{metrics['num_steps_current_round']:.1f}",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    findings.append(
                        Finding(
                            finding_id=f"FL-SMOKE-{target['n_clients']}-{target['alpha']}",
                            severity="P1",
                            confidence=0.95,
                            combo=f"FL smoke {split_path.name}",
                            evidence=row["error"],
                            root_cause="Generated multivariate split failed downstream FL classification smoke.",
                            fix="Inspect datamodule parsing, split schema, and engine compatibility.",
                            test_delta="Add this split to stable FL smoke regression set.",
                        )
                    )
                fl_rows.append(row)

    # ---------------------------------------------------------------------
    # Phase 5: reporting
    # ---------------------------------------------------------------------
    log.info("Phase 5: writing reports")
    write_csv(audit_dir / "run_results.csv", run_rows, RUN_RESULTS_COLUMNS)
    write_csv(
        audit_dir / "multivariate_validation.csv",
        multivariate_rows,
        list(multivariate_rows[0].keys()) if multivariate_rows else ["file"],
    )
    write_csv(
        audit_dir / "fl_smoke_results.csv",
        fl_rows,
        list(fl_rows[0].keys()) if fl_rows else ["split_file"],
    )
    write_csv(audit_dir / "backlog.csv", [f.to_backlog_row() for f in findings], BACKLOG_COLUMNS)

    failures_lines = ["# Failures", ""]
    open_findings = [f for f in findings if f.status == "open"]
    if not open_findings:
        failures_lines.append("No open findings.")
    else:
        severity_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        for finding in sorted(open_findings, key=lambda f: severity_order.get(f.severity, 9)):
            failures_lines.extend(
                [
                    f"## {finding.finding_id} [{finding.severity}]",
                    f"- Combo: `{finding.combo}`",
                    f"- Evidence: {finding.evidence}",
                    f"- Root cause: {finding.root_cause}",
                    f"- Fix: {finding.fix}",
                    f"- Test delta: {finding.test_delta}",
                    f"- Confidence: {finding.confidence:.2f}",
                    "",
                ]
            )
    (audit_dir / "failures.md").write_text("\n".join(failures_lines) + "\n", encoding="utf-8")

    multivariate_report_lines = [
        "# Multivariate Report",
        "",
        f"- Generation return code: {rc}",
        f"- Low-alpha specialization mean (alpha=0.1): {low_alpha_mean:.6f}",
        f"- High-alpha specialization mean (alpha=5.0): {high_alpha_mean:.6f}",
        f"- Heterogeneity order check (low > high): {heterogeneity_order_ok}",
        "",
        "## Per-run validation",
    ]
    for row in multivariate_rows:
        multivariate_report_lines.append(
            f"- `{Path(row['file']).name}`: status={row['status']} no_overlap={row['no_overlap']} "
            f"coverage={row['full_coverage']} min_samples_ok={row['min_samples_ok']} "
            f"specialization={row['specialization_score']}"
        )
    (audit_dir / "multivariate_report.md").write_text("\n".join(multivariate_report_lines) + "\n", encoding="utf-8")

    fl_report_lines = ["# FL Smoke Report", ""]
    for row in fl_rows:
        fl_report_lines.append(
            f"- `{Path(row['split_file']).name}`: status={row['status']} "
            f"train_loss={row['train_loss']} val_loss={row['val_loss']} "
            f"accuracy={row['accuracy']} steps={row['num_steps_current_round']} error={row['error']}"
        )
    (audit_dir / "fl_smoke_report.md").write_text("\n".join(fl_report_lines) + "\n", encoding="utf-8")

    passed_runs = len([r for r in run_rows if r["status"] == "passed"])
    failed_runs = len([r for r in run_rows if r["status"] != "passed"])
    open_p0 = len([f for f in open_findings if f.severity == "P0"])
    open_p1 = len([f for f in open_findings if f.severity == "P1"])
    closure_lines = [
        "# Closure Report",
        "",
        f"- Timestamp (UTC): {now_utc()}",
        f"- Embedding runs passed: {passed_runs}",
        f"- Embedding runs failed: {failed_runs}",
        f"- Open findings: {len(open_findings)}",
        f"- Open P0: {open_p0}",
        f"- Open P1: {open_p1}",
        "",
        "## Reproducibility Checks",
    ]
    for check in repro_checks:
        closure_lines.append(f"- {check['check']}: {check['result']} ({check['a']} vs {check['b']})")
    closure_lines.extend(
        [
            "",
            "## Cache Checks",
            f"- cache_hit_before: {cache_checks['cache_hit_before']}",
            f"- cache_hit_after: {cache_checks['cache_hit_after']}",
            f"- shape_equal: {cache_checks['shape_equal']}",
            f"- values_equal: {cache_checks['values_equal']}",
            f"- first_duration_sec: {cache_checks['first_duration_sec']}",
            f"- second_duration_sec: {cache_checks['second_duration_sec']}",
            "",
            "## Gate Summary",
            f"- no open P0: {open_p0 == 0}",
            f"- no open P1 without defer rationale: {open_p1 == 0}",
        ]
    )
    (audit_dir / "closure_report.md").write_text("\n".join(closure_lines) + "\n", encoding="utf-8")

    log.info("Audit complete. Results written to %s", audit_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
