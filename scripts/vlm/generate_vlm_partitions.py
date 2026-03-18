"""
generate_vlm_partitions.py
Generates VLM partitions for datasets where label-based Dirichlet splitting
is not applicable (e.g. ade20k, pascal_voc). Optionally also generates
Random and/or Dirichlet baseline splits alongside VLM.

Examples
--------
VLM only (default):
    python scripts/vlm/generate_vlm_partitions.py \\
        --dataset ade20k \\
        --base_path /path/to/data \\
        --output ./partitions/ade20k \\
        --cluster_labels_dir ./partitioning/vlm/data

VLM + Random baseline:
    python scripts/vlm/generate_vlm_partitions.py \\
        --dataset ade20k \\
        --base_path /path/to/data \\
        --output ./partitions/ade20k \\
        --cluster_labels_dir ./partitioning/vlm/data \\
        --baseline random

VLM + Dirichlet + Random:
    python scripts/vlm/generate_vlm_partitions.py \\
        --dataset food101 \\
        --base_path /path/to/data \\
        --output ./partitions/food101 \\
        --cluster_labels_dir ./partitioning/vlm/data \\
        --baseline random dirichlet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data_loading.dataset import BaseDataset
from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from partitioning.random_splitter import RandomSplitter

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate VLM partitions, with optional Random/Dirichlet baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. ade20k, pascal_voc, food101).")
    parser.add_argument("--base_path", required=True, help="Root directory containing the dataset.")
    parser.add_argument("--output", required=True, help="Output directory for partition JSON files.")
    parser.add_argument("--cluster_labels_dir", required=True, help="Path to VLM annotations directory.")
    parser.add_argument("--n_clients", type=int, nargs="+", default=[10], help="Client counts to generate (default: 10).")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0, 5.0], help="Alpha values for VLM (and Dirichlet if enabled).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds.")
    parser.add_argument(
        "--baseline",
        type=str,
        nargs="+",
        choices=["random", "dirichlet"],
        default=[],
        help="Additional baseline splits to generate alongside VLM. Choices: random, dirichlet.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config without writing files.")
    return parser.parse_args()


def save_splits(splitter, output_path: Path, label: str) -> None:
    splits = splitter._make_serializable(splitter.get_client_splits())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    log.info("Saved %s: %s", label, output_path.name)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    output_dir = Path(args.output)
    vlm_path = Path(args.cluster_labels_dir)

    if args.dry_run:
        log.info("[dry-run] Dataset:            %s", args.dataset)
        log.info("[dry-run] Base path:           %s", args.base_path)
        log.info("[dry-run] Output:              %s", output_dir)
        log.info("[dry-run] Cluster labels dir:  %s", vlm_path)
        log.info("[dry-run] n_clients:           %s", args.n_clients)
        log.info("[dry-run] alphas:              %s", args.alphas)
        log.info("[dry-run] seeds:               %s", args.seeds)
        log.info("[dry-run] baselines:           %s", args.baseline or "none")
        return 0

    if not vlm_path.exists():
        log.error("Cluster labels directory does not exist: %s", vlm_path)
        return 1

    log.info("Loading datasets...")
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    val_dataset = BaseDataset(args.dataset, args.base_path, "val")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for n_clients in args.n_clients:

            # --- Random baseline (one per seed/n_clients, no alpha) ---
            if "random" in args.baseline:
                output_path = output_dir / f"random_n{n_clients}_seed{seed}.json"
                if output_path.exists():
                    log.info("Skipping (already exists): %s", output_path.name)
                else:
                    log.info("Generating Random split (n_clients=%s, seed=%s)...", n_clients, seed)
                    try:
                        splitter = RandomSplitter(
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            n_clients=n_clients,
                            valset_type="global",
                            seed=seed,
                        )
                        save_splits(splitter, output_path, "Random")
                    except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
                        log.error("Failed Random split: %s", exc)

            for alpha in args.alphas:

                # --- Dirichlet baseline ---
                if "dirichlet" in args.baseline:
                    output_path = output_dir / f"dirichlet_n{n_clients}_alpha{alpha}_seed{seed}.json"
                    if output_path.exists():
                        log.info("Skipping (already exists): %s", output_path.name)
                    else:
                        log.info("Generating Dirichlet split (alpha=%s, n_clients=%s, seed=%s)...", alpha, n_clients, seed)
                        try:
                            from partitioning.dirichlet_splitter import DirichletSplitter
                            splitter = DirichletSplitter(
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                n_clients=n_clients,
                                heterogeneity=alpha,
                                valset_type="global",
                                seed=seed,
                            )
                            save_splits(splitter, output_path, "Dirichlet")
                        except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
                            log.error("Failed Dirichlet split (alpha=%s): %s", alpha, exc)

                # --- VLM ---
                output_path = output_dir / f"vlm_n{n_clients}_alpha{alpha}_seed{seed}.json"
                if output_path.exists():
                    log.info("Skipping (already exists): %s", output_path.name)
                    continue

                log.info("Generating VLM split (alpha=%s, n_clients=%s, seed=%s)...", alpha, n_clients, seed)
                try:
                    splitter = MultivariateDirichletSplitter(
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        n_clients=n_clients,
                        heterogeneity=alpha,
                        cluster_labels_dir=str(vlm_path),
                        valset_type="global",
                        seed=seed,
                        min_samples_per_client=10,
                    )
                    save_splits(splitter, output_path, "VLM")
                except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
                    log.error("Failed VLM split (alpha=%s): %s", alpha, exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
