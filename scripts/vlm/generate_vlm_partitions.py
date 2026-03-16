"""
generate_vlm_partitions.py
Generates VLM-only partitions for segmentation datasets (ade20k, pascal_voc)
where label-based Dirichlet splitting is not applicable.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data_loading.dataset import BaseDataset
from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate VLM partitions for segmentation datasets."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. ade20k, pascal_voc).")
    parser.add_argument("--base_path", required=True, help="Root directory containing the dataset.")
    parser.add_argument("--output", required=True, help="Output directory for partition JSON files.")
    parser.add_argument("--cluster_labels_dir", required=True, help="Path to VLM annotations directory.")
    parser.add_argument("--n_clients", type=int, nargs="+", default=[10], help="Client counts to generate.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0, 5.0], help="Alpha values.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds.")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing files.")
    return parser.parse_args()


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
            for alpha in args.alphas:
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
                    splits = splitter._make_serializable(splitter.get_client_splits())
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(splits, f, indent=2)
                    log.info("Saved: %s", output_path.name)
                except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
                    log.error("Failed (alpha=%s): %s", alpha, exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
