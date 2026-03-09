"""
generate_baselines.py
Generates baseline partitions (Random, Dirichlet, VLM) and saves them as JSON
to match the format of embedding-based partitions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from data_loading.dataset import BaseDataset
from partitioning.random_splitter import RandomSplitter
from partitioning.dirichlet_splitter import DirichletSplitter
from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from utils.cli_common import add_standard_cli_args

log = logging.getLogger(__name__)


def _resolve_seeds(single_seed: int | None, seeds: Sequence[int] | None) -> list[int]:
    if single_seed is not None:
        return [int(single_seed)]
    if seeds:
        return [int(seed) for seed in seeds]
    return [42]


def save_split(splitter, output_path: Path, method_name: str, alpha: float, n_clients: int, dry_run: bool) -> None:
    # Preserve full sample schema (image + task-specific fields) so generated
    # baselines are consumable by both evaluation and FL training datamodules.
    serializable_splits = splitter._make_serializable(splitter.get_client_splits())

    if dry_run:
        log.info("[dry-run] Would write %s split to %s", method_name, output_path)
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_splits, f, indent=2)
    log.info("Saved %s (n=%s, alpha=%s) to %s", method_name, n_clients, alpha, output_path.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate baseline FL partition JSON files.")
    add_standard_cli_args(
        parser,
        include_dataset=True,
        dataset_required=True,
        include_base_path=True,
        base_path_required=True,
        include_output=True,
        output_required=True,
        output_help="Output directory for generated baseline split JSON files.",
        include_dry_run=True,
    )
    parser.add_argument("--n_clients", type=int, nargs="+", default=[10], help="Client counts to generate.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5], help="Dirichlet/VLM alphas.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of random seeds. Overrides default seed list.",
    )
    parser.add_argument(
        "--seed",
        dest="single_seed",
        type=int,
        default=None,
        help="Single-seed convenience alias (overrides --seeds).",
    )
    parser.add_argument(
        "--cluster_labels_dir",
        type=str,
        default=None,
        help="Path to VLM annotations directory (required for VLM splits).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output)
    seeds = _resolve_seeds(args.single_seed, args.seeds)

    if args.dry_run:
        log.info("[dry-run] Dataset: %s", args.dataset)
        log.info("[dry-run] Base path: %s", args.base_path)
        log.info("[dry-run] Output directory: %s", output_dir)
        log.info("[dry-run] n_clients: %s", args.n_clients)
        log.info("[dry-run] alphas: %s", args.alphas)
        log.info("[dry-run] seeds: %s", seeds)
        return 0

    # Load dataset
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    val_dataset = BaseDataset(args.dataset, args.base_path, "val")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        for n_clients in args.n_clients:
            # Random (IID)
            splitter = RandomSplitter(train_dataset, val_dataset, n_clients, heterogeneity=0.0, seed=seed)
            save_split(
                splitter,
                output_dir / f"random_n{n_clients}_seed{seed}.json",
                "Random",
                0.0,
                n_clients,
                args.dry_run,
            )

            # Dirichlet (Non-IID)
            for alpha in args.alphas:
                splitter = DirichletSplitter(train_dataset, val_dataset, n_clients, heterogeneity=alpha, seed=seed)
                save_split(
                    splitter,
                    output_dir / f"dirichlet_n{n_clients}_alpha{alpha}_seed{seed}.json",
                    "Dirichlet",
                    alpha,
                    n_clients,
                    args.dry_run,
                )

            # VLM
            if not args.cluster_labels_dir:
                log.info("Skipping VLM (no --cluster_labels_dir provided).")
                continue

            vlm_path = Path(args.cluster_labels_dir)
            if not vlm_path.exists():
                log.warning("Cluster labels directory %s does not exist. Skipping VLM.", vlm_path)
                continue

            for alpha in args.alphas:
                try:
                    log.info("Generating VLM split (alpha=%s, n_clients=%s, seed=%s)", alpha, n_clients, seed)
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
                    save_split(
                        splitter,
                        output_dir / f"vlm_n{n_clients}_alpha{alpha}_seed{seed}.json",
                        "VLM",
                        alpha,
                        n_clients,
                        args.dry_run,
                    )
                except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
                    log.error("Failed VLM split generation (alpha=%s): %s", alpha, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
