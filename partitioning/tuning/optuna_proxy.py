from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .embedding_optuna import run_cli as run_embedding_optuna_cli


@dataclass(frozen=True)
class EmbeddingStudyPreset:
    key: str
    dataset_name: str
    study_mode: str
    default_target_n_clients: int | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    def default_study_name(self) -> str:
        return f"{self.dataset_name}_{self.study_mode}_embedding_optuna"


def build_arg_parser(preset: EmbeddingStudyPreset) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Preset launcher for embedding partition Optuna: {preset.key}"
    )
    parser.add_argument("--base_path", required=True, type=str, help="Raw dataset base path.")
    parser.add_argument("--study_dir", required=True, type=str, help="Output directory for study DB and artifacts.")
    parser.add_argument("--study_name", default=None, type=str, help="Optuna study name override.")
    parser.add_argument("--n_trials", default=100, type=int, help="Number of Optuna trials to run.")
    parser.add_argument("--timeout", default=None, type=int, help="Optional walltime limit in seconds.")
    parser.add_argument("--seed", default=42, type=int, help="Base random seed.")
    parser.add_argument("--num_workers", default=0, type=int, help="Worker count for embedding and metrics.")
    parser.add_argument("--device", default=None, type=str, help="Optional device override.")
    parser.add_argument("--cache_dir", default=None, type=str, help="Optional cache directory override.")
    parser.add_argument("--proxy", default=None, type=str, help="Optional proxy override.")
    parser.add_argument("--embed_batch_size", default=100, type=int, help="Embedding extraction batch size.")
    parser.add_argument("--metric_batch_size", default=128, type=int, help="Metric feature batch size.")
    parser.add_argument("--artifact_top_k", default=10, type=int, help="How many completed trial artifacts to retain.")
    parser.add_argument(
        "--target_n_clients",
        default=preset.default_target_n_clients,
        type=int,
        help="Target client count for fixed-mode presets.",
    )
    parser.add_argument("--min_client_size", default=20, type=int, help="Minimum acceptable client size.")
    parser.add_argument(
        "--min_discovered_clients",
        default=2,
        type=int,
        help="Minimum acceptable discovered client count in discovery mode.",
    )
    parser.add_argument(
        "--max_discovered_clients",
        default=50,
        type=int,
        help="Maximum acceptable discovered client count in discovery mode.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    parser.add_argument("--no_cache", action="store_true", help="Disable embedding and metric caching.")
    return parser


def build_embedding_optuna_argv(
    preset: EmbeddingStudyPreset,
    *,
    base_path: Path,
    study_dir: Path,
    study_name: str | None,
    n_trials: int,
    timeout: int | None,
    seed: int,
    num_workers: int,
    device: str | None,
    cache_dir: Path | None,
    proxy: str | None,
    embed_batch_size: int,
    metric_batch_size: int,
    artifact_top_k: int,
    target_n_clients: int | None,
    min_client_size: int,
    min_discovered_clients: int,
    max_discovered_clients: int,
    no_cache: bool,
    dry_run: bool,
) -> list[str]:
    argv = [
        "--dataset",
        preset.dataset_name,
        "--base_path",
        str(base_path),
        "--study_dir",
        str(study_dir),
        "--study_mode",
        preset.study_mode,
        "--n_trials",
        str(int(n_trials)),
        "--seed",
        str(int(seed)),
        "--num_workers",
        str(int(num_workers)),
        "--embed_batch_size",
        str(int(embed_batch_size)),
        "--metric_batch_size",
        str(int(metric_batch_size)),
        "--artifact_top_k",
        str(int(artifact_top_k)),
        "--min_client_size",
        str(int(min_client_size)),
        "--min_discovered_clients",
        str(int(min_discovered_clients)),
        "--max_discovered_clients",
        str(int(max_discovered_clients)),
    ]
    if study_name:
        argv.extend(["--study_name", study_name])
    if timeout is not None:
        argv.extend(["--timeout", str(int(timeout))])
    if device:
        argv.extend(["--device", device])
    if cache_dir is not None:
        argv.extend(["--cache_dir", str(cache_dir)])
    if proxy is not None:
        argv.extend(["--proxy", proxy])
    if preset.study_mode == "fixed":
        if target_n_clients is None:
            raise ValueError(f"Preset '{preset.key}' requires target_n_clients for fixed-mode execution.")
        argv.extend(["--target_n_clients", str(int(target_n_clients))])
    if no_cache:
        argv.append("--no_cache")
    if dry_run:
        argv.append("--dry-run")

    for key, value in preset.extra_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(item) for item in value)
            continue
        argv.extend([flag, str(value)])
    return argv


def run_preset_cli(preset: EmbeddingStudyPreset, argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser(preset).parse_args(list(argv) if argv is not None else None)
    return run_embedding_optuna_cli(
        build_embedding_optuna_argv(
            preset,
            base_path=Path(args.base_path).expanduser().resolve(),
            study_dir=Path(args.study_dir).expanduser().resolve(),
            study_name=args.study_name,
            n_trials=int(args.n_trials),
            timeout=args.timeout,
            seed=int(args.seed),
            num_workers=int(args.num_workers),
            device=args.device,
            cache_dir=Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None,
            proxy=args.proxy,
            embed_batch_size=int(args.embed_batch_size),
            metric_batch_size=int(args.metric_batch_size),
            artifact_top_k=int(args.artifact_top_k),
            target_n_clients=args.target_n_clients,
            min_client_size=int(args.min_client_size),
            min_discovered_clients=int(args.min_discovered_clients),
            max_discovered_clients=int(args.max_discovered_clients),
            no_cache=bool(args.no_cache),
            dry_run=bool(args.dry_run),
        )
    )
