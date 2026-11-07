"""Shared CLI argument helpers for command-line scripts.

Keeps commonly used flag names consistent across script entrypoints.
"""

from __future__ import annotations

import argparse


def add_standard_cli_args(
    parser: argparse.ArgumentParser,
    *,
    include_dataset: bool = False,
    dataset_required: bool = False,
    dataset_default: str | None = None,
    include_base_path: bool = False,
    base_path_required: bool = False,
    include_output: bool = False,
    output_required: bool = False,
    output_default: str | None = None,
    output_help: str = "Output path",
    include_seed: bool = False,
    seed_default: int = 42,
    include_dry_run: bool = True,
) -> None:
    """Attach shared CLI options to a parser.

    This helper introduces shared canonical flags:
    - ``--dataset``
    - ``--base_path``
    - ``--output`` (with compatibility alias ``--output_dir``)
    - ``--seed``
    - ``--dry-run``
    """

    if include_dataset:
        kwargs = {"type": str, "help": "Dataset name"}
        if dataset_required:
            kwargs["required"] = True
        elif dataset_default is not None:
            kwargs["default"] = dataset_default
        parser.add_argument("--dataset", **kwargs)

    if include_base_path:
        kwargs = {
            "type": str,
            "help": "Dataset base path",
        }
        if base_path_required:
            kwargs["required"] = True
        parser.add_argument("--base_path", **kwargs)

    if include_output:
        kwargs = {
            "dest": "output",
            "type": str,
            "help": output_help,
        }
        if output_required:
            kwargs["required"] = True
        elif output_default is not None:
            kwargs["default"] = output_default
        parser.add_argument("--output", "--output_dir", **kwargs)

    if include_seed:
        parser.add_argument(
            "--seed",
            type=int,
            default=seed_default,
            help=f"Random seed (default: {seed_default})",
        )

    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate arguments and print planned actions without mutating files.",
        )
