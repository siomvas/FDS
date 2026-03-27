from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download flwrlabs/fed-isic2019 via Flower Datasets and export a "
            "canonical local layout with images/ and metadata.csv."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "local_data" / "FedISIC2019",
        help="Dataset export root.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache dir. Defaults to <output-dir>/hf_cache.",
    )
    return parser


def _require_flower():
    try:
        from flwr_datasets import FederatedDataset
    except ImportError as exc:
        raise SystemExit(
            "flwr-datasets is required for this downloader. "
            "Install it in a temporary environment, run this script, then uninstall it."
        ) from exc
    return FederatedDataset


def export_fed_isic2019(output_dir: Path, cache_dir: Path | None = None) -> Path:
    FederatedDataset = _require_flower()

    output_dir = output_dir.expanduser().resolve()
    cache_dir = (cache_dir or (output_dir / "hf_cache")).expanduser().resolve()
    images_root = output_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fds = FederatedDataset(
        dataset="flwrlabs/fed-isic2019",
        partitioners={"train": 1, "test": 1},
        cache_dir=str(cache_dir),
    )

    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        dataset = fds.load_split(split)
        split_dir = images_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(dataset):
            rel_path = Path("images") / split / f"{split}_{idx:05d}.jpg"
            out_path = output_dir / rel_path
            if not out_path.exists():
                sample["image"].save(out_path, format="JPEG", quality=95)
            rows.append(
                {
                    "image": str(rel_path).replace("\\", "/"),
                    "label": str(int(sample["label"])),
                    "center": str(int(sample["center"])),
                    "split": split,
                }
            )

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label", "center", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path


def main() -> int:
    args = _build_parser().parse_args()
    metadata_path = export_fed_isic2019(args.output_dir, args.cache_dir)
    print(f"Exported Fed-ISIC2019 to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
