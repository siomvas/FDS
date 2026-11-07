#!/usr/bin/env python3
"""Download and normalize the public datasets used by the FDS paper package."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree


DATASETS = ("tiny_imagenet", "food101", "pascal_voc", "ade20k")
DOWNLOAD_URLS = {
    "tiny_imagenet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "food101": "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
    "pascal_voc": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="Dataset to prepare. ADE20K requires a manually downloaded archive.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("local_data"),
        help="Root directory for prepared datasets.",
    )
    parser.add_argument(
        "--download-cache",
        type=Path,
        default=None,
        help="Directory for downloaded archives. Defaults to OUTPUT_ROOT/downloads.",
    )
    parser.add_argument(
        "--ade20k-archive",
        type=Path,
        default=None,
        help="Path to the manually downloaded ADE20K archive.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy Pascal VOC image files instead of creating symlinks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run normalization even when prepared directories already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading, extracting, or writing files.",
    )
    return parser.parse_args(argv)


def _planned(message: str) -> None:
    print(f"[plan] {message}")


def _safe_members_tar(archive: tarfile.TarFile, output_root: Path):
    output_root = output_root.resolve()
    for member in archive.getmembers():
        target = (output_root / member.name).resolve()
        if output_root != target and output_root not in target.parents:
            raise RuntimeError(f"Refusing to extract archive member outside output root: {member.name}")
        yield member


def _download(url: str, destination: Path, dry_run: bool) -> None:
    if dry_run:
        _planned(f"download {url} -> {destination}")
        return
    if destination.exists():
        print(f"[skip] archive already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, destination)


def _extract_archive(archive_path: Path, output_root: Path, dry_run: bool) -> None:
    if dry_run:
        _planned(f"extract {archive_path} -> {output_root}")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive_path}")
    suffixes = "".join(archive_path.suffixes)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_root)
    elif suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or archive_path.suffix == ".tar":
        with tarfile.open(archive_path) as archive:
            archive.extractall(output_root, members=_safe_members_tar(archive, output_root))
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_tiny_imagenet(output_root: Path, cache_dir: Path, args: argparse.Namespace) -> int:
    target = output_root / "tiny-imagenet-200"
    if target.exists() and not args.force:
        print(f"[ok] Tiny ImageNet already prepared at {target}")
        return 0
    archive = cache_dir / "tiny-imagenet-200.zip"
    _download(DOWNLOAD_URLS["tiny_imagenet"], archive, args.dry_run)
    _extract_archive(archive, output_root, args.dry_run)
    return 0


def _prepare_food101(output_root: Path, cache_dir: Path, args: argparse.Namespace) -> int:
    target = output_root / "food-101"
    if target.exists() and not args.force:
        print(f"[ok] Food-101 already prepared at {target}")
        return 0
    archive = cache_dir / "food-101.tar.gz"
    _download(DOWNLOAD_URLS["food101"], archive, args.dry_run)
    _extract_archive(archive, output_root, args.dry_run)
    return 0


def _voc_rows_from_xml(annotation_path: Path) -> list[str]:
    root = ElementTree.parse(annotation_path).getroot()
    rows: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        try:
            xmin = float(box.findtext("xmin", ""))
            ymin = float(box.findtext("ymin", ""))
            xmax = float(box.findtext("xmax", ""))
            ymax = float(box.findtext("ymax", ""))
        except ValueError:
            continue
        rows.append(f"{name} {xmin:.6g} {ymin:.6g} {xmax:.6g} {ymax:.6g}")
    return rows


def _normalize_voc2012(voc_root: Path, output_root: Path, *, copy: bool, dry_run: bool) -> None:
    target_root = output_root / "VOC2012"
    if dry_run:
        _planned(f"normalize Pascal VOC from {voc_root} -> {target_root}")
        return

    image_sets = voc_root / "ImageSets" / "Main"
    jpeg_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"
    for required in (image_sets / "train.txt", image_sets / "val.txt", jpeg_dir, annotation_dir):
        if not required.exists():
            raise FileNotFoundError(f"Pascal VOC source is missing {required}")

    for split in ("train", "val"):
        split_file = image_sets / f"{split}.txt"
        image_out = target_root / "images" / split
        label_out = target_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        names = [line.strip().split()[0] for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for name in names:
            src_image = jpeg_dir / f"{name}.jpg"
            src_annotation = annotation_dir / f"{name}.xml"
            if not src_image.exists() or not src_annotation.exists():
                raise FileNotFoundError(f"Pascal VOC sample is incomplete: {name}")
            _link_or_copy(src_image, image_out / src_image.name, copy=copy)
            rows = _voc_rows_from_xml(src_annotation)
            (label_out / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _prepare_pascal_voc(output_root: Path, cache_dir: Path, args: argparse.Namespace) -> int:
    target = output_root / "VOC2012" / "images" / "train"
    if target.exists() and not args.force:
        print(f"[ok] Pascal VOC already prepared at {output_root / 'VOC2012'}")
        return 0
    archive = cache_dir / "VOCtrainval_11-May-2012.tar"
    _download(DOWNLOAD_URLS["pascal_voc"], archive, args.dry_run)
    _extract_archive(archive, output_root, args.dry_run)
    source_root = output_root / "VOCdevkit" / "VOC2012"
    _normalize_voc2012(source_root, output_root, copy=args.copy, dry_run=args.dry_run)
    return 0


def _find_ade2021_root(output_root: Path) -> Path | None:
    expected = output_root / "ADE20K" / "ADE20K_2021_17_01"
    if expected.exists():
        return expected
    for candidate in output_root.rglob("ADE20K_2021_17_01"):
        if candidate.is_dir():
            return candidate
    return None


def _prepare_ade20k(output_root: Path, args: argparse.Namespace, *, required: bool) -> int:
    expected_images = output_root / "ADE20K" / "ADE20K_2021_17_01" / "images" / "ADE" / "training"
    if expected_images.exists() and not args.force:
        print(f"[ok] ADE20K already prepared at {expected_images.parents[3]}")
        return 0
    if args.ade20k_archive is None:
        message = (
            "ADE20K requires the official manually downloaded archive. "
            "Register/download it from the ADE20K project site, then rerun with "
            "--ade20k-archive /path/to/ADE20K_2021_17_01.zip."
        )
        if required:
            print(f"[error] {message}", file=sys.stderr)
            return 2
        print(f"[skip] {message}")
        return 0
    archive = args.ade20k_archive
    if not archive.exists() and not args.dry_run:
        print(f"[error] ADE20K archive not found: {archive}", file=sys.stderr)
        return 2
    _extract_archive(archive, output_root, args.dry_run)
    if args.dry_run:
        return 0

    discovered = _find_ade2021_root(output_root)
    if discovered is None:
        raise FileNotFoundError(
            "Could not find ADE20K_2021_17_01 after extraction. "
            "Use the official ADE20K 2021 archive or normalize manually."
        )

    canonical = output_root / "ADE20K" / "ADE20K_2021_17_01"
    if discovered.resolve() != canonical.resolve():
        canonical.parent.mkdir(parents=True, exist_ok=True)
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.move(str(discovered), str(canonical))

    if not expected_images.exists():
        raise FileNotFoundError(f"ADE20K normalized layout is missing {expected_images}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root
    cache_dir = args.download_cache or output_root / "downloads"
    requested = DATASETS if args.dataset == "all" else (args.dataset,)

    if args.dry_run:
        _planned(f"prepare {', '.join(requested)} under {output_root}")

    status = 0
    for dataset in requested:
        if dataset == "tiny_imagenet":
            status = max(status, _prepare_tiny_imagenet(output_root, cache_dir, args))
        elif dataset == "food101":
            status = max(status, _prepare_food101(output_root, cache_dir, args))
        elif dataset == "pascal_voc":
            status = max(status, _prepare_pascal_voc(output_root, cache_dir, args))
        elif dataset == "ade20k":
            status = max(status, _prepare_ade20k(output_root, args, required=args.dataset == "ade20k"))
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
