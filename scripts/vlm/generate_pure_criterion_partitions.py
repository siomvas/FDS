#!/usr/bin/env python3
"""
generate_pure_criterion_partitions.py

Generates "pure separation" partitions for each VLM criterion:
  - 1 client per distinct label value (total separation)
  - Tests whether each criterion captures real visual heterogeneity

High FID/MMD vs random => criterion is visually meaningful.
Low FID/MMD ≈ random   => labels are not visually discriminative.

Usage:
    python scripts/vlm/generate_pure_criterion_partitions.py \
        --datasets tiny_imagenet ade20k pascal_voc \
        --output_dir /path/to/gtai/FDS/vlm_analysis/pure_criterion

Then evaluate with:
    python evaluation/evaluate_partitions.py \
        --partition_dir /path/to/gtai/FDS/vlm_analysis/pure_criterion/tiny_imagenet \
        --dataset tiny_imagenet \
        --base_path /path/to/gtai/FDS \
        --output_dir /path/to/gtai/FDS/vlm_eval/pure_criterion/tiny_imagenet \
        --cache_dir /path/to/gtai/FDS/embedding_cache \
        --feature_model dinov2-base \
        --feature_only
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# ── Dataset configs ──────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "tiny_imagenet": {
        "vlm_data_dir": "partitioning/vlm/data/tiny_imagenet",
        "random_partition": "/path/to/gtai/FDS/vlm_partitions/tiny_imagenet/random_n10_seed42.json",
        # Priority order: most → least semantically meaningful
        "criteria": ["main_object", "location", "color", "size", "time"],
        "anchor": "tiny-imagenet-200",
    },
    "ade20k": {
        "vlm_data_dir": "partitioning/vlm/data/ade20k",
        "random_partition": "/path/to/gtai/FDS/vlm_partitions/ade20k/random_n10_seed42.json",
        "criteria": ["scene", "main_object", "camera", "light", "color", "size"],
        "anchor": "ADE20K",
    },
    "pascal_voc": {
        "vlm_data_dir": "partitioning/vlm/data/PascalVOC",
        "random_partition": "/path/to/gtai/FDS/vlm_partitions/pascal_voc/random_n10_seed42.json",
        "criteria": ["main_object", "location", "light", "color", "size"],
        "anchor": "VOC2012",
    },
    "food101": {
        "vlm_data_dir": "partitioning/vlm/data/food101",
        "random_partition": "/path/to/gtai/FDS/vlm_partitions/food101/random_n10_seed42.json",
        "criteria": ["cuisine", "formality", "preparation", "texture", "composition"],
        "anchor": "food101",
    },
}

# ── Path normalisation ───────────────────────────────────────────────────────

ANCHORS = ["VOC2012", "tiny-imagenet-200", "food101", "ADE20K", "bdd100k"]


def normalize_path(path: str, anchor: str | None = None) -> str:
    """Return anchor-relative path for robust cross-machine matching."""
    parts = Path(path).parts
    search = [anchor] if anchor else ANCHORS
    for a in search:
        if a in parts:
            idx = parts.index(a)
            return str(Path(*parts[idx:]))
    # Fallback: last 5 components
    return str(Path(*parts[-5:]))


# ── Core logic ───────────────────────────────────────────────────────────────

def load_valid_labels(criterion_dir: Path) -> set[str] | None:
    """Parse valid label names from step2b_result.txt ('Answer N: <label>' lines).

    Returns None if the file is absent (no filtering applied).
    """
    step2b = criterion_dir / "step2b_result.txt"
    if not step2b.exists():
        return None
    import re
    labels: set[str] = set()
    for line in step2b.read_text().splitlines():
        m = re.match(r"Answer\s+\d+:\s*(.+)", line.strip())
        if m:
            labels.add(m.group(1).strip())
    return labels if labels else None


def clean_label(raw: str, valid_labels: set[str] | None = None) -> str | None:
    """Return a clean label or None if the value cannot be mapped to a valid label.

    Strategy:
    1. If raw exactly matches a valid label → return it.
    2. Take the last non-empty line; if it matches → return it.
    3. Search the full string for any valid label (last occurrence wins).
    4. Otherwise return None (will be skipped).

    When valid_labels is None all non-empty strings are accepted as-is.
    """
    raw = raw.strip()
    if not raw:
        return None

    if valid_labels is None:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[-1] if lines else raw

    # Exact match
    if raw in valid_labels:
        return raw

    # Last non-empty line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    last = lines[-1] if lines else raw
    if last in valid_labels:
        return last

    # Search full text for any valid label (prefer longest match, take last occurrence)
    found = None
    found_pos = -1
    for lbl in sorted(valid_labels, key=len, reverse=True):
        pos = raw.rfind(lbl)
        if pos > found_pos:
            found = lbl
            found_pos = pos
    return found  # None if nothing matched


def load_criterion_labels(criterion_dir: Path) -> dict[str, str]:
    """Load {normalized_image_path: label_value} from step3_result.jsonl.

    Labels are cleaned and filtered against the step2b valid-label set when
    available; images with unresolvable labels are silently dropped.
    """
    jsonl = criterion_dir / "step3_result.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"Missing: {jsonl}")

    valid_labels = load_valid_labels(criterion_dir)
    if valid_labels:
        log.info("    Valid labels from step2b: %s", sorted(valid_labels))

    result: dict[str, str] = {}
    skipped = 0
    anchor = None
    for a in ANCHORS:
        if a in str(criterion_dir):
            anchor = a
            break

    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = normalize_path(entry["image"], anchor)
            label = clean_label(entry["class"], valid_labels)
            if label is None:
                skipped += 1
                continue
            result[key] = label

    if skipped:
        log.warning("    Skipped %d images with unresolvable labels", skipped)
    return result


def build_image_index(partition_path: Path, anchor: str) -> dict[str, dict]:
    """Build {normalized_path: {image, label}} from an existing partition."""
    with open(partition_path) as f:
        data = json.load(f)

    index: dict[str, dict] = {}
    for client_data in data["clients"].values():
        for entry in client_data.get("train", []):
            key = normalize_path(entry["image"], anchor)
            index[key] = entry
    return index


def create_pure_criterion_partition(
    image_index: dict[str, dict],
    criterion_labels: dict[str, str],
    global_val: list | None,
) -> dict:
    """Group images by criterion label; 1 client per distinct value."""
    groups: dict[str, list] = defaultdict(list)
    unmatched = 0

    for norm_path, entry in image_index.items():
        label_value = criterion_labels.get(norm_path)
        if label_value is None:
            unmatched += 1
            continue
        groups[label_value].append(entry)

    if unmatched:
        pct = 100 * unmatched / max(len(image_index), 1)
        log.warning("  %d images (%.1f%%) had no VLM annotation", unmatched, pct)

    clients: dict[str, dict] = {}
    for idx, label_value in enumerate(sorted(groups)):
        clients[f"client_{idx}"] = {"train": groups[label_value], "val": []}
        log.info("    client_%d (%s): %d images", idx, label_value, len(groups[label_value]))

    result: dict = {"clients": clients}
    if global_val is not None:
        result["global_val"] = global_val
    return result


def load_global_val(partition_path: Path) -> list | None:
    with open(partition_path) as f:
        data = json.load(f)
    return data.get("global_val")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS),
        choices=list(DATASET_CONFIGS),
        help="Datasets to process (default: all).",
    )
    parser.add_argument(
        "--output_dir",
        default="/path/to/gtai/FDS/vlm_analysis/pure_criterion",
        help="Root output directory for generated partition JSONs.",
    )
    parser.add_argument(
        "--repo_root",
        default="/path/to/FDS",
        help="Path to the FDS repo root (for vlm/data dirs).",
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        default=None,
        help="Restrict to specific criteria (default: all for each dataset).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    output_root = Path(args.output_dir)
    repo_root = Path(args.repo_root)

    for dataset_name in args.datasets:
        cfg = DATASET_CONFIGS[dataset_name]
        dataset_out = output_root / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)

        log.info("=== %s ===", dataset_name)
        anchor = cfg["anchor"]
        random_path = Path(cfg["random_partition"])
        vlm_data_root = repo_root / cfg["vlm_data_dir"]

        log.info("  Building image index from %s", random_path.name)
        image_index = build_image_index(random_path, anchor)
        global_val = load_global_val(random_path)
        log.info("  Total images in index: %d", len(image_index))

        criteria = args.criteria if args.criteria else cfg["criteria"]
        for criterion in criteria:
            criterion_dir = vlm_data_root / f"output_{criterion}_train"
            if not criterion_dir.exists():
                log.warning("  Skipping %s — directory not found: %s", criterion, criterion_dir)
                continue

            out_path = dataset_out / f"pure_{criterion}.json"
            if out_path.exists():
                log.info("  Skipping %s (already exists)", out_path.name)
                continue

            log.info("  Criterion: %s", criterion)
            try:
                criterion_labels = load_criterion_labels(criterion_dir)
                log.info("    Loaded %d VLM annotations", len(criterion_labels))
                partition = create_pure_criterion_partition(image_index, criterion_labels, global_val)
                n_clients = len(partition["clients"])
                total_assigned = sum(len(c["train"]) for c in partition["clients"].values())
                log.info("    → %d clients, %d/%d images assigned", n_clients, total_assigned, len(image_index))

                with open(out_path, "w") as f:
                    json.dump(partition, f, indent=2)
                log.info("    Saved: %s", out_path)
            except Exception as exc:
                log.error("  FAILED %s/%s: %s", dataset_name, criterion, exc)

    # Print evaluation commands
    print("\n" + "=" * 70)
    print("EVALUATION COMMANDS")
    print("=" * 70)

    base_path = "/path/to/gtai/FDS"
    cache_dir = "/path/to/gtai/FDS/embedding_cache"

    for dataset_name in args.datasets:
        dataset_dir = output_root / dataset_name
        eval_out = f"/path/to/gtai/FDS/vlm_eval/pure_criterion/{dataset_name}"
        random_part = DATASET_CONFIGS[dataset_name]["random_partition"]
        print(f"\n# {dataset_name} — pure criterion partitions")
        print(
            f"python evaluation/evaluate_partitions.py \\\n"
            f"    --partition_dir {dataset_dir} \\\n"
            f"    --dataset {dataset_name} \\\n"
            f"    --base_path {base_path} \\\n"
            f"    --output_dir {eval_out} \\\n"
            f"    --cache_dir {cache_dir} \\\n"
            f"    --feature_model dinov2-base \\\n"
            f"    --feature_only"
        )
        print(f"\n# {dataset_name} — random baseline (for comparison)")
        print(
            f"python evaluation/evaluate_partitions.py \\\n"
            f"    --partition {random_part} \\\n"
            f"    --dataset {dataset_name} \\\n"
            f"    --base_path {base_path} \\\n"
            f"    --output_dir {eval_out} \\\n"
            f"    --cache_dir {cache_dir} \\\n"
            f"    --feature_model dinov2-base \\\n"
            f"    --feature_only"
        )


if __name__ == "__main__":
    main()
