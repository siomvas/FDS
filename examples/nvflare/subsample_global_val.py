"""Keep a fixed number of global validation images per class in an FDS split file.

Usage:
    python examples/nvflare/subsample_global_val.py SPLIT.json --output OUT.json [--per-class 5] [--seed 42]

The client training sets are left unchanged. Only ``global_val`` is reduced, which makes the
per-round validation of a short CPU run affordable.
"""

import argparse
import collections
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.split.open(encoding="utf-8") as handle:
        split = json.load(handle)
    by_label = collections.defaultdict(list)
    for sample in split["global_val"]:
        by_label[sample["label"]].append(sample)
    rng = random.Random(args.seed)
    kept = []
    for label in sorted(by_label):
        kept += rng.sample(by_label[label], min(args.per_class, len(by_label[label])))
    split["global_val"] = kept
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(split, handle)
    print(f"{len(kept)} validation images from {len(by_label)} classes written to {args.output}")


if __name__ == "__main__":
    main()
