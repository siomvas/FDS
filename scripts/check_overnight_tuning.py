#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _pick_study_root(studies_dir: Path, prefix: str) -> Path | None:
    if not studies_dir.exists():
        return None
    matches = sorted([path for path in studies_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)])
    return matches[-1] if matches else None


def summarize(run_root: Path) -> int:
    manifest = _load_json(run_root / "run_manifest.json")
    if manifest is None:
        print(f"Missing run manifest: {run_root / 'run_manifest.json'}")
        return 1

    studies_dir = Path(manifest["study_dir"])
    targets = [
        "tiny_imagenet_resnet18",
        "food101_convnext_small",
        "ade20k_deeplabv3_resnet50",
        "pascal_voc_fasterrcnn_resnet50_fpn_v2",
    ]

    print(f"Run root: {run_root}")
    print(f"Studies dir: {studies_dir}")
    print()

    for key in targets:
        study_root = _pick_study_root(studies_dir, key + "_optuna")
        print(f"[{key}]")
        if study_root is None:
            print("  - status: MISSING_STUDY")
            continue

        best_params = _load_json(study_root / "best_params.json")
        summary = (study_root / "summary.md")
        trials_csv = (study_root / "trials.csv")

        print(f"  - study_root: {study_root}")
        print(f"  - summary: {summary if summary.exists() else 'MISSING'}")
        print(f"  - trials_csv: {trials_csv if trials_csv.exists() else 'MISSING'}")

        if best_params and best_params.get("best_trial"):
            bt = best_params["best_trial"]
            print(f"  - best_trial: {bt.get('number')}")
            print(f"  - best_value: {bt.get('value')}")
            print(f"  - best_params: {json.dumps(bt.get('params', {}), sort_keys=True)}")
        else:
            print("  - best_trial: NONE")
        print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize overnight Optuna tuning artifacts.")
    parser.add_argument("--run-root", type=str, required=True, help="Run root emitted by run_overnight_tuning.sh")
    args = parser.parse_args()
    return summarize(Path(args.run_root).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
