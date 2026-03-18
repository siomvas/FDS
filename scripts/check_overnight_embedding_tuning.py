#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partitioning.tuning.presets import PRESETS


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(run_root: Path) -> int:
    manifest = _load_json(run_root / "run_manifest.json")
    if manifest is None:
        print(f"Missing run manifest: {run_root / 'run_manifest.json'}")
        return 1

    studies_dir = Path(manifest["study_dir"])
    preset_keys = list(manifest.get("presets", {}).keys())

    print(f"Run root: {run_root}")
    print(f"Studies dir: {studies_dir}")
    print()

    for key in preset_keys:
        preset = PRESETS.get(key)
        print(f"[{key}]")
        if preset is None:
            print("  - status: UNKNOWN_PRESET")
            print()
            continue

        study_root = studies_dir / preset.default_study_name()
        best_trial = _load_json(study_root / "best" / "best_trial.json")
        summary = study_root / "summary.md"
        trials_csv = study_root / "trials.csv"

        print(f"  - study_root: {study_root if study_root.exists() else 'MISSING'}")
        print(f"  - summary: {summary if summary.exists() else 'MISSING'}")
        print(f"  - trials_csv: {trials_csv if trials_csv.exists() else 'MISSING'}")

        if best_trial and best_trial.get("number") is not None:
            print(f"  - best_trial: {best_trial.get('number')}")
            print(f"  - best_value: {best_trial.get('value')}")
            print(f"  - best_params: {json.dumps(best_trial.get('params', {}), sort_keys=True)}")
        else:
            print("  - best_trial: NONE")
        print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize overnight embedding Optuna tuning artifacts.")
    parser.add_argument("--run-root", type=str, required=True, help="Run root emitted by run_overnight_embedding_tuning.sh")
    args = parser.parse_args()
    return summarize(Path(args.run_root).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
