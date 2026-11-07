"""Release checks for retained reviewer VLM label artifacts."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_LABEL_ROOT = REPO_ROOT / "resources" / "vlm_labels"
PRIVATE_PATH_RE = re.compile(r"/(?:Users|users|home)/[^/]+")


def _open_label_file(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def test_retained_vlm_labels_are_anonymous_release_schema():
    manifest = json.loads((VLM_LABEL_ROOT / "manifest.json").read_text(encoding="utf-8"))

    for dataset_name, dataset in manifest["datasets"].items():
        for criterion_name, criterion in dataset["criteria"].items():
            label_path = (
                VLM_LABEL_ROOT
                / dataset_name
                / criterion["directory"]
                / criterion["labels"]
            )
            rows = 0

            with _open_label_file(label_path) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    rows += 1
                    record = json.loads(line)
                    assert set(record) == {"image", "class"}, (
                        f"{label_path}:{line_number} has non-release fields: "
                        f"{sorted(record)}"
                    )
                    assert not str(record["image"]).startswith("/")
                    assert not PRIVATE_PATH_RE.search(json.dumps(record))

            assert rows == criterion["label_count"], (
                f"{dataset_name}/{criterion_name} row count changed: "
                f"{rows} != {criterion['label_count']}"
            )
