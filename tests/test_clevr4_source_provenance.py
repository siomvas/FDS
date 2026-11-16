"""The Clevr-4 pipeline sources are the files that produced the released labels.

Every run manifest under resources/paper_artifacts/clevr4/labels records the
SHA-256 digests of the sources that ran its text stages. The files in this
repository must keep those digests, so copy them before changing them.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = REPO_ROOT / "resources" / "paper_artifacts" / "clevr4" / "labels"
SOURCES = {
    "pipeline": REPO_ROOT / "partitioning" / "vlm" / "clevr4_pipeline.py",
    "runner": REPO_ROOT / "scripts" / "run_clevr4_large_models.py",
    "internvl_adapter": REPO_ROOT / "partitioning" / "vlm" / "internvl_adapter.py",
}
PROMPT_DIR = REPO_ROOT / "partitioning" / "vlm" / "data" / "clevr4"
MANIFESTS = sorted(LABEL_ROOT.glob("*/*/run_manifest.json"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifests_are_present():
    assert len(MANIFESTS) == 8


@pytest.mark.parametrize("manifest_path", MANIFESTS, ids=lambda p: f"{p.parts[-3]}/{p.parts[-2]}")
def test_sources_and_prompts_match_the_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest["source_sha256s"].items():
        assert _sha256(SOURCES[name]) == digest, name
    for filename, digest in manifest["prompt_sha256"].items():
        assert _sha256(PROMPT_DIR / filename) == digest, filename
