import csv
import json
from pathlib import Path

import pytest

from research import write_results_index
from scripts import run_embedding_matrix_audit as audit
from scripts.run_embedding_matrix_audit import load_selected_run_ids, validate_split_invariants


def test_validate_split_invariants_requires_clients_mapping():
    with pytest.raises(KeyError, match="missing a valid 'clients' mapping"):
        validate_split_invariants({}, expected_train_count=0)


def test_validate_split_invariants_requires_train_image_fields():
    payload = {
        "clients": {
            "client_0": {
                "train": [{}],
            }
        }
    }
    with pytest.raises(KeyError, match="without an 'image' field"):
        validate_split_invariants(payload, expected_train_count=1)


def test_load_selected_run_ids_requires_named_csv_column(tmp_path):
    csv_path = tmp_path / "run_ids.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["wrong"])
        writer.writeheader()
        writer.writerow({"wrong": "tier1_example"})

    class Args:
        only_run_id = []
        run_ids_file = str(csv_path)

    with pytest.raises(ValueError, match="must contain a 'run_id' column"):
        load_selected_run_ids(Args())


class _Dataset:
    dataset_name = "tiny_imagenet"
    requested_dataset_name = "tiny_imagenet"

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_run_embedding_combo_writes_canonical_audit_artifacts(tmp_path, monkeypatch):
    class DummySplitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.n_clients = 2
            self.cluster_info_ = {"num_clusters": 2}
            self._splits = {
                "clients": {
                    "client_0": {"train": [{"image": "img_a.jpg"}]},
                    "client_1": {"train": [{"image": "img_b.jpg"}]},
                }
            }

        def get_client_splits(self):
            return self._splits

        def validate_splits(self, splits):
            assert splits == self._splits

        def store_client_splits(self, path, format="json"):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(self._splits), encoding="utf-8")

        def get_split_summary(self):
            return {"n_clients": 2, "total_train_samples": 2, "global_val_samples": 1}

    monkeypatch.setattr(audit, "EmbeddingSplitter", DummySplitter)

    train = _Dataset([("img_a.jpg", 0), ("img_b.jpg", 1)])
    val = _Dataset([("img_val.jpg", 0)])
    findings = []

    row = audit.run_embedding_combo(
        run_id="tier1_fixture_hdbscan_s42",
        phase="embedding_matrix",
        tier="tier1",
        extractor="fixture-extractor",
        clustering="hdbscan",
        seed=42,
        train_full=train,
        val_full=val,
        subset_train=None,
        subset_val=None,
        output_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        dataset_name="tiny_imagenet",
        dataset_root=str(tmp_path),
        device="cpu",
        batch_size=4,
        num_workers=0,
        proxy=None,
        findings=findings,
    )

    assert row["status"] == "success"
    run_dir = tmp_path / "tier1_fixture_hdbscan_s42"
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "result.json").exists()

    rows = write_results_index(tmp_path, result_type="audit_run")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "tier1_fixture_hdbscan_s42"
