import json

import pytest

from evaluation.eval_config import PartitionConfig


def test_partition_config_reads_client_count_from_partition_file(tmp_path):
    partition_path = tmp_path / "dinov2-base" / "cls_mean_norm-true_dr-umap_hdbscan_seed42.json"
    partition_path.parent.mkdir(parents=True)
    partition_path.write_text(
        json.dumps({"clients": {"client_0": {}, "client_1": {}, "client_2": {}}}),
        encoding="utf-8",
    )

    config = PartitionConfig.from_path(str(partition_path))

    assert config.method == "Embedding"
    assert config.n_clients == 3
    assert config.seed == 42


def test_partition_config_raises_when_client_mapping_is_missing(tmp_path):
    partition_path = tmp_path / "dinov2-base" / "cls_mean_norm-true_dr-umap_hdbscan_seed42.json"
    partition_path.parent.mkdir(parents=True)
    partition_path.write_text(json.dumps({"not_clients": {}}), encoding="utf-8")

    with pytest.raises(KeyError, match="missing a valid 'clients' mapping"):
        PartitionConfig.from_path(str(partition_path))
