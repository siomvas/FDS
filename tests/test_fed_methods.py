from __future__ import annotations

import pytest

from fl_training.fed_methods import method_hook_from_config, validate_method_task_compatibility


def test_feddecorr_rejects_non_classification_tasks():
    cfg = {"fed": {"method": "feddecorr", "args": {}}, "training": {"task_type": "segmentation"}}
    with pytest.raises(ValueError, match="does not support"):
        validate_method_task_compatibility(cfg, "segmentation")


def test_fednova_hook_exports_tau_metadata():
    cfg = {"fed": {"method": "fednova", "args": {}}, "training": {"task_type": "classification"}}
    hook = method_hook_from_config(cfg, "classification")
    meta = hook.export_round_meta(model=None, ctx=None, executed_steps=5, train_samples=12)
    assert meta["fednova_tau"] == 5.0
    assert meta["train_samples"] == 12.0
