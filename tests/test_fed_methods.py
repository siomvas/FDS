from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from fl_training.fed_methods import (
    MethodContext,
    ScaffoldHook,
    method_hook_from_config,
    validate_method_task_compatibility,
)


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


# ---------------------------------------------------------------------------
# ScaffoldHook tests
# ---------------------------------------------------------------------------

def _make_ctx(**overrides):
    defaults = dict(
        method_name="scaffold",
        round_idx=0,
        global_params={},
        input_meta={},
    )
    defaults.update(overrides)
    return MethodContext(**defaults)


def _param_dicts(model: nn.Module):
    """Return zero-valued c_global and c_local dicts matching model.named_parameters()."""
    c_global = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    c_local = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    return c_global, c_local


@pytest.fixture()
def mock_helper():
    helper = MagicMock()
    helper.cnt = 0
    return helper


@pytest.fixture()
def scaffold_hook(mock_helper):
    with patch("nvflare.app_opt.pt.scaffold.PTScaffoldHelper", return_value=mock_helper):
        hook = ScaffoldHook()
    return hook


def test_scaffold_hook_initializes_helper_only_once(scaffold_hook, mock_helper):
    model = nn.Linear(4, 2)
    c_global, c_local = _param_dicts(model)
    mock_helper.get_params.return_value = (c_global, c_local)

    ctx = _make_ctx()
    scaffold_hook.on_round_start(model, ctx)
    scaffold_hook.on_round_start(model, ctx)

    mock_helper.init.assert_called_once_with(model)


def test_scaffold_hook_round_start_stores_global_copy_on_cpu(scaffold_hook, mock_helper):
    model = nn.Linear(4, 2)
    c_global, c_local = _param_dicts(model)
    mock_helper.get_params.return_value = (c_global, c_local)

    scaffold_hook.on_round_start(model, _make_ctx())

    assert scaffold_hook.global_model_copy is not None
    devices = {p.device.type for p in scaffold_hook.global_model_copy.parameters()}
    assert devices == {"cpu"}


def test_scaffold_hook_round_start_loads_global_controls_when_present(scaffold_hook, mock_helper):
    from nvflare.app_common.app_constant import AlgorithmConstants

    model = nn.Linear(4, 2)
    c_global, c_local = _param_dicts(model)
    mock_helper.get_params.return_value = (c_global, c_local)

    ctrl = {name: p.numpy() for name, p in model.named_parameters()}
    ctx = _make_ctx(input_meta={AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: ctrl})
    scaffold_hook.on_round_start(model, ctx)

    mock_helper.load_global_controls.assert_called_once()


def test_scaffold_hook_round_start_skips_load_global_controls_when_absent(scaffold_hook, mock_helper):
    model = nn.Linear(4, 2)
    c_global, c_local = _param_dicts(model)
    mock_helper.get_params.return_value = (c_global, c_local)

    scaffold_hook.on_round_start(model, _make_ctx())

    mock_helper.load_global_controls.assert_not_called()


def test_scaffold_hook_step_end_applies_vectorised_correction(scaffold_hook, mock_helper):
    model = nn.Linear(4, 2, bias=False)
    # c_global = ones, c_local = zeros → correction = ones → param -= lr * ones
    c_global = {"weight": torch.ones(2, 4)}
    c_local = {"weight": torch.zeros(2, 4)}
    mock_helper.get_params.return_value = (c_global, c_local)

    scaffold_hook.on_round_start(model, _make_ctx())
    initial_weight = model.weight.data.clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaffold_hook.on_step_end(model=model, optimizer=optimizer, ctx=_make_ctx())

    expected = initial_weight - 0.1 * torch.ones(2, 4)
    assert torch.allclose(model.weight.data, expected, atol=1e-6)


def test_scaffold_hook_step_end_increments_cnt(scaffold_hook, mock_helper):
    model = nn.Linear(4, 2)
    c_global, c_local = _param_dicts(model)
    mock_helper.get_params.return_value = (c_global, c_local)
    mock_helper.cnt = 0

    scaffold_hook.on_round_start(model, _make_ctx())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(3):
        scaffold_hook.on_step_end(model=model, optimizer=optimizer, ctx=_make_ctx())

    assert mock_helper.cnt == 3


def test_scaffold_hook_round_end_warns_and_skips_when_cnt_zero(scaffold_hook, mock_helper):
    model = nn.Linear(4, 2)
    c_global, c_local = _param_dicts(model)
    mock_helper.get_params.return_value = (c_global, c_local)
    mock_helper.cnt = 0

    scaffold_hook.on_round_start(model, _make_ctx())
    scaffold_hook.on_round_end(model, _make_ctx())  # should not raise

    mock_helper.terms_update.assert_not_called()


def test_scaffold_hook_round_end_calls_terms_update(scaffold_hook, mock_helper):
    model = nn.Linear(4, 2)
    c_global, c_local = _param_dicts(model)
    mock_helper.get_params.return_value = (c_global, c_local)
    mock_helper.cnt = 5  # non-zero so terms_update is invoked

    scaffold_hook.on_round_start(model, _make_ctx())
    scaffold_hook.on_round_end(model, _make_ctx())

    mock_helper.terms_update.assert_called_once()


def test_scaffold_hook_round_end_offloads_global_copy_to_cpu(scaffold_hook, mock_helper):
    model = nn.Linear(4, 2)
    c_global, c_local = _param_dicts(model)
    mock_helper.get_params.return_value = (c_global, c_local)
    mock_helper.cnt = 1

    scaffold_hook.on_round_start(model, _make_ctx())
    scaffold_hook.on_round_end(model, _make_ctx())

    # global_model_copy should be back on CPU after round end
    devices = {p.device.type for p in scaffold_hook.global_model_copy.parameters()}
    assert devices == {"cpu"}


def test_scaffold_hook_export_round_meta_returns_ctrl_diff(scaffold_hook, mock_helper):
    from nvflare.app_common.app_constant import AlgorithmConstants

    model = nn.Linear(4, 2)
    ctrl_diff = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    mock_helper.get_delta_controls.return_value = ctrl_diff

    meta = scaffold_hook.export_round_meta(
        model=model, ctx=_make_ctx(), executed_steps=10, train_samples=64
    )

    assert AlgorithmConstants.SCAFFOLD_CTRL_DIFF in meta


def test_scaffold_hook_export_round_meta_handles_delta_controls_failure(scaffold_hook, mock_helper):
    from nvflare.app_common.app_constant import AlgorithmConstants

    model = nn.Linear(4, 2)
    mock_helper.get_delta_controls.side_effect = ValueError("no steps recorded")

    meta = scaffold_hook.export_round_meta(
        model=model, ctx=_make_ctx(), executed_steps=0, train_samples=0
    )

    assert meta[AlgorithmConstants.SCAFFOLD_CTRL_DIFF] == {}


def test_scaffold_hook_export_round_meta_filters_integer_buffers(scaffold_hook, mock_helper):
    from nvflare.app_common.app_constant import AlgorithmConstants

    model = nn.BatchNorm1d(4)  # has num_batches_tracked (int64 buffer)
    # Provide ctrl_diff that includes both float params and the int buffer
    ctrl_diff = {
        "weight": torch.zeros(4),
        "bias": torch.zeros(4),
        "running_mean": torch.zeros(4),
        "running_var": torch.ones(4),
        "num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }
    mock_helper.get_delta_controls.return_value = ctrl_diff

    meta = scaffold_hook.export_round_meta(
        model=model, ctx=_make_ctx(), executed_steps=5, train_samples=32
    )

    result = meta[AlgorithmConstants.SCAFFOLD_CTRL_DIFF]
    assert "num_batches_tracked" not in result
    assert "weight" in result
    assert "bias" in result
