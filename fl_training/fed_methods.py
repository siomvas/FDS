from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import logging
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AlgorithmConstants, AppConstants
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.fedopt import PTFedOptModelShareableGenerator
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe

log = logging.getLogger(__name__)

META_SCAFFOLD_CTRL = "scaffold_ctrl_diff"
META_FEDNOVA_TAU = "fednova_tau"
META_TRAIN_SAMPLES = "train_samples"


class FlexibleRecipe(Recipe):
    def __init__(
        self,
        *,
        name: str,
        initial_model: Any,
        min_clients: int,
        num_rounds: int,
        train_script: str,
        train_args: str = "",
        aggregator: Optional[Aggregator] = None,
        shareable_generator: Optional[Any] = None,
        aggregator_data_kind: DataKind = DataKind.WEIGHTS,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
    ):
        from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob

        job = BaseFedJob(initial_model=initial_model, name=name, min_clients=min_clients)
        if aggregator is None:
            from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator

            aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=aggregator_data_kind)
        if shareable_generator is None:
            shareable_generator = FullModelShareableGenerator()

        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")
        aggregator_id = job.to_server(aggregator, id="aggregator")

        controller = ScatterAndGather(
            min_clients=min_clients,
            num_rounds=num_rounds,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id=job.comp_ids["persistor_id"] if initial_model is not None else "",
            shareable_generator_id=shareable_generator_id,
        )
        job.to_server(controller)

        executor = ScriptRunner(
            script=train_script,
            script_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
            framework=FrameworkType.PYTORCH,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
        )
        job.to_clients(executor)
        Recipe.__init__(self, job)


class ScaffoldAggregator(Aggregator):
    def __init__(self):
        super().__init__()
        self.weight_helper = WeightedAggregationHelper(weigh_by_local_iter=True)
        self.ctrl_helper = WeightedAggregationHelper(weigh_by_local_iter=True)

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        dxo = from_shareable(shareable)
        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        steps = float(dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
        if dxo.data_kind != DataKind.WEIGHTS:
            self.log_error(fl_ctx, f"SCAFFOLD expects WEIGHTS but got {dxo.data_kind}.")
            return False
        ctrl_diff = dxo.get_meta_prop(META_SCAFFOLD_CTRL)
        if not isinstance(ctrl_diff, dict):
            self.log_error(fl_ctx, "SCAFFOLD contribution missing control diff meta.")
            return False
        self.weight_helper.add(dxo.data, steps, contributor_name, contribution_round)
        self.ctrl_helper.add(ctrl_diff, steps, contributor_name, contribution_round)
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        result = DXO(
            data_kind=DataKind.COLLECTION,
            data={
                AppConstants.MODEL_WEIGHTS: DXO(data_kind=DataKind.WEIGHTS, data=self.weight_helper.get_result()),
                AlgorithmConstants.SCAFFOLD_CTRL_DIFF: DXO(
                    data_kind=DataKind.WEIGHT_DIFF, data=self.ctrl_helper.get_result()
                ),
            },
        )
        self.weight_helper.reset_stats()
        self.ctrl_helper.reset_stats()
        return result.to_shareable()


class ScaffoldShareableGenerator(FullModelShareableGenerator):
    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        shareable = super().learnable_to_shareable(model_learnable, fl_ctx)
        meta = model_learnable.get(ModelLearnableKey.META, {}) or {}
        ctrl = meta.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
        if ctrl is not None:
            dxo = from_shareable(shareable)
            dxo.set_meta_prop(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL, ctrl)
            shareable = dxo.to_shareable()
        return shareable

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        dxo = from_shareable(shareable)
        if dxo.data_kind != DataKind.COLLECTION:
            raise ValueError("SCAFFOLD shareable generator expects COLLECTION data.")
        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            self.system_panic("No global base model for SCAFFOLD update.", fl_ctx=fl_ctx)
            return ModelLearnable()

        model_dxo = dxo.data[AppConstants.MODEL_WEIGHTS]
        ctrl_dxo = dxo.data[AlgorithmConstants.SCAFFOLD_CTRL_DIFF]
        learnable = super().shareable_to_learnable(model_dxo.to_shareable(), fl_ctx)
        meta = dict(learnable.get(ModelLearnableKey.META, {}) or {})
        global_ctrl = meta.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
        if global_ctrl is None:
            global_ctrl = {k: np.zeros_like(v) for k, v in learnable[ModelLearnableKey.WEIGHTS].items()}
        for key, value in ctrl_dxo.data.items():
            global_ctrl[key] = global_ctrl[key] + value
        meta[AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL] = global_ctrl
        learnable[ModelLearnableKey.META] = meta
        return learnable


class FedNovaAggregator(Aggregator):
    def __init__(self):
        super().__init__()
        self.helper = WeightedAggregationHelper(weigh_by_local_iter=False)

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        dxo = from_shareable(shareable)
        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        contribution_round = shareable.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        if dxo.data_kind != DataKind.WEIGHT_DIFF:
            self.log_error(fl_ctx, f"FedNova expects WEIGHT_DIFF but got {dxo.data_kind}.")
            return False
        tau = float(dxo.get_meta_prop(META_FEDNOVA_TAU, dxo.get_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)))
        tau = max(tau, 1.0)
        normalized = {k: v / tau for k, v in dxo.data.items()}
        self.helper.add(normalized, 1.0, contributor_name, contribution_round)
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        result = DXO(data_kind=DataKind.WEIGHT_DIFF, data=self.helper.get_result()).to_shareable()
        self.helper.reset_stats()
        return result


class FedDynShareableGenerator(FullModelShareableGenerator):
    def __init__(self, alpha: float = 1e-3):
        super().__init__()
        self.alpha = float(alpha)
        self.server_state: Optional[Dict[str, np.ndarray]] = None

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        if not base_model:
            self.system_panic("No global base model for FedDyn.", fl_ctx=fl_ctx)
            return ModelLearnable()
        dxo = from_shareable(shareable)
        if dxo.data_kind != DataKind.WEIGHTS:
            raise ValueError("FedDyn expects aggregated WEIGHTS.")

        prev = base_model[ModelLearnableKey.WEIGHTS]
        avg_weights = dxo.data
        if self.server_state is None:
            self.server_state = {k: np.zeros_like(v) for k, v in prev.items()}

        new_weights: Dict[str, np.ndarray] = {}
        for key, avg_value in avg_weights.items():
            drift = avg_value - prev[key]
            self.server_state[key] = self.server_state[key] - self.alpha * drift
            new_weights[key] = avg_value - (self.server_state[key] / max(self.alpha, 1e-12))
        base_model[ModelLearnableKey.WEIGHTS] = new_weights
        meta = dict(base_model.get(ModelLearnableKey.META, {}) or {})
        meta["feddyn_server_state"] = self.server_state
        base_model[ModelLearnableKey.META] = meta
        return base_model


@dataclass
class MethodContext:
    method_name: str
    round_idx: int
    global_params: Dict[str, torch.Tensor]
    input_meta: Dict[str, Any]


class MethodHook:
    supported_tasks: Optional[Sequence[str]] = None

    def validate_task(self, task_type: str) -> None:
        if self.supported_tasks is not None and task_type not in self.supported_tasks:
            raise ValueError(f"Method '{self.method_name}' does not support task_type='{task_type}'.")

    @property
    def method_name(self) -> str:
        return "fedavg"

    def on_round_start(self, model: torch.nn.Module, ctx: MethodContext) -> None:
        return None

    def augment_loss(
        self,
        *,
        base_loss: torch.Tensor,
        model: torch.nn.Module,
        outputs: Any,
        batch: Any,
        ctx: MethodContext,
    ) -> torch.Tensor:
        return base_loss

    def on_step_end(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ctx: MethodContext,
    ) -> None:
        return None

    def on_round_end(self, model: torch.nn.Module, ctx: MethodContext) -> None:
        return None

    def export_round_meta(
        self,
        *,
        model: torch.nn.Module,
        ctx: MethodContext,
        executed_steps: int,
        train_samples: int,
    ) -> Dict[str, Any]:
        return {}


class NoOpHook(MethodHook):
    pass


class FedNovaHook(MethodHook):
    @property
    def method_name(self) -> str:
        return "fednova"

    def export_round_meta(
        self,
        *,
        model: torch.nn.Module,
        ctx: MethodContext,
        executed_steps: int,
        train_samples: int,
    ) -> Dict[str, Any]:
        return {
            META_FEDNOVA_TAU: float(max(executed_steps, 1)),
            META_TRAIN_SAMPLES: float(max(train_samples, 1)),
        }


class FedDynHook(MethodHook):
    def __init__(self, alpha: float = 1e-3):
        self.alpha = float(alpha)
        self.local_state: Optional[Dict[str, torch.Tensor]] = None
        self.global_params: Optional[Dict[str, torch.Tensor]] = None

    @property
    def method_name(self) -> str:
        return "feddyn"

    def on_round_start(self, model: torch.nn.Module, ctx: MethodContext) -> None:
        self.global_params = {k: v.detach().clone().to(next(model.parameters()).device) for k, v in ctx.global_params.items()}
        if self.local_state is None:
            self.local_state = {k: torch.zeros_like(v) for k, v in self.global_params.items()}

    def augment_loss(
        self,
        *,
        base_loss: torch.Tensor,
        model: torch.nn.Module,
        outputs: Any,
        batch: Any,
        ctx: MethodContext,
    ) -> torch.Tensor:
        if self.global_params is None or self.local_state is None:
            return base_loss
        penalty = base_loss.new_zeros(())
        for name, param in model.named_parameters():
            if name not in self.global_params:
                continue
            global_param = self.global_params[name].to(param.device)
            dual = self.local_state[name].to(param.device)
            penalty = penalty + ((dual - global_param) * param).sum()
            penalty = penalty + 0.5 * self.alpha * torch.sum((param - global_param) ** 2)
        return base_loss + self.alpha * penalty

    def on_round_end(self, model: torch.nn.Module, ctx: MethodContext) -> None:
        if self.global_params is None:
            return
        if self.local_state is None:
            self.local_state = {}
        for name, param in model.named_parameters():
            if name not in self.global_params:
                continue
            delta = param.detach().cpu() - self.global_params[name].detach().cpu()
            self.local_state[name] = self.local_state.get(name, torch.zeros_like(delta)) + delta


class FedDecorrHook(MethodHook):
    supported_tasks = ("classification",)

    def __init__(self, weight: float = 1e-3):
        self.weight = float(weight)

    @property
    def method_name(self) -> str:
        return "feddecorr"

    def augment_loss(
        self,
        *,
        base_loss: torch.Tensor,
        model: torch.nn.Module,
        outputs: Any,
        batch: Any,
        ctx: MethodContext,
    ) -> torch.Tensor:
        if not torch.is_tensor(outputs) or outputs.ndim != 2 or outputs.shape[0] <= 1:
            return base_loss
        centered = outputs - outputs.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(outputs.shape[0] - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return base_loss + self.weight * torch.mean(off_diag ** 2)


class FedLESAMHook(MethodHook):
    supported_tasks = ("classification",)

    def __init__(self, weight: float = 1e-3):
        self.weight = float(weight)

    @property
    def method_name(self) -> str:
        return "fedlesam"

    def augment_loss(
        self,
        *,
        base_loss: torch.Tensor,
        model: torch.nn.Module,
        outputs: Any,
        batch: Any,
        ctx: MethodContext,
    ) -> torch.Tensor:
        if not torch.is_tensor(outputs) or outputs.ndim != 2:
            return base_loss
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)), dim=1).mean()
        return base_loss + self.weight * entropy


class ScaffoldHook(MethodHook):
    def __init__(self):
        from nvflare.app_opt.pt.scaffold import PTScaffoldHelper

        self.helper = PTScaffoldHelper()
        self.initialized = False
        self.global_model_copy: Optional[torch.nn.Module] = None
        self.last_lr = 1e-3
        self._c_global: Optional[dict] = None
        self._c_local: Optional[dict] = None
        # Precomputed once per round for vectorised step correction (avoids per-step Python loop).
        self._correction_list: list = []
        self._params_for_correction: list = []

    @property
    def method_name(self) -> str:
        return "scaffold"

    def on_round_start(self, model: torch.nn.Module, ctx: MethodContext) -> None:
        if not self.initialized:
            self.helper.init(model)
            self.initialized = True
        global_ctrl = ctx.input_meta.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
        if global_ctrl is not None:
            # Load onto CPU — avoids c_global sitting on GPU for the entire local
            # training phase when load_global_controls receives GPU tensors.
            tensor_ctrl = {k: torch.as_tensor(v, dtype=model.state_dict()[k].dtype) for k, v in global_ctrl.items()}
            self.helper.load_global_controls(tensor_ctrl)
        # Keep c_global/c_local on CPU throughout local training.  They are only
        # needed on GPU in on_round_end for terms_update (moved there at that point).
        # Saves ~2 model-copies of GPU memory per client during all training steps.
        self.helper.c_global.cpu()
        self.helper.c_local.cpu()
        # Fetch params once per round — get_params() resets cnt so must not be
        # called again until the next round starts.
        self._c_global, self._c_local = self.helper.get_params()  # CPU dicts
        # Precompute (c_global - c_local) on CPU, then move to GPU once.
        # correction_list stays on GPU for the duration of training (needed every step),
        # but c_global/c_local modules and dicts remain on CPU.
        device = next(model.parameters()).device
        self._correction_list = []
        self._params_for_correction = []
        for name, param in model.named_parameters():
            if name in self._c_global:
                diff = (self._c_global[name] - self._c_local[name]).detach().to(device)
                self._correction_list.append(diff)
                self._params_for_correction.append(param)
        # Zero grads before deepcopy so gradient buffers are not copied onto GPU,
        # halving the transient GPU spike during the copy.
        model.zero_grad()
        self.global_model_copy = deepcopy(model).cpu()

    def on_step_end(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ctx: MethodContext,
    ) -> None:
        current_lr = float(optimizer.param_groups[0]["lr"])
        self.last_lr = current_lr
        # Vectorised SCAFFOLD correction — replaces PTScaffoldHelper.model_update's
        # per-parameter Python loop with a single fused CUDA call.  The loop launches
        # one kernel per parameter tensor (~200 for ConvNeXt-Small) and is the dominant
        # cost causing the ~22× slowdown vs FedAvg on architectures with many layers.
        if self._params_for_correction:
            param_data = [p.data for p in self._params_for_correction]
            torch._foreach_sub_(param_data, self._correction_list, alpha=current_lr)
        self.helper.cnt += 1

    def on_round_end(self, model: torch.nn.Module, ctx: MethodContext) -> None:
        if self.global_model_copy is None:
            return
        if self.helper.cnt == 0:
            log.warning("SCAFFOLD: cnt=0 at round end (no training steps executed), skipping terms_update.")
            return
        lr = max(self.last_lr, 1e-6)
        device = next(model.parameters()).device
        self.global_model_copy = self.global_model_copy.to(device)
        # _c_global/_c_local are state_dict snapshots (dicts of tensors) — move them explicitly
        self._c_global = {k: v.to(device) for k, v in self._c_global.items()}
        self._c_local = {k: v.to(device) for k, v in self._c_local.items()}
        # Ensure c_local/c_global modules are also on device (state_dict in terms_update pulls from these)
        self.helper.c_global.to(device)
        self.helper.c_local.to(device)
        self.helper.terms_update(model, lr, self._c_global, self._c_local, self.global_model_copy)
        # Return to CPU immediately — in simulation all n clients' copies would otherwise
        # accumulate on GPU until the next round's on_round_start replaces the reference.
        self.global_model_copy.cpu()
        self.helper.c_global.cpu()
        self.helper.c_local.cpu()

    def export_round_meta(
        self,
        *,
        model: torch.nn.Module,
        ctx: MethodContext,
        executed_steps: int,
        train_samples: int,
    ) -> Dict[str, Any]:
        try:
            ctrl_diff = self.helper.get_delta_controls()
        except ValueError as exc:
            log.error(
                "SCAFFOLD: get_delta_controls() failed at round %d: %s — "
                "control diff will be empty; global control variate update skipped for this round.",
                ctx.round_idx,
                exc,
            )
            ctrl_diff = {}
        # Filter to only float-typed model parameters — integer buffers like
        # num_batches_tracked produce float64 deltas that break server-side
        # numpy in-place aggregation into int64 _global_ctrl_weights.
        model_state = model.state_dict()
        ctrl_diff = {k: v for k, v in ctrl_diff.items()
                     if k in model_state and model_state[k].is_floating_point()}
        return {AlgorithmConstants.SCAFFOLD_CTRL_DIFF: ctrl_diff}


def method_hook_from_config(config: Dict[str, Any], task_type: str) -> MethodHook:
    method_name = str(config.get("fed", {}).get("method", "fedavg")).lower()
    method_args = dict(config.get("fed", {}).get("args", {}) or {})
    if method_name == "fedavg":
        hook: MethodHook = NoOpHook()
    elif method_name == "fedadam":
        hook = NoOpHook()
    elif method_name == "scaffold":
        hook = ScaffoldHook()
    elif method_name == "fednova":
        hook = FedNovaHook()
    elif method_name == "feddyn":
        hook = FedDynHook(alpha=float(method_args.get("alpha", 1e-3)))
    elif method_name == "feddecorr":
        hook = FedDecorrHook(weight=float(method_args.get("decorrelation_weight", 1e-3)))
    elif method_name == "fedlesam":
        hook = FedLESAMHook(weight=float(method_args.get("entropy_weight", 1e-3)))
    else:
        raise ValueError(f"Unsupported fed.method '{method_name}'.")
    hook.validate_task(task_type)
    return hook


def validate_method_task_compatibility(config: Dict[str, Any], task_type: str) -> None:
    method_hook_from_config(config, task_type)


def build_recipe(method_name: str, *, name: str, initial_model: Any, min_clients: int, num_rounds: int, train_script: str, train_args: str, method_args: Dict[str, Any]) -> Recipe:
    normalized = method_name.lower()
    if normalized == "fedavg":
        from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

        # NVFlare changed this kwarg from `initial_model` to `model` in newer releases.
        # Keep runtime compatibility with both signatures.
        fedavg_kwargs = {
            "name": name,
            "min_clients": min_clients,
            "num_rounds": num_rounds,
            "train_script": train_script,
            "train_args": train_args,
        }
        signature = inspect.signature(FedAvgRecipe.__init__)
        if "model" in signature.parameters:
            fedavg_kwargs["model"] = initial_model
        else:
            fedavg_kwargs["initial_model"] = initial_model
        return FedAvgRecipe(**fedavg_kwargs)
    if normalized == "fedadam":
        shareable_generator = PTFedOptModelShareableGenerator(
            optimizer_args={
                "path": "torch.optim.Adam",
                "args": {
                    "lr": float(method_args.get("server_lr", 1e-3)),
                    "betas": tuple(method_args.get("betas", (0.9, 0.999))),
                    "eps": float(method_args.get("eps", 1e-8)),
                },
            }
        )
        return FlexibleRecipe(
            name=name,
            initial_model=initial_model,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            shareable_generator=shareable_generator,
            aggregator_data_kind=DataKind.WEIGHT_DIFF,
            params_transfer_type=TransferType.DIFF,
        )
    if normalized == "scaffold":
        from nvflare.app_opt.pt.recipes.scaffold import ScaffoldRecipe

        scaffold_kwargs = {
            "name": name,
            "min_clients": min_clients,
            "num_rounds": num_rounds,
            "train_script": train_script,
            "train_args": train_args,
        }
        signature = inspect.signature(ScaffoldRecipe.__init__)
        if "model" in signature.parameters:
            scaffold_kwargs["model"] = initial_model
        else:
            scaffold_kwargs["initial_model"] = initial_model
        return ScaffoldRecipe(**scaffold_kwargs)
    if normalized == "fednova":
        return FlexibleRecipe(
            name=name,
            initial_model=initial_model,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            aggregator=FedNovaAggregator(),
            shareable_generator=FullModelShareableGenerator(),
            aggregator_data_kind=DataKind.WEIGHT_DIFF,
            params_transfer_type=TransferType.DIFF,
        )
    if normalized == "feddyn":
        return FlexibleRecipe(
            name=name,
            initial_model=initial_model,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            shareable_generator=FedDynShareableGenerator(alpha=float(method_args.get("alpha", 1e-3))),
            aggregator_data_kind=DataKind.WEIGHTS,
            params_transfer_type=TransferType.FULL,
        )
    if normalized in {"feddecorr", "fedlesam"}:
        return FlexibleRecipe(
            name=name,
            initial_model=initial_model,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            aggregator_data_kind=DataKind.WEIGHTS,
            params_transfer_type=TransferType.FULL,
        )
    raise ValueError(f"Unsupported fed.method '{method_name}'.")
