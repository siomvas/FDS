"""Narrow, version-pinned Fluke compatibility patches used by the adapter."""

from __future__ import annotations

import math
import statistics
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence


_PATCHES_BY_TEMPLATE: dict[str, tuple[str, ...]] = {
    "feddyn": ("feddyn_dual_state_sign_and_scale_v1",),
    "fednova": ("fednova_delta_aggregation_v1",),
    "scaffold": ("scaffold_option_ii_median_steps_weighted_control_v3",),
}


def compatibility_patches(template_key: str) -> list[str]:
    """Return artifact-visible compatibility patches for one catalog method."""

    return list(_PATCHES_BY_TEMPLATE.get(template_key, ()))


def sgd_momentum_normalization(steps: int, momentum: float) -> float:
    """Return FedNova's effective SGD step count for ``steps`` local updates."""

    if steps < 1:
        raise ValueError("SGD momentum normalization requires at least one local optimizer step.")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("SGD momentum must be in [0, 1) for momentum normalization.")
    if momentum == 0.0:
        return float(steps)
    return (steps - momentum * (1.0 - momentum**steps) / (1.0 - momentum)) / (
        1.0 - momentum
    )


def scaffold_momentum_normalization(steps: int, momentum: float) -> float:
    """Return the Option-II SCAFFOLD denominator's effective step count."""

    return sgd_momentum_normalization(steps, momentum)


def scaffold_steps_per_epoch(batches_per_epoch: Sequence[int]) -> int:
    """Return the local SCAFFOLD steps per configured epoch: the median client epoch length.

    ``batches_per_epoch`` holds the number of batches in one epoch of every client. With an
    even number of clients the median is the mean of the two central values, rounded to the
    nearest integer with halves rounded up.
    """

    values = [int(value) for value in batches_per_epoch]
    if not values or min(values) < 1:
        raise ValueError("Every SCAFFOLD client needs at least one batch per epoch.")
    return max(1, math.floor(statistics.median(values) + 0.5))


def scaffold_control_weights(
    n_examples: Mapping[int, int], effective_steps: Mapping[int, float]
) -> dict[int, float]:
    """Return the weight of each client's control update on the server.

    The global model moves by ``sum_i w_i (y_i - x)`` with ``w_i`` proportional to the client's
    sample count, and client ``i`` applies the correction ``c - c_i`` over ``K_i`` effective
    steps. The corrections cancel in the aggregate only when ``c = sum_i a_i c_i`` with ``a_i``
    proportional to ``n_i * K_i``. Fluke averages the controls uniformly, which matches only
    when every client has the same size and runs the same number of steps.
    """

    raw = {index: float(n_examples[index]) * float(effective_steps[index]) for index in n_examples}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("SCAFFOLD control weights need positive sample counts and steps.")
    return {index: value / total for index, value in raw.items()}


def configure_scaffold_local_steps(clients: Sequence[Any]) -> int:
    """Give every SCAFFOLD client the same local steps per epoch, the median epoch length."""

    steps = scaffold_steps_per_epoch([len(client.train_set) for client in clients])
    for client in clients:
        client._fds_scaffold_steps_per_epoch = steps
    return steps


def executed_local_steps(train_set: Any, epochs: int) -> int:
    """Count executable batches for Fluke and the adapter's lazy loader contract."""

    if epochs < 1:
        raise ValueError("Local epochs must be >= 1.")
    batches = int(len(train_set))
    # LazyFastDataLoader deliberately skips a final singleton batch. Its public
    # ``n_batches`` remains ceil(size / batch_size), so account for that omitted step.
    if (
        bool(getattr(train_set, "skip_singleton", False))
        and int(getattr(train_set, "size", 0)) % int(getattr(train_set, "batch_size", 1)) == 1
    ):
        batches -= 1
    steps = epochs * batches
    if steps < 1:
        raise RuntimeError("FedNova requires at least one executed local optimizer step.")
    return steps


def corrected_feddyn_dual(
    previous: Any, global_params: Any, local_params: Any
) -> Any:
    """Apply FedDyn's unscaled, positive dual accumulation convention."""

    return previous + (local_params - global_params)


def fednova_delta_weights(
    client_weights: Sequence[float], normalizers: Sequence[float]
) -> tuple[float, list[float]]:
    """Return FedNova's aggregate coefficient and anchored-delta weights."""

    if len(client_weights) != len(normalizers) or not client_weights:
        raise ValueError("FedNova requires one positive normalization coefficient per client.")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in client_weights):
        raise ValueError("FedNova received an invalid client aggregation weight.")
    if any(not math.isfinite(value) or value <= 0.0 for value in normalizers):
        raise ValueError("FedNova received a non-positive normalization coefficient.")
    coefficient = sum(weight * normalizer for weight, normalizer in zip(client_weights, normalizers))
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("FedNova client aggregation weights must have a positive finite sum.")
    return coefficient, [
        weight * coefficient / normalizer
        for weight, normalizer in zip(client_weights, normalizers)
    ]


def zero_sgd_momentum_buffers(optimizer: Any) -> int:
    """Reset persistent SGD momentum so each SCAFFOLD local solve starts at ``m_0 = 0``."""

    cleared = 0
    for state in optimizer.state.values():
        buffer = state.get("momentum_buffer")
        if buffer is not None:
            buffer.zero_()
            cleared += 1
    return cleared


def _fednova_state_tensors(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return trainable parameters and aggregatable floating buffers by state name."""

    parameters = dict(model.named_parameters())
    buffers = {
        name: buffer
        for name, buffer in model.named_buffers()
        if "num_batches_tracked" not in name and (buffer.is_floating_point() or buffer.is_complex())
    }
    return parameters, buffers


def _fednova_batch_tracker_max(models: Sequence[Any]) -> dict[str, int]:
    """Preserve Fluke's max ``num_batches_tracked`` behavior for BatchNorm."""

    maximums: dict[str, int] = {}
    for model in models:
        for name, buffer in model.named_buffers():
            if "num_batches_tracked" in name:
                maximums[name] = max(maximums.get(name, 0), int(buffer.item()))
    return maximums


def apply_fednova_delta_update(
    model: Any,
    client_models: Sequence[Any],
    client_weights: Sequence[float],
    normalizers: Sequence[float],
    *,
    eta: float,
) -> dict[str, float]:
    """Apply FedNova as an anchored delta update and return its scalar diagnostics."""

    import torch

    if not math.isfinite(eta):
        raise ValueError("FedNova server learning rate must be finite.")
    if len(client_models) != len(client_weights):
        raise ValueError("FedNova requires one model and weight per participating client.")
    coefficient, delta_weights = fednova_delta_weights(client_weights, normalizers)
    target_parameters, target_buffers = _fednova_state_tensors(model)
    target = {**target_parameters, **target_buffers}
    base = {name: tensor.detach().clone() for name, tensor in target.items()}
    delta = {name: torch.zeros_like(tensor) for name, tensor in base.items()}
    for client_model, weight in zip(client_models, delta_weights):
        parameters, buffers = _fednova_state_tensors(client_model)
        source = {**parameters, **buffers}
        if source.keys() != base.keys():
            raise ValueError("FedNova client model state does not match the server model.")
        for name, value in source.items():
            delta[name].add_(value.detach().to(base[name].device) - base[name], alpha=weight)
    for name, tensor in target.items():
        tensor.data.copy_(base[name] + eta * delta[name])

    batch_tracker_maximums = _fednova_batch_tracker_max(client_models)
    for name, buffer in model.named_buffers():
        if name in batch_tracker_maximums:
            buffer.data.fill_(batch_tracker_maximums[name])
    return {
        "fednova_normalizer_min": min(normalizers),
        "fednova_normalizer_max": max(normalizers),
        "fednova_coefficient": coefficient,
        "fednova_delta_weight_sum": sum(delta_weights),
    }


def install_feddyn_dual_correction() -> None:
    """Correct the pinned Fluke FedDyn dual update without copying its fit loop."""

    from fluke.algorithms.feddyn import FedDynClient, FedDynServer, get_all_params_of
    from fluke.comm import Message

    if getattr(FedDynClient, "_fds_dual_state_correction_installed", False):
        return

    native_fit = FedDynClient.fit
    native_send_model = FedDynClient.send_model
    native_aggregate = FedDynServer.aggregate

    def fit(self: Any, override_local_epochs: int = 0) -> float:
        if self.model is None or self.prev_grads is None:
            raise RuntimeError("FedDyn must receive a model and initialize dual state before fit.")
        global_params = get_all_params_of(self.model).detach().clone()
        previous = self.prev_grads.detach().clone()
        before_norm = float(previous.norm().item())
        loss = native_fit(self, override_local_epochs)
        local_params = get_all_params_of(self.model).detach().to(previous.device)
        corrected = corrected_feddyn_dual(previous, global_params.to(previous.device), local_params)
        self.prev_grads = corrected
        self._fds_feddyn_dual_norms = {
            "before": before_norm,
            "after": float(corrected.norm().item()),
        }
        return loss

    def send_model(self: Any) -> None:
        native_send_model(self)
        diagnostics = getattr(self, "_fds_feddyn_dual_norms", None)
        if diagnostics is not None:
            self.channel.send(
                Message(dict(diagnostics), "feddyn_dual_norms", self.index, inmemory=True),
                "server",
            )

    def aggregate(self: Any, eligible: Sequence[Any], client_models: Iterable[Any]) -> None:
        native_aggregate(self, eligible, client_models)
        diagnostics = [
            self.channel.receive("server", client.index, msg_type="feddyn_dual_norms").payload
            for client in eligible
        ]
        before = [float(item["before"]) for item in diagnostics]
        after = [float(item["after"]) for item in diagnostics]
        self._fds_last_feddyn_dual_norms = {
            "feddyn_dual_norm_before_mean": sum(before) / len(before),
            "feddyn_dual_norm_before_max": max(before),
            "feddyn_dual_norm_after_mean": sum(after) / len(after),
            "feddyn_dual_norm_after_max": max(after),
        }

    FedDynClient.fit = fit
    FedDynClient.send_model = send_model
    FedDynServer.aggregate = aggregate
    FedDynClient._fds_dual_state_correction_installed = True


def install_fednova_delta_aggregation() -> None:
    """Correct Fluke FedNova's effective-step accounting and anchored server update."""

    import torch
    from fluke.algorithms.fednova import FedNovaClient, FedNovaServer
    from fluke.client import Client
    from fluke.comm import Message

    if getattr(FedNovaClient, "_fds_delta_aggregation_installed", False):
        return

    def fit(self: Any, override_local_epochs: int = 0) -> float:
        epochs = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )
        loss = Client.fit(self, override_local_epochs)
        self.tau = executed_local_steps(self.train_set, int(epochs))
        momentum = float(self._get_momentum())
        self.a = sgd_momentum_normalization(self.tau, momentum)
        if not math.isfinite(self.a) or self.a <= 0.0:
            raise RuntimeError("FedNova produced a non-positive normalization coefficient.")
        self.channel.send(Message(self.a, "local_a", self.index, inmemory=True), "server")
        return loss

    @torch.no_grad()
    def aggregate(self: Any, eligible: Sequence[Any], client_models: Iterable[Any]) -> None:
        models = list(client_models)
        if len(models) != len(eligible):
            raise RuntimeError("FedNova received a model count that does not match eligible clients.")
        client_weights = [float(weight) for weight in self._get_client_weights(eligible)]
        normalizers = [
            float(self.channel.receive("server", client.index, "local_a").payload)
            for client in eligible
        ]
        eta = float(self.hyper_params.lr)

        try:
            self._fds_last_fednova_normalization = apply_fednova_delta_update(
                self.model,
                models,
                client_weights,
                normalizers,
                eta=eta,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    FedNovaClient.fit = fit
    FedNovaServer.aggregate = aggregate
    FedNovaClient._fds_delta_aggregation_installed = True


def install_scaffold_momentum_normalization() -> None:
    """Patch Fluke SCAFFOLD Option II for momentum, equal local work and a consistent control.

    Against Fluke 0.8.1 this patch makes three changes.

    * Each local solve starts from zero SGD momentum and divides the model change by the
      effective number of momentum steps.
    * Every client runs the same number of optimizer steps per round: the configured local
      epochs times the median number of batches per epoch across clients, set by
      ``configure_scaffold_local_steps``. A client cycles through its own data as often as
      needed, so small clients make several passes and large clients part of one.
    * The server weights each client's control update by its sample count times its effective
      steps (``scaffold_control_weights``), so that the corrections ``c - c_i`` cancel in the
      sample-weighted model aggregate.

    Fluke runs one epoch per client and averages the controls uniformly, so on partitions
    with clients of very different sizes the corrections no longer cancel.
    """

    import torch
    from fluke.algorithms.scaffold import SCAFFOLDClient, SCAFFOLDServer
    from fluke.utils import clear_cuda_cache
    from fluke.utils.model import state_dict_zero_like, unwrap

    if getattr(SCAFFOLDClient, "_fds_momentum_normalization_installed", False):
        return

    def fit(self: Any, override_local_epochs: int = 0) -> float:
        epochs = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )
        steps_per_epoch = getattr(self, "_fds_scaffold_steps_per_epoch", None)
        if steps_per_epoch is None:
            raise RuntimeError(
                "configure_scaffold_local_steps() must run before SCAFFOLD clients train."
            )
        if len(self.train_set) < 1:
            raise RuntimeError("A SCAFFOLD client has an empty training set.")
        target_steps = int(epochs) * int(steps_per_epoch)
        self.model.to(self.device)
        self.model.train()

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)
        # Option II's momentum normalization assumes each local solve begins at m_0 = 0.
        zero_sgd_momentum_buffers(self.optimizer)

        steps = 0
        learning_rate: float | None = None
        running_loss = 0.0
        while steps < target_steps:
            for features, labels in self.train_set:
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                if learning_rate is None:
                    learning_rate = current_lr
                elif current_lr != learning_rate:
                    raise RuntimeError(
                        "The SCAFFOLD compatibility patch requires a constant local "
                        "learning rate."
                    )
                features, labels = features.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.hyper_params.loss_fn(logits, labels)
                loss.backward()
                for name, parameter in unwrap(self.model).named_parameters():
                    parameter.grad.data = parameter.grad.data + (
                        self.server_control[name].to(self.device)
                        - self.control[name].to(self.device)
                    )
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
                steps += 1
                if steps % steps_per_epoch == 0:
                    self.scheduler.step()
                if steps >= target_steps:
                    break

        self.model.cpu()
        with torch.no_grad():
            momentum = float(self.optimizer.param_groups[0].get("momentum", 0.0))
            effective_steps = scaffold_momentum_normalization(steps, momentum)
            self._fds_scaffold_effective_steps = effective_steps
            denominator = learning_rate * effective_steps
            c_plus = state_dict_zero_like(self.control)
            c_delta = state_dict_zero_like(self.control)
            model_params = unwrap(self.model).state_dict()
            for key in model_params:
                c_plus[key] = (
                    self.control[key]
                    - self.server_control[key]
                    + (self.server_model[key] - model_params[key]) / denominator
                )
            for key in model_params:
                c_delta[key] = c_plus[key] - self.control[key]
            self.control = deepcopy(c_plus)
            self.delta_control = c_delta

        running_loss /= steps
        clear_cuda_cache()
        return running_loss

    @torch.no_grad()
    def aggregate(self: Any, eligible: Sequence[Any], client_models: Any) -> None:
        recorded = {
            client.index: getattr(client, "_fds_scaffold_effective_steps", None)
            for client in self.clients
        }
        known = [value for value in recorded.values() if value is not None]
        if not known:
            raise RuntimeError("No SCAFFOLD client reported its effective local steps.")
        fallback = statistics.median(known)
        weights = scaffold_control_weights(
            {client.index: client.n_examples for client in self.clients},
            {index: fallback if value is None else value for index, value in recorded.items()},
        )
        for client in eligible:
            delta = self.channel.receive("server", client.index, "control").payload
            weight = weights[client.index]
            for key in self.control:
                self.control[key] = self.control[key] + weight * delta[key].to("cpu")
        return super(SCAFFOLDServer, self).aggregate(eligible, client_models)

    SCAFFOLDClient.fit = fit
    SCAFFOLDServer.aggregate = aggregate
    SCAFFOLDClient._fds_momentum_normalization_installed = True


def install_fluke_compatibility(template_key: str) -> None:
    """Install only the version-pinned correction required by one algorithm template."""

    if template_key == "feddyn":
        install_feddyn_dual_correction()
    elif template_key == "fednova":
        install_fednova_delta_aggregation()
    elif template_key == "scaffold":
        install_scaffold_momentum_normalization()
