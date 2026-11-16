"""Method contracts for the Fluke 0.8.1 adapter.

The catalog deliberately contains configuration contracts only.  Algorithm behaviour
continues to come from Fluke itself, which keeps this downstream evaluation surface
small and avoids maintaining duplicate method implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal, Mapping

Protocol = Literal["centralized", "decentralized", "parallel"]
ModelComponent = Literal["full", "encoder", "head"]


@dataclass(frozen=True)
class AlgorithmTemplate:
    """Default contract for one public Fluke method entrypoint."""

    key: str
    class_path: str
    protocol: Protocol = "centralized"
    model_component: ModelComponent = "full"
    client_model_component: ModelComponent | None = None
    transform_profile: Literal["imagenet", "unit_interval"] = "imagenet"
    normalization: Literal["group", "batch", "none"] = "group"
    result_scope: Literal[
        "global_model", "personalized_shared_global_val", "decentralized_shared_global_val"
    ] = "global_model"
    client_defaults: Mapping[str, Any] = field(default_factory=dict)
    server_defaults: Mapping[str, Any] = field(default_factory=dict)


def _template(
    key: str,
    class_path: str,
    *,
    protocol: Protocol = "centralized",
    model_component: ModelComponent = "full",
    client_model_component: ModelComponent | None = None,
    transform_profile: Literal["imagenet", "unit_interval"] = "imagenet",
    normalization: Literal["group", "batch", "none"] = "group",
    result_scope: Literal[
        "global_model", "personalized_shared_global_val", "decentralized_shared_global_val"
    ] = "global_model",
    client: Mapping[str, Any] | None = None,
    server: Mapping[str, Any] | None = None,
) -> AlgorithmTemplate:
    return AlgorithmTemplate(
        key=key,
        class_path=class_path,
        protocol=protocol,
        model_component=model_component,
        client_model_component=client_model_component,
        transform_profile=transform_profile,
        normalization=normalization,
        result_scope=result_scope,
        client_defaults=dict(client or {}),
        server_defaults=dict(server or {}),
    )


# Public Fluke method entrypoints. FedOpt variants intentionally retain
# individual keys while sharing the same Fluke class path.
FLUKE_ALGORITHM_CATALOG: dict[str, AlgorithmTemplate] = {
    "fedavg": _template("fedavg", "fluke.algorithms.fedavg.FedAVG"),
    "fedavgm": _template(
        "fedavgm", "fluke.algorithms.fedavgm.FedAVGM", server={"momentum": 0.9}
    ),
    "fedopt_adam": _template(
        "fedopt_adam",
        "fluke.algorithms.fedopt.FedOpt",
        server={
            "mode": "adam",
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "tau": 1e-4,
            "bias_correction": True,
        },
    ),
    "fedopt_yogi": _template(
        "fedopt_yogi",
        "fluke.algorithms.fedopt.FedOpt",
        server={"mode": "yogi", "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "tau": 1e-4},
    ),
    "fedopt_adagrad": _template(
        "fedopt_adagrad",
        "fluke.algorithms.fedopt.FedOpt",
        server={"mode": "adagrad", "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "tau": 1e-4},
    ),
    "fedprox": _template(
        "fedprox", "fluke.algorithms.fedprox.FedProx", client={"mu": 0.01}
    ),
    "fednova": _template(
        "fednova", "fluke.algorithms.fednova.FedNova", server={"lr": 1.0}
    ),
    "scaffold": _template(
        # Fluke applies the server rate separately in ``aggregate_models``. Keep
        # this multiplier at one so the normalized client weights remain normalized.
        "scaffold", "fluke.algorithms.scaffold.SCAFFOLD", server={"global_step": 1.0}
    ),
    "feddyn": _template(
        "feddyn",
        "fluke.algorithms.feddyn.FedDyn",
        client={"alpha": 0.01},
        server={"alpha": 0.01},
    ),
    "fedsam": _template(
        "fedsam", "fluke.algorithms.fedsam.FedSAM", client={"rho": 0.05}
    ),
    "fedsgd": _template("fedsgd", "fluke.algorithms.fedsgd.FedSGD"),
    "fedexp": _template(
        "fedexp", "fluke.algorithms.fedexp.FedExP", server={"lr": 1.0}
    ),
    "fedala": _template(
        "fedala",
        "fluke.algorithms.fedala.FedALA",
        client={
            "ala_sample_size": 0.8,
            "eta": 1.0,
            "conergence_threshold": 0.001,
            "loss_window_size": 10,
        },
    ),
    "fedaws": _template(
        "fedaws",
        "fluke.algorithms.fedaws.FedAwS",
        server={
            "aws_lr": 0.1,
            "aws_steps": 100,
            "margin": 0.5,
            "last_layer_name": "_head.classifier",
        },
    ),
    "fedbn": _template(
        "fedbn", "fluke.algorithms.fedbn.FedBN", normalization="batch"
    ),
    "ccvr": _template(
        "ccvr",
        "fluke.algorithms.ccvr.CCVR",
        client_model_component="full",
        server={"lr": 0.1, "batch_size": 64, "sample_per_class": 100},
    ),
    "fedlc": _template(
        "fedlc", "fluke.algorithms.fedlc.FedLC", client={"tau": 0.1}
    ),
    "fedld": _template(
        "fedld", "fluke.algorithms.fedld.FedLD", client={"margin_lam": 0.2}, server={"lr": 1.0, "k_proportion": 0.8}
    ),
    "fedrs": _template(
        "fedrs", "fluke.algorithms.fedrs.FedRS", client={"alpha": 0.5, "count_as_missing": 2}
    ),
    "fednh": _template(
        "fednh",
        "fluke.algorithms.fednh.FedNH",
        model_component="encoder",
        client_model_component="encoder",
        result_scope="personalized_shared_global_val",
        client={"n_protos": 200, "fine_tuning_epochs": 1},
        server={"n_protos": 200, "rho": 0.1},
    ),
    "fedhp": _template(
        "fedhp",
        "fluke.algorithms.fedhp.FedHP",
        model_component="encoder",
        client_model_component="encoder",
        result_scope="personalized_shared_global_val",
        client={"n_protos": 200, "lam": 0.1, "fine_tuning_epochs": 1},
        server={"n_protos": 200, "embedding_size": 128},
    ),
    "fedproto": _template(
        "fedproto",
        "fluke.algorithms.fedproto.FedProto",
        client_model_component="full",
        result_scope="personalized_shared_global_val",
        client={"n_protos": 200, "lam": 1.0},
        server={"n_protos": 200},
    ),
    "fedrod": _template(
        "fedrod",
        "fluke.algorithms.fedrod.FedROD",
        result_scope="personalized_shared_global_val",
    ),
    "fedbabu": _template(
        "fedbabu",
        "fluke.algorithms.fedbabu.FedBABU",
        model_component="encoder",
        client_model_component="full",
        result_scope="personalized_shared_global_val",
        client={"mode": "head", "fine_tuning_epochs": 1},
    ),
    "fedper": _template(
        "fedper",
        "fluke.algorithms.fedper.FedPer",
        model_component="encoder",
        client_model_component="full",
        result_scope="personalized_shared_global_val",
    ),
    "fedrep": _template(
        "fedrep",
        "fluke.algorithms.fedrep.FedRep",
        model_component="encoder",
        client_model_component="full",
        result_scope="personalized_shared_global_val",
        client={"tau": 3},
    ),
    "lg_fedavg": _template(
        "lg_fedavg",
        "fluke.algorithms.lg_fedavg.LGFedAVG",
        model_component="head",
        client_model_component="full",
        result_scope="personalized_shared_global_val",
    ),
    "apfl": _template(
        "apfl",
        "fluke.algorithms.apfl.APFL",
        client_model_component="full",
        result_scope="personalized_shared_global_val",
        client={"lam": 0.25},
        server={"tau": 3},
    ),
    "ditto": _template(
        "ditto",
        "fluke.algorithms.ditto.Ditto",
        client_model_component="full",
        result_scope="personalized_shared_global_val",
        client={"tau": 3, "lam": 0.1},
    ),
    "fedamp": _template(
        "fedamp",
        "fluke.algorithms.fedamp.FedAMP",
        client_model_component="full",
        result_scope="personalized_shared_global_val",
        client={"lam": 0.2},
        server={"sigma": 0.1, "alpha": 0.1},
    ),
    "per_fedavg": _template(
        "per_fedavg",
        "fluke.algorithms.per_fedavg.PerFedAVG",
        result_scope="personalized_shared_global_val",
        client={"mode": "HF", "beta": 0.01},
    ),
    "pfedme": _template(
        "pfedme",
        "fluke.algorithms.pfedme.PFedMe",
        result_scope="personalized_shared_global_val",
        client={"k": 5},
        server={"beta": 0.5},
    ),
    "superfed": _template(
        "superfed",
        "fluke.algorithms.superfed.SuPerFed",
        client_model_component="full",
        normalization="none",
        result_scope="personalized_shared_global_val",
        client={"mode": "mm", "start_mix": 10, "mu": 0.1, "nu": 0.1},
    ),
    "fat": _template(
        "fat",
        "fluke.algorithms.fat.FAT",
        transform_profile="unit_interval",
        client={"eps": 0.1, "alpha": 2.0 / 255.0, "adv_iters": 10, "k_prop": 1.0},
    ),
    "gear": _template(
        "gear",
        "fluke.algorithms.gear.GEAR",
        transform_profile="unit_interval",
        client={"eps": 0.1, "delta": 0.01, "alpha": 2.0 / 255.0, "adv_iters": 10},
    ),
    "kafe": _template(
        "kafe", "fluke.algorithms.kafe.Kafe", server={"bandwidth": 1.0}
    ),
    "dpfedavg": _template(
        "dpfedavg",
        "fluke.algorithms.dpfedavg.DPFedAVG",
        client={"noise_mul": 1.1, "max_grad_norm": 1.0},
        server={"lr": 1.0},
    ),
    "moon": _template(
        "moon",
        "fluke.algorithms.moon.MOON",
        client={"mu": 1.0, "tau": 0.5},
    ),
    "gossip_dfl": _template(
        "gossip_dfl",
        "fluke.algorithms.decentralized.GossipDFL",
        protocol="decentralized",
        result_scope="decentralized_shared_global_val",
        client={"policy": "random", "activation_rate": 1.0},
    ),
    "parallel": _template(
        "parallel", "fluke.distr.ParallelAlgorithm", protocol="parallel"
    ),
}


# The class-level list is intentionally separate from keyed variants such as FedOpt.
EXPECTED_PUBLIC_CLASS_PATHS = frozenset(
    template.class_path for template in FLUKE_ALGORITHM_CATALOG.values()
)


def _import_class(class_path: str) -> type:
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Algorithm class_path must be a fully qualified class name: {class_path!r}")
    module = import_module(module_name)
    value = getattr(module, class_name)
    if not isinstance(value, type):
        raise TypeError(f"Algorithm target is not a class: {class_path}")
    return value


def supported_protocol_for_class(class_path: str) -> Protocol:
    """Return a supported Fluke protocol or raise a direct configuration error."""

    try:
        from fluke.algorithms import CentralizedFL
        from fluke.algorithms.decentralized import DecentralizedFL
        from fluke.distr import ParallelAlgorithm
    except ImportError as exc:  # pragma: no cover - exercised by CLI dependency guard
        raise RuntimeError("Fluke support requires `pip install -e '.[fluke]'`.") from exc

    algorithm_class = _import_class(class_path)
    if issubclass(algorithm_class, ParallelAlgorithm):
        return "parallel"
    if issubclass(algorithm_class, DecentralizedFL):
        return "decentralized"
    if issubclass(algorithm_class, CentralizedFL):
        return "centralized"
    raise TypeError(
        f"Unsupported Fluke algorithm class {class_path}. Expected a CentralizedFL, "
        "DecentralizedFL, or ParallelAlgorithm subclass."
    )


def resolve_algorithm_template(spec: str | Mapping[str, Any]) -> AlgorithmTemplate:
    """Resolve a catalog key or an arbitrary supported Fluke FQCN.

    Direct FQCNs intentionally receive only the generic FedAvg-like contract.  Their
    preflight construction then reports any required method-specific arguments instead
    of silently guessing them.
    """

    if isinstance(spec, str):
        if spec in FLUKE_ALGORITHM_CATALOG:
            return FLUKE_ALGORITHM_CATALOG[spec]
        class_path = spec
    else:
        class_path = str(spec.get("class_path") or spec.get("name") or "")
        key = str(spec.get("key") or "")
        if key and key in FLUKE_ALGORITHM_CATALOG:
            template = FLUKE_ALGORITHM_CATALOG[key]
            if class_path and class_path != template.class_path:
                raise ValueError(
                    f"algorithm.key={key!r} resolves to {template.class_path}, which does not "
                    f"match algorithm.class_path={class_path!r}."
                )
            return template
        if class_path in FLUKE_ALGORITHM_CATALOG:
            return FLUKE_ALGORITHM_CATALOG[class_path]
        for template in FLUKE_ALGORITHM_CATALOG.values():
            if template.class_path == class_path:
                return template
    if not class_path:
        raise ValueError("algorithm requires either a catalog key or algorithm.class_path.")
    return _template(
        key=class_path.rsplit(".", 1)[-1].lower(),
        class_path=class_path,
        protocol=supported_protocol_for_class(class_path),
    )


def catalog_coverage_gaps() -> set[str]:
    """Return public Fluke algorithm classes missing a first-party contract.

    This is version-specific by design.  A Fluke upgrade that adds a public algorithm
    makes the test fail until its model/data/result contract is explicitly reviewed.
    """

    try:
        import inspect
        import pkgutil
        from fluke import algorithms
        from fluke.algorithms import CentralizedFL
        from fluke.algorithms.decentralized import DecentralizedFL
        from fluke.distr import ParallelAlgorithm
    except ImportError as exc:  # pragma: no cover - optional dependency behavior
        raise RuntimeError("Fluke support requires `pip install -e '.[fluke]'`.") from exc

    discovered: set[str] = {
        "fluke.algorithms.decentralized.GossipDFL",
        "fluke.distr.ParallelAlgorithm",
    }
    for module_info in pkgutil.walk_packages(algorithms.__path__, f"{algorithms.__name__}."):
        module = import_module(module_info.name)
        for _, candidate in vars(module).items():
            if not inspect.isclass(candidate) or candidate.__module__ != module.__name__:
                continue
            if candidate in {CentralizedFL, DecentralizedFL, ParallelAlgorithm}:
                continue
            if issubclass(candidate, (CentralizedFL, DecentralizedFL, ParallelAlgorithm)):
                discovered.add(f"{candidate.__module__}.{candidate.__name__}")
    return discovered - EXPECTED_PUBLIC_CLASS_PATHS
