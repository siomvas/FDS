"""Canonical provenance manifest for research runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


SCHEMA_VERSION = "1"


def _require_mapping(name: str, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping.")
    return value


@dataclass
class Manifest:
    """Publication-grade provenance captured for one atomic run."""

    run_id: str
    created_at_utc: str
    producer: Dict[str, Any]
    git: Dict[str, Any]
    resolved_config: Dict[str, Any]
    dataset: Dict[str, Any]
    partition: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Manifest.run_id is required.")
        if not self.created_at_utc:
            raise ValueError("Manifest.created_at_utc is required.")
        self.producer = _require_mapping("producer", self.producer)
        self.git = _require_mapping("git", self.git)
        self.resolved_config = _require_mapping("resolved_config", self.resolved_config)
        self.dataset = _require_mapping("dataset", self.dataset)
        self.partition = _require_mapping("partition", self.partition)
        self.model = _require_mapping("model", self.model)
        self.environment = _require_mapping("environment", self.environment)
        self.artifacts = _require_mapping("artifacts", self.artifacts)
        self.extras = _require_mapping("extras", self.extras)
        if "script_name" not in self.producer or not self.producer["script_name"]:
            raise ValueError("Manifest.producer.script_name is required.")
        if "name" not in self.dataset or not self.dataset["name"]:
            raise ValueError("Manifest.dataset.name is required.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "producer": self.producer,
            "git": self.git,
            "resolved_config": self.resolved_config,
            "dataset": self.dataset,
            "partition": self.partition,
            "model": self.model,
            "seed": self.seed,
            "environment": self.environment,
            "artifacts": self.artifacts,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Manifest":
        if not isinstance(payload, dict):
            raise TypeError("Manifest payload must be a mapping.")
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            run_id=str(payload["run_id"]),
            created_at_utc=str(payload["created_at_utc"]),
            producer=_require_mapping("producer", payload.get("producer")),
            git=_require_mapping("git", payload.get("git")),
            resolved_config=_require_mapping("resolved_config", payload.get("resolved_config")),
            dataset=_require_mapping("dataset", payload.get("dataset")),
            partition=_require_mapping("partition", payload.get("partition")),
            model=_require_mapping("model", payload.get("model")),
            seed=payload.get("seed"),
            environment=_require_mapping("environment", payload.get("environment")),
            artifacts=_require_mapping("artifacts", payload.get("artifacts")),
            extras=_require_mapping("extras", payload.get("extras")),
        )
