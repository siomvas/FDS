"""Canonical outcome schema for research runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


SCHEMA_VERSION = "1"
VALID_RESULT_TYPES = {"partition_eval", "fl_run", "audit_run"}
VALID_STATUSES = {"success", "failed", "partial"}


def _require_mapping(name: str, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping.")
    return value


@dataclass
class Result:
    """Validated run outcome."""

    run_id: str
    result_type: str
    status: str
    summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    failure: Optional[Dict[str, Any]] = None
    completed_at_utc: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Result.run_id is required.")
        if self.result_type not in VALID_RESULT_TYPES:
            raise ValueError(f"Unsupported result_type: {self.result_type}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unsupported status: {self.status}")
        self.summary = _require_mapping("summary", self.summary)
        self.metrics = _require_mapping("metrics", self.metrics)
        self.artifacts = _require_mapping("artifacts", self.artifacts)
        if self.failure is not None and not isinstance(self.failure, dict):
            raise TypeError("failure must be a mapping when provided.")
        if self.status == "failed":
            if not self.failure:
                raise ValueError("Failed results must include failure details.")
            if "error_type" not in self.failure or "error_message" not in self.failure:
                raise ValueError("failure must include error_type and error_message.")
        elif self.failure:
            raise ValueError("Only failed results may include failure details.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "result_type": self.result_type,
            "status": self.status,
            "completed_at_utc": self.completed_at_utc,
            "summary": self.summary,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Result":
        if not isinstance(payload, dict):
            raise TypeError("Result payload must be a mapping.")
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            run_id=str(payload["run_id"]),
            result_type=str(payload["result_type"]),
            status=str(payload["status"]),
            completed_at_utc=payload.get("completed_at_utc"),
            summary=_require_mapping("summary", payload.get("summary")),
            metrics=_require_mapping("metrics", payload.get("metrics")),
            artifacts=_require_mapping("artifacts", payload.get("artifacts")),
            failure=payload.get("failure"),
        )
