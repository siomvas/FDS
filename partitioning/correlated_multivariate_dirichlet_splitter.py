from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .multivariate_dirichlet_splitter import MultivariateDirichletSplitter

logger = logging.getLogger(__name__)


def _canonicalize_label(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_LABEL_ALIASES = {
    "nature outdoors": ("natural", "outdoor"),
    "natural outdoors": ("natural", "outdoor"),
    "urban outdoors": ("urban", "outdoor"),
    "sunset dusk": ("dusk", "sunset"),
    "office indoors": ("indoor", "office"),
    "residential indoors": ("indoor", "residential"),
}


@dataclass(frozen=True)
class CorrelationRule:
    left_criterion: str
    left_labels: Sequence[str]
    right_criterion: str
    right_labels: Sequence[str]


@dataclass(frozen=True)
class ResolvedCorrelationRule:
    left_criterion: str
    left_labels: frozenset[str]
    right_criterion: str
    right_labels: frozenset[str]

    def matches(self, labels: Dict[str, str]) -> bool:
        return (
            labels.get(self.left_criterion) in self.left_labels
            and labels.get(self.right_criterion) in self.right_labels
        )


class CorrelatedMultivariateDirichletSplitter(MultivariateDirichletSplitter):
    """Multivariate splitter with additive bonuses for selected cross-criterion pairs."""

    _RULES: Dict[str, List[CorrelationRule]] = {
        "tinyimagenet": [
            CorrelationRule("location", ("Nature Outdoors", "Urban Outdoors", "Aquatic"), "time", ("Day", "Sunset/Dusk")),
            CorrelationRule("location", ("Indoors",), "time", ("Night", "Unknown")),
        ],
        "ade20k": [
            CorrelationRule("scene", ("Natural Outdoors", "Urban Outdoors"), "light", ("Intense", "Moderate")),
            CorrelationRule("scene", ("Office Indoors", "Residential Indoors"), "light", ("Dim", "Moderate")),
        ],
        "food101": [
            CorrelationRule("angle", ("Overhead/Top-down",), "lighting", ("Bright", "Intense")),
            CorrelationRule("angle", ("Closeup",), "lighting", ("Moderate", "Dim")),
        ],
        "pascalvoc": [
            CorrelationRule("location", ("Nature", "Urban Outdoors"), "light", ("Bright", "Intense")),
            CorrelationRule("location", ("Indoors",), "light", ("Dim", "Moderate")),
        ],
    }

    def __init__(self, *args, correlation_strength: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.correlation_strength = float(correlation_strength)
        self._resolved_rules: List[ResolvedCorrelationRule] = []

    def _dataset_rule_key(self) -> str:
        dataset_name = str(getattr(self.train_dataset, "dataset_name", getattr(self.train_dataset, "requested_dataset_name", "")))
        return _canonicalize_label(dataset_name).replace(" ", "")

    @staticmethod
    def _resolve_label_aliases(available_labels: Sequence[str], configured_labels: Sequence[str]) -> frozenset[str]:
        normalized_map = {_canonicalize_label(label): label for label in available_labels}
        resolved: List[str] = []
        for configured in configured_labels:
            canonical = _canonicalize_label(configured)
            if canonical in normalized_map:
                resolved.append(normalized_map[canonical])
                continue
            for alias in _LABEL_ALIASES.get(canonical, ()):
                if alias in normalized_map:
                    resolved.append(normalized_map[alias])
        return frozenset(resolved)

    def _resolve_rules(self) -> None:
        self._resolved_rules = []
        rules = self._RULES.get(self._dataset_rule_key(), [])
        for rule in rules:
            left_available = self.unique_labels.get(rule.left_criterion, [])
            right_available = self.unique_labels.get(rule.right_criterion, [])
            left_labels = self._resolve_label_aliases(left_available, rule.left_labels)
            right_labels = self._resolve_label_aliases(right_available, rule.right_labels)
            if not left_labels or not right_labels:
                logger.warning(
                    "Skipping unresolved correlation rule for dataset=%s: %s -> %s",
                    self._dataset_rule_key(),
                    rule.left_criterion,
                    rule.right_criterion,
                )
                continue
            self._resolved_rules.append(
                ResolvedCorrelationRule(
                    left_criterion=rule.left_criterion,
                    left_labels=left_labels,
                    right_criterion=rule.right_criterion,
                    right_labels=right_labels,
                )
            )

    def _load_cluster_labels(self) -> None:
        super()._load_cluster_labels()
        self._resolve_rules()

    def _sample_labels_for_image(self, image_path: str) -> Dict[str, str]:
        return {criterion: self._get_cluster_label_for_sample(image_path, criterion) for criterion in self.CRITERIA}

    def _compute_affinity_score(self, image_path: str, client_id: int) -> float:
        affinity = super()._compute_affinity_score(image_path, client_id)
        if not self._resolved_rules or self.correlation_strength == 0.0:
            return affinity

        sample_labels = self._sample_labels_for_image(image_path)
        for rule in self._resolved_rules:
            if not rule.matches(sample_labels):
                continue
            left_label = sample_labels[rule.left_criterion]
            right_label = sample_labels[rule.right_criterion]
            left_idx = self.unique_labels[rule.left_criterion].index(left_label)
            right_idx = self.unique_labels[rule.right_criterion].index(right_label)
            left_pref = self.client_preferences[client_id][rule.left_criterion][left_idx]
            right_pref = self.client_preferences[client_id][rule.right_criterion][right_idx]
            affinity += self.correlation_strength * (left_pref * right_pref)
        return affinity
