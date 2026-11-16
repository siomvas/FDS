"""Fluke-backed downstream federated-learning evaluation for existing partitions.

The package is intentionally isolated from the NVFlare training path.  Importing this
module does not import the optional ``fluke-fl`` dependency; use ``python -m
fl_training.fluke_downstream_fl_evaluation`` after installing the ``fluke`` extra.
"""

from __future__ import annotations

__all__ = ["catalog", "data", "models", "runner"]
