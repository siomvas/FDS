"""Aggregator integration helpers for NVFlare 2.7.

Provides utilities to convert between NVFlare 2.7 DXO format and aggregator_factory
expected formats. Ensures seamless integration with existing aggregation strategies.
"""

from typing import List, Dict, Any
import logging

from nvflare.apis.dxo import DXO, DataKind

logger = logging.getLogger(__name__)


def dxo_list_to_aggregator_input(dxo_list: List[DXO]) -> List[Dict[str, Any]]:
    """Convert list of DXO objects to aggregator_factory input format.

    Args:
        dxo_list: List of DXO objects from clients

    Returns:
        List of dicts with keys: 'weights', 'num_samples', 'client_id'
    """
    aggregator_input = []

    for dxo in dxo_list:
        # Extract model weights
        if dxo.data_kind != DataKind.WEIGHTS:
            logger.warning(f"DXO data_kind is {dxo.data_kind}, expected WEIGHTS")
            continue

        weights = dxo.data

        # Extract metadata
        num_samples = dxo.get_meta_prop("num_samples", 1)  # Default to 1 if missing
        client_id = dxo.get_meta_prop("client_id", "unknown")

        aggregator_input.append({
            "weights": weights,
            "num_samples": num_samples,
            "client_id": client_id
        })

    return aggregator_input


def aggregator_output_to_dxo(aggregated_weights: Dict[str, Any]) -> DXO:
    """Convert aggregator_factory output to DXO format.

    Args:
        aggregated_weights: Dict with aggregated model weights

    Returns:
        DXO object with aggregated weights
    """
    dxo = DXO(data_kind=DataKind.WEIGHTS, data=aggregated_weights)
    dxo.set_meta_prop("aggregated", True)

    return dxo


def extract_client_metrics(dxo_list: List[DXO]) -> Dict[str, Dict[str, float]]:
    """Extract training metrics from client DXOs.

    Args:
        dxo_list: List of DXO objects from clients

    Returns:
        Dict mapping client_id to metrics dict
    """
    client_metrics = {}

    for dxo in dxo_list:
        client_id = dxo.get_meta_prop("client_id", "unknown")
        metrics = dxo.get_meta_prop("metrics", {})

        if metrics:
            client_metrics[client_id] = metrics

    return client_metrics
