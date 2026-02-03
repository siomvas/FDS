"""FDSFedAvgController: Server-side orchestration for FDS federated learning.

Implements FedAvg workflow using NVFlare 2.7 ModelController API. Integrates with
existing aggregator_factory.py for aggregation strategies (FedAvg, FedProx, SCAFFOLD).

Architecture: ModelController-based orchestration (Phase 0 Research - DISCOVERY.md Section 3)
"""

from typing import Dict, Any, Optional, List
import logging

from nvflare.apis.dxo import DXO, from_shareable, DataKind
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.workflows.model_controller import ModelController
from nvflare.app_common.app_constant import AppConstants

# Existing FDS imports
from fl_training.nvflare_refactor.common.aggregator_factory import get_aggregator
from fl_mine.gradient_tracker import ServerGradientTracker

logger = logging.getLogger(__name__)


class FDSFedAvgController(ModelController):
    """FedAvg controller for FDS multi-task federated learning.

    Orchestrates FL training rounds:
    1. Broadcast global model to clients
    2. Clients perform local training (via BaseLearner)
    3. Aggregate client updates using aggregator_factory
    4. Update global model
    5. Track gradient diversity (optional)

    Args:
        num_rounds: Number of FL rounds
        num_clients_per_round: Clients per round (use all if None)
        aggregator_type: Aggregation strategy ("fedavg", "fedprox", "scaffold")
        aggregator_config: Config dict for aggregator
        grad_tracking_enabled: Enable server-side gradient tracking
        persistor_id: ID of model persistor component
        shareable_generator_id: ID of shareable generator
        min_clients: Minimum clients before starting round
        wait_time_after_min_received: Wait time after min_clients (seconds)
    """

    def __init__(
        self,
        num_rounds: int = 10,
        num_clients_per_round: Optional[int] = None,
        aggregator_type: str = "fedavg",
        aggregator_config: Optional[Dict[str, Any]] = None,
        grad_tracking_enabled: bool = False,
        persistor_id: str = "persistor",
        shareable_generator_id: str = "shareable_generator",
        min_clients: int = 2,
        wait_time_after_min_received: int = 10,
    ):
        super().__init__(
            num_rounds=num_rounds,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
        )

        self.num_clients_per_round = num_clients_per_round
        self.min_clients = min_clients
        self.wait_time_after_min_received = wait_time_after_min_received

        # Aggregation setup
        self.aggregator_type = aggregator_type
        self.aggregator_config = aggregator_config or {}
        self.aggregator = None  # Created in run()

        # Gradient tracking
        self.grad_tracking_enabled = grad_tracking_enabled
        self.grad_tracker: Optional[ServerGradientTracker] = None

        # Round state
        self.current_round = 0

    def run(self) -> None:
        """Execute federated learning workflow.

        Implements the standard FedAvg training loop:
        1. Initialize aggregator and gradient tracker
        2. For each round:
            a. Load current global model
            b. Select clients
            c. Send model and wait for results
            d. Aggregate client updates
            e. Update global model
            f. Track gradient diversity (optional)
            g. Save model
        """
        # Initialize aggregator
        self.aggregator = get_aggregator(
            self.aggregator_type,
            **self.aggregator_config
        )
        logger.info(f"Initialized {self.aggregator_type} aggregator with config: {self.aggregator_config}")

        # Initialize gradient tracker
        if self.grad_tracking_enabled:
            self.grad_tracker = ServerGradientTracker()
            logger.info("Server-side gradient tracking enabled")

        # FL training loop
        for round_num in range(self.num_rounds):
            self.current_round = round_num
            logger.info(f"Starting round {round_num + 1}/{self.num_rounds}")

            # 1. Load current global model
            model_shareable = self.load_model(fl_ctx=None)
            logger.info(f"Loaded global model for round {round_num}")

            # 2. Determine target clients
            targets = None  # None = all clients
            if self.num_clients_per_round:
                # Sample subset of clients
                # Note: sample_clients() handled by framework based on min_clients
                logger.info(f"Expecting {self.num_clients_per_round} clients per round")
            else:
                logger.info("Using all available clients")

            # 3. Send model to clients and wait for results
            # send_model_and_wait broadcasts model and collects client results
            results = self.send_model_and_wait(
                targets=targets,
                data=model_shareable,
                min_responses=self.min_clients,
                wait_time_after_min_received=self.wait_time_after_min_received
            )

            if not results:
                logger.warning(f"No results received in round {round_num}, skipping aggregation")
                continue

            logger.info(f"Received {len(results)} client results")

            # 4. Aggregate client updates
            aggregate_results = self.aggregate(
                results=results,
                aggregate_fn=self._aggregate_fn
            )

            # 5. Update global model with aggregated results
            model_shareable = self.update_model(
                curr_model=model_shareable,
                new_model=aggregate_results
            )

            # 6. Gradient tracking (if enabled)
            if self.grad_tracker:
                # Extract gradients from client results
                client_gradients = self._extract_gradients_from_results(results)
                if client_gradients:
                    diversity_metrics = self.grad_tracker.compute_cross_client_diversity(
                        client_gradients
                    )
                    logger.info(f"Round {round_num} gradient diversity: {diversity_metrics}")

            # 7. Save updated model
            self.save_model(model=model_shareable, fl_ctx=None)
            logger.info(f"Round {round_num} complete, model saved")

    def _aggregate_fn(self, results: List[Shareable]) -> Shareable:
        """Aggregate client results using aggregator_factory.

        Args:
            results: List of Shareable objects from clients

        Returns:
            Aggregated Shareable
        """
        # Convert Shareables to DXO format
        dxo_list = []
        for shareable in results:
            try:
                dxo = from_shareable(shareable)
                dxo_list.append(dxo)
            except Exception as e:
                logger.warning(f"Failed to convert shareable to DXO: {e}")
                continue

        if not dxo_list:
            raise ValueError("No valid DXO objects to aggregate")

        # Use aggregator from factory
        aggregated_dxo = self.aggregator.aggregate(dxo_list)

        # Convert back to Shareable
        aggregated_shareable = aggregated_dxo.to_shareable()

        return aggregated_shareable

    def _extract_gradients_from_results(self, results: List[Shareable]) -> Dict[str, Any]:
        """Extract gradient information from client results for tracking.

        Args:
            results: Client Shareables with metadata

        Returns:
            Dict mapping client_id to gradient statistics
        """
        client_gradients = {}

        for shareable in results:
            # Extract gradient metadata if present
            # Clients send gradient stats in metadata
            metadata = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)
            if metadata and "grad_stats" in metadata:
                client_id = shareable.get_peer_prop(key="client_name", default="unknown")
                client_gradients[client_id] = metadata["grad_stats"]

        return client_gradients
