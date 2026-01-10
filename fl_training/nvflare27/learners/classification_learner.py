"""ClassificationLearner: Classification-specific learner for NVFlare 2.7 Client API.

Extends BaseLearner with classification model creation, training loop, and evaluation.
Integrates with existing model_factory for model instantiation and handles HPC proxy
requirements for pretrained model downloads.

Task-specific responsibilities:
- Create classification models via model_factory
- Implement cross-entropy loss training loop
- Compute accuracy and loss metrics
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_learner import BaseLearner


class ClassificationLearner(BaseLearner):
    """Classification-specific learner extending BaseLearner.

    Implements classification model creation, training, and evaluation using
    cross-entropy loss and accuracy metrics. Integrates with existing model_factory
    for model instantiation.

    Args:
        data_path: Path to dataset splits (DatasetSplits JSON)
        model_name: Model architecture identifier (vgg11, resnet18, etc.)
        config: Training configuration dict with keys:
            - num_classes (int): Number of output classes (REQUIRED)
            - lr: Learning rate
            - local_epochs: Epochs per round
            - batch_size: Training batch size
            - device: torch device
            - amp_enabled: Use automatic mixed precision
            - grad_tracking_enabled: Enable gradient diversity tracking
            - seed: Random seed
        client_id: Client identifier (auto-detected if None)

    Raises:
        ValueError: If num_classes not in config

    Examples:
        >>> config = {
        ...     "num_classes": 200,
        ...     "lr": 0.001,
        ...     "local_epochs": 1,
        ...     "batch_size": 32,
        ...     "device": "cuda",
        ...     "splits_path": "outputs/splits.json",
        ...     "task_type": "classification"
        ... }
        >>> learner = ClassificationLearner(
        ...     data_path="outputs/splits.json",
        ...     model_name="vgg11",
        ...     config=config
        ... )
        >>> learner.execute()
    """

    def __init__(
        self,
        data_path: str,
        model_name: str,
        config: Dict[str, Any],
        client_id: Optional[str] = None
    ):
        # Initialize base learner
        super().__init__(data_path, model_name, config, client_id)

        # Extract and validate num_classes
        if "num_classes" not in config:
            raise ValueError(
                "Missing required config key 'num_classes'. "
                "ClassificationLearner requires num_classes to be specified. "
                "Example: config['num_classes'] = 200"
            )

        self.num_classes = config["num_classes"]

    def create_model(self) -> nn.Module:
        """Create classification model architecture.

        TODO: Implement model creation using model_factory.
        """
        pass

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch with classification-specific logic.

        TODO: Implement training loop with cross-entropy loss.
        """
        pass

    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model with classification-specific metrics.

        TODO: Implement evaluation with accuracy and loss metrics.
        """
        pass
