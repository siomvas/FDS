"""Experiment configuration schema for FDS federated learning.

Defines user-facing YAML configuration structure for FL experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str
    task_type: str  # "classification", "segmentation", "detection"
    dataset: str
    dataset_splits: str  # Path to DatasetSplits JSON


@dataclass
class FederatedLearningConfig:
    """Federated learning configuration."""
    num_rounds: int = 10
    num_clients: int = 10
    aggregator: str = "fedavg"
    aggregator_config: Dict[str, Any] = field(default_factory=dict)
    min_clients: Optional[int] = None  # Defaults to num_clients
    wait_time: int = 10  # seconds


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    model: str
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    amp_enabled: bool = False
    grad_tracking: bool = False
    seed: int = 42


@dataclass
class ResourceConfig:
    """Resource requirements (optional)."""
    client_gpu_mem: str = "4GB"
    server_cpu: int = 2


@dataclass
class FDSExperimentConfig:
    """Complete FDS experiment configuration.

    Load from YAML:
        config = FDSExperimentConfig.from_yaml("experiment.yaml")

    Access fields:
        config.experiment.name
        config.federated_learning.num_rounds
        config.training.learning_rate
    """
    experiment: ExperimentConfig
    federated_learning: FederatedLearningConfig
    training: TrainingConfig
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FDSExperimentConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            FDSExperimentConfig instance
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            experiment=ExperimentConfig(**data['experiment']),
            federated_learning=FederatedLearningConfig(**data['federated_learning']),
            training=TrainingConfig(**data['training']),
            resources=ResourceConfig(**data.get('resources', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
