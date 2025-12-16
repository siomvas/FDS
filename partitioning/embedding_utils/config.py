"""
Configuration system for embedding-based partitioning.

Provides dataclasses for configuring the entire embedding pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for embedding extraction model."""

    model_type: str  # "dinov2", "clip", "siglip"
    variant: str  # Model variant (e.g., "base", "large")
    device: Optional[str] = None  # "cuda", "cpu", or None for auto
    batch_size: int = 32
    num_workers: int = 4
    proxy: Optional[str] = None  # HTTP proxy for model downloads

    def __post_init__(self):
        """Validate model configuration."""
        valid_types = ["dinov2", "clip", "siglip"]
        if self.model_type.lower() not in valid_types:
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                f"Choose from {valid_types}"
            )


@dataclass
class PoolingConfig:
    """Configuration for pooling strategy."""

    strategy: str  # "cls", "mean", "max", "gem", "lastk", "spectral"
    params: Dict[str, Any] = field(default_factory=dict)  # Strategy-specific params

    def __post_init__(self):
        """Validate pooling configuration."""
        valid_strategies = ["cls", "mean", "max", "gem", "lastk", "spectral"]
        if self.strategy.lower() not in valid_strategies:
            raise ValueError(
                f"Invalid pooling strategy: {self.strategy}. "
                f"Choose from {valid_strategies}"
            )


@dataclass
class ClusteringConfig:
    """Configuration for clustering method."""

    method: str  # "hdbscan", "kmeans", "knn", "spectral"
    params: Dict[str, Any] = field(default_factory=dict)  # Method-specific params

    def __post_init__(self):
        """Validate clustering configuration."""
        valid_methods = ["hdbscan", "kmeans", "knn", "spectral"]
        if self.method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid clustering method: {self.method}. "
                f"Choose from {valid_methods}"
            )


@dataclass
class CacheConfig:
    """Configuration for embedding caching."""

    enabled: bool = True
    cache_dir: str = "./.embedding_cache"


@dataclass
class AssignmentConfig:
    """Configuration for cluster-to-client assignment."""

    strategy: str = "round_robin"  # "round_robin", "random", "size_balanced"
    min_samples_per_client: int = 1  # Minimum samples per client
    handle_noise: str = "distribute"  # "distribute", "ignore", "separate_client"
    random_seed: Optional[int] = 42


@dataclass
class EmbeddingConfig:
    """
    Complete configuration for embedding-based partitioning pipeline.

    This is the main configuration class that combines all sub-configurations.
    """

    # Core components
    model: ModelConfig
    pooling: PoolingConfig
    clustering: ClusteringConfig

    # Number of clients
    n_clients: int

    # Optional components
    cache: CacheConfig = field(default_factory=CacheConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)

    # Data configuration
    valset_type: str = "global"  # "global", "local", or "mixed"

    # Output configuration
    output_dir: str = "./embedding_splits"
    save_embeddings: bool = False  # Whether to save embeddings alongside splits
    save_cluster_info: bool = True  # Whether to save clustering metadata

    def __post_init__(self):
        """Validate configuration."""
        if self.n_clients < 1:
            raise ValueError("n_clients must be >= 1")

        valid_valset_types = ["global", "local", "mixed"]
        if self.valset_type not in valid_valset_types:
            raise ValueError(
                f"Invalid valset_type: {self.valset_type}. "
                f"Choose from {valid_valset_types}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Convert configuration to JSON.

        Args:
            filepath: If provided, save to this file

        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            Path(filepath).write_text(json_str)

        return json_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EmbeddingConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            EmbeddingConfig instance
        """
        # Reconstruct nested dataclasses
        model = ModelConfig(**config_dict["model"])
        pooling = PoolingConfig(**config_dict["pooling"])
        clustering = ClusteringConfig(**config_dict["clustering"])

        # Optional configs
        cache = CacheConfig(**config_dict.get("cache", {}))
        assignment = AssignmentConfig(**config_dict.get("assignment", {}))

        # Create main config
        return cls(
            model=model,
            pooling=pooling,
            clustering=clustering,
            n_clients=config_dict["n_clients"],
            cache=cache,
            assignment=assignment,
            valset_type=config_dict.get("valset_type", "global"),
            output_dir=config_dict.get("output_dir", "./embedding_splits"),
            save_embeddings=config_dict.get("save_embeddings", False),
            save_cluster_info=config_dict.get("save_cluster_info", True),
        )

    @classmethod
    def from_json(cls, filepath: str) -> "EmbeddingConfig":
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            EmbeddingConfig instance
        """
        config_dict = json.loads(Path(filepath).read_text())
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        return (
            f"EmbeddingConfig(\n"
            f"  model={self.model.model_type}/{self.model.variant},\n"
            f"  pooling={self.pooling.strategy},\n"
            f"  clustering={self.clustering.method},\n"
            f"  n_clients={self.n_clients}\n"
            f")"
        )


# Preset configurations for common use cases
def get_preset_config(preset: str, n_clients: int) -> EmbeddingConfig:
    """
    Get a preset configuration.

    Args:
        preset: Preset name ("default", "fast", "high_quality")
        n_clients: Number of clients

    Returns:
        EmbeddingConfig instance

    Available presets:
        - "default": DINOv2-base + CLS pooling + HDBSCAN
        - "fast": DINOv2-small + CLS pooling + K-means
        - "high_quality": DINOv2-large + GeM pooling + HDBSCAN
        - "spectral": DINOv2-base + Spectral sharpening + Spectral clustering
    """
    presets = {
        "default": EmbeddingConfig(
            model=ModelConfig(model_type="dinov2", variant="base"),
            pooling=PoolingConfig(strategy="cls"),
            clustering=ClusteringConfig(method="hdbscan", params={"min_cluster_size": 15}),
            n_clients=n_clients,
        ),
        "fast": EmbeddingConfig(
            model=ModelConfig(model_type="dinov2", variant="small", batch_size=64),
            pooling=PoolingConfig(strategy="cls"),
            clustering=ClusteringConfig(method="kmeans", params={"n_clusters": n_clients}),
            n_clients=n_clients,
        ),
        "high_quality": EmbeddingConfig(
            model=ModelConfig(model_type="dinov2", variant="large", batch_size=16),
            pooling=PoolingConfig(strategy="gem", params={"p": 3.0}),
            clustering=ClusteringConfig(method="hdbscan", params={"min_cluster_size": 10}),
            n_clients=n_clients,
        ),
        "spectral": EmbeddingConfig(
            model=ModelConfig(model_type="dinov2", variant="base"),
            pooling=PoolingConfig(strategy="spectral", params={"k": 50}),
            clustering=ClusteringConfig(
                method="spectral",
                params={"n_clusters": n_clients, "affinity": "rbf"},
            ),
            n_clients=n_clients,
        ),
    }

    if preset not in presets:
        raise ValueError(
            f"Unknown preset: {preset}. Choose from {list(presets.keys())}"
        )

    return presets[preset]
