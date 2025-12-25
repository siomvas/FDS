"""Configuration system for embedding-based data partitioning.

This module provides a clean, user-friendly configuration API for the embedding
pipeline. It supports:
- Quick start with sensible presets
- Fine-grained customization
- Serialization/deserialization for reproducibility
- Validation of all parameters

User Stories Addressed:
1. Quick experimentation with get_preset_config()
2. Custom configurations with builder-style API
3. Save/load from JSON for reproducibility
4. Programmatic access and modification
5. Simple cache management
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import json


# =============================================================================
# Core Configuration Classes
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for embedding extraction model.

    Args:
        model_type: Model family - "dinov2", "clip", or "siglip"
        variant: Model size variant (e.g., "base", "large", "small")
        device: Device for inference - "cuda", "cpu", or None for auto-detect
        batch_size: Batch size for embedding extraction
        num_workers: DataLoader workers for parallel processing
        proxy: Optional HTTP proxy for model downloads (HPC environments)

    Examples:
        >>> # Default DINOv2 base
        >>> config = ModelConfig(model_type="dinov2", variant="base")

        >>> # CLIP with custom batch size
        >>> config = ModelConfig(
        ...     model_type="clip",
        ...     variant="base-patch16",
        ...     batch_size=64
        ... )
    """

    model_type: str
    variant: str
    device: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    proxy: Optional[str] = None

    def __post_init__(self):
        """Validate model configuration."""
        valid_types = ["dinov2", "clip", "siglip"]
        if self.model_type.lower() not in valid_types:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. "
                f"Must be one of: {valid_types}"
            )


@dataclass
class PoolingConfig:
    """Configuration for spatial pooling strategy.

    Args:
        strategy: Pooling method - "cls", "mean", "max", "gem", "lastk", "spectral"
        params: Strategy-specific parameters (e.g., {"p": 3.0} for GeM pooling)

    Examples:
        >>> # CLS token pooling (default for transformers)
        >>> config = PoolingConfig(strategy="cls")

        >>> # Generalized mean pooling with p=3
        >>> config = PoolingConfig(strategy="gem", params={"p": 3.0})
    """

    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate pooling configuration."""
        valid_strategies = ["cls", "mean", "max", "gem", "lastk", "spectral"]
        if self.strategy.lower() not in valid_strategies:
            raise ValueError(
                f"Invalid pooling strategy '{self.strategy}'. "
                f"Must be one of: {valid_strategies}"
            )


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithm.

    Args:
        method: Clustering algorithm - "hdbscan", "kmeans", "knn", "spectral"
        params: Algorithm-specific parameters
            - HDBSCAN: {"min_cluster_size": int, "min_samples": int}
            - KMeans: {"n_clusters": int}
            - Spectral: {"n_clusters": int, "affinity": str}

    Examples:
        >>> # HDBSCAN with automatic cluster detection
        >>> config = ClusteringConfig(
        ...     method="hdbscan",
        ...     params={"min_cluster_size": 15}
        ... )

        >>> # KMeans with fixed number of clusters
        >>> config = ClusteringConfig(
        ...     method="kmeans",
        ...     params={"n_clusters": 10}
        ... )
    """

    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate clustering configuration."""
        valid_methods = ["hdbscan", "kmeans", "knn", "spectral"]
        if self.method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid clustering method '{self.method}'. "
                f"Must be one of: {valid_methods}"
            )


@dataclass
class EmbeddingConfig:
    """Complete configuration for embedding-based data partitioning.

    This is the main configuration class that orchestrates the entire pipeline.

    Args:
        model: Model configuration for embedding extraction
        pooling: Pooling strategy configuration
        clustering: Clustering algorithm configuration
        n_clients: Number of federated learning clients
        cache_dir: Directory for caching embeddings (default: /path/to/gtai/FDS/embedding_cache)
        cache_enabled: Whether to enable embedding caching
        assignment_strategy: How to assign clusters to clients -
            "round_robin", "random", or "size_balanced"
        handle_noise: How to handle HDBSCAN noise points -
            "distribute", "ignore", or "separate_client"

    Examples:
        >>> # Using preset (recommended for quick start)
        >>> config = get_preset_config("default", n_clients=10)

        >>> # Custom configuration
        >>> config = EmbeddingConfig(
        ...     model=ModelConfig(model_type="dinov2", variant="base"),
        ...     pooling=PoolingConfig(strategy="cls"),
        ...     clustering=ClusteringConfig(method="hdbscan"),
        ...     n_clients=10
        ... )

        >>> # Modify cache settings
        >>> config.cache_dir = "/scratch/embeddings"
        >>> config.cache_enabled = True
    """

    # Core pipeline components
    model: ModelConfig
    pooling: PoolingConfig
    clustering: ClusteringConfig
    n_clients: int

    # Cache settings (simplified from old nested CacheConfig)
    cache_dir: str = "./.embedding_cache"
    cache_enabled: bool = True

    # Cluster assignment settings
    assignment_strategy: str = "round_robin"
    handle_noise: str = "distribute"

    def __post_init__(self):
        """Validate configuration."""
        if self.n_clients < 1:
            raise ValueError("n_clients must be >= 1")

        valid_assignments = ["round_robin", "random", "size_balanced"]
        if self.assignment_strategy not in valid_assignments:
            raise ValueError(
                f"Invalid assignment_strategy '{self.assignment_strategy}'. "
                f"Must be one of: {valid_assignments}"
            )

        valid_noise = ["distribute", "ignore", "separate_client"]
        if self.handle_noise not in valid_noise:
            raise ValueError(
                f"Invalid handle_noise '{self.handle_noise}'. "
                f"Must be one of: {valid_noise}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of config

        Examples:
            >>> config = get_preset_config("default", n_clients=10)
            >>> config_dict = config.to_dict()
            >>> assert "model" in config_dict
            >>> assert "n_clients" in config_dict
        """
        return asdict(self)

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Save configuration to JSON file.

        Args:
            filepath: Path to save JSON file. If None, returns JSON string.

        Returns:
            JSON string representation

        Examples:
            >>> config = get_preset_config("default", n_clients=10)
            >>> config.to_json("my_config.json")
            >>>
            >>> # Later, reload it
            >>> loaded = EmbeddingConfig.from_json("my_config.json")
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            Path(filepath).write_text(json_str)

        return json_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EmbeddingConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            EmbeddingConfig instance

        Examples:
            >>> config_dict = {
            ...     "model": {"model_type": "dinov2", "variant": "base"},
            ...     "pooling": {"strategy": "cls"},
            ...     "clustering": {"method": "hdbscan"},
            ...     "n_clients": 10
            ... }
            >>> config = EmbeddingConfig.from_dict(config_dict)
        """
        # Reconstruct nested dataclasses
        model = ModelConfig(**config_dict["model"])
        pooling = PoolingConfig(**config_dict["pooling"])
        clustering = ClusteringConfig(**config_dict["clustering"])

        # Build config with all fields
        return cls(
            model=model,
            pooling=pooling,
            clustering=clustering,
            n_clients=config_dict["n_clients"],
            cache_dir=config_dict.get("cache_dir", "./.embedding_cache"),
            cache_enabled=config_dict.get("cache_enabled", True),
            assignment_strategy=config_dict.get("assignment_strategy", "round_robin"),
            handle_noise=config_dict.get("handle_noise", "distribute"),
        )

    @classmethod
    def from_json(cls, filepath: str) -> "EmbeddingConfig":
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            EmbeddingConfig instance

        Examples:
            >>> config = EmbeddingConfig.from_json("saved_config.json")
            >>> splits = splitter.get_client_splits()
        """
        config_dict = json.loads(Path(filepath).read_text())
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """User-friendly string representation."""
        return (
            f"EmbeddingConfig(\n"
            f"  model={self.model.model_type}-{self.model.variant}\n"
            f"  pooling={self.pooling.strategy}\n"
            f"  clustering={self.clustering.method}\n"
            f"  n_clients={self.n_clients}\n"
            f"  cache={'enabled' if self.cache_enabled else 'disabled'}\n"
            f")"
        )


# =============================================================================
# Preset Configurations
# =============================================================================


def get_preset_config(preset: str, n_clients: int) -> EmbeddingConfig:
    """Get a pre-configured embedding pipeline setup.

    This is the recommended way to get started with embedding-based partitioning.
    Presets provide sensible defaults for common use cases.

    Args:
        preset: Preset name - "default", "fast", "high_quality", or "spectral"
        n_clients: Number of federated learning clients

    Returns:
        EmbeddingConfig instance ready to use

    Raises:
        ValueError: If preset name is unknown

    Available Presets:
        - **default**: Balanced quality/speed with DINOv2-base + HDBSCAN
            - Model: DINOv2 base (86M params)
            - Pooling: CLS token
            - Clustering: HDBSCAN (automatic cluster detection)
            - Use case: General-purpose, works well for most datasets

        - **fast**: Fastest option with DINOv2-small + KMeans
            - Model: DINOv2 small (22M params)
            - Pooling: CLS token
            - Clustering: KMeans (fixed K = n_clients)
            - Use case: Quick experiments, large datasets, limited compute

        - **high_quality**: Best quality with DINOv2-large + GeM + HDBSCAN
            - Model: DINOv2 large (300M params)
            - Pooling: GeM pooling (more robust than CLS)
            - Clustering: HDBSCAN (min_cluster_size=10)
            - Use case: Production, research papers, when quality matters

        - **spectral**: Spectral clustering with sharpening
            - Model: DINOv2 base
            - Pooling: Spectral sharpening
            - Clustering: Spectral clustering
            - Use case: Complex data distributions, research experiments

    Examples:
        >>> # Quick start for 10 clients
        >>> config = get_preset_config("default", n_clients=10)
        >>>
        >>> # Fast configuration for 50 clients
        >>> config = get_preset_config("fast", n_clients=50)
        >>>
        >>> # High quality for paper experiments
        >>> config = get_preset_config("high_quality", n_clients=20)
        >>>
        >>> # Modify preset settings
        >>> config = get_preset_config("default", n_clients=10)
        >>> config.model.batch_size = 64  # Increase batch size
        >>> config.cache_dir = "/scratch/cache"  # Custom cache location
    """

    presets = {
        "default": EmbeddingConfig(
            model=ModelConfig(
                model_type="dinov2",
                variant="base",
                batch_size=32,
            ),
            pooling=PoolingConfig(strategy="cls"),
            clustering=ClusteringConfig(
                method="hdbscan",
                params={"min_cluster_size": 15}
            ),
            n_clients=n_clients,
            cache_enabled=True,
            assignment_strategy="round_robin",
        ),

        "fast": EmbeddingConfig(
            model=ModelConfig(
                model_type="dinov2",
                variant="small",
                batch_size=64,
            ),
            pooling=PoolingConfig(strategy="cls"),
            clustering=ClusteringConfig(
                method="kmeans",
                params={"n_clusters": n_clients}
            ),
            n_clients=n_clients,
            cache_enabled=True,
            assignment_strategy="round_robin",
        ),

        "high_quality": EmbeddingConfig(
            model=ModelConfig(
                model_type="dinov2",
                variant="large",
                batch_size=16,
            ),
            pooling=PoolingConfig(
                strategy="gem",
                params={"p": 3.0}
            ),
            clustering=ClusteringConfig(
                method="hdbscan",
                params={"min_cluster_size": 10}
            ),
            n_clients=n_clients,
            cache_enabled=True,
            assignment_strategy="size_balanced",
        ),

        "spectral": EmbeddingConfig(
            model=ModelConfig(
                model_type="dinov2",
                variant="base",
                batch_size=32,
            ),
            pooling=PoolingConfig(
                strategy="spectral",
                params={"k": 50}
            ),
            clustering=ClusteringConfig(
                method="spectral",
                params={
                    "n_clusters": n_clients,
                    "affinity": "rbf"
                }
            ),
            n_clients=n_clients,
            cache_enabled=True,
            assignment_strategy="round_robin",
        ),
    }

    if preset not in presets:
        raise ValueError(
            f"Unknown preset '{preset}'. "
            f"Available presets: {list(presets.keys())}"
        )

    return presets[preset]


# =============================================================================
# Convenience Functions
# =============================================================================


def create_custom_config(
    n_clients: int,
    model_type: str = "dinov2",
    variant: str = "base",
    clustering_method: str = "hdbscan",
    **kwargs
) -> EmbeddingConfig:
    """Create a custom configuration with common parameters.

    This is a convenience function for users who want to customize a few
    parameters without specifying all nested dataclasses.

    Args:
        n_clients: Number of clients
        model_type: Model family (default: "dinov2")
        variant: Model variant (default: "base")
        clustering_method: Clustering algorithm (default: "hdbscan")
        **kwargs: Additional parameters to override
            - batch_size: Model batch size
            - pooling_strategy: Pooling method
            - cache_dir: Cache directory
            - cache_enabled: Enable/disable caching

    Returns:
        EmbeddingConfig instance

    Examples:
        >>> # Simple custom config
        >>> config = create_custom_config(
        ...     n_clients=10,
        ...     model_type="clip",
        ...     variant="base-patch16"
        ... )

        >>> # With additional customization
        >>> config = create_custom_config(
        ...     n_clients=20,
        ...     model_type="dinov2",
        ...     variant="large",
        ...     batch_size=64,
        ...     pooling_strategy="gem",
        ...     cache_dir="/scratch/cache"
        ... )
    """

    # Extract kwargs
    batch_size = kwargs.get("batch_size", 32)
    pooling_strategy = kwargs.get("pooling_strategy", "cls")
    cache_dir = kwargs.get("cache_dir", "./.embedding_cache")
    cache_enabled = kwargs.get("cache_enabled", True)

    # Build config
    return EmbeddingConfig(
        model=ModelConfig(
            model_type=model_type,
            variant=variant,
            batch_size=batch_size,
        ),
        pooling=PoolingConfig(strategy=pooling_strategy),
        clustering=ClusteringConfig(method=clustering_method),
        n_clients=n_clients,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )
