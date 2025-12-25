"""
Embedding-based data partitioning using foundation models and clustering.

This module provides tools for partitioning datasets using embeddings extracted
from frontier foundation models (DINO, CLIP, SigLIP) combined with clustering
algorithms (HDBSCAN, K-means, etc.).
"""

from embedding.embedding_extractors import (
    EmbeddingExtractor,
    DINOv2Extractor,
    CLIPExtractor,
    SigLIPExtractor,
)
from .pooling_strategies import (
    PoolingStrategy,
    CLSPooling,
    MeanPooling,
    MaxPooling,
    GeMLPooling,
    LastKLayerPooling,
    SpectralSharpeningPooling,
)
from .clustering_methods import (
    ClusteringMethod,
    HDBSCANMethod,
    KMeansMethod,
    KNNGraphMethod,
    SpectralMethod,
    get_clustering_method,
)

from embedding.cache_manager import CacheManager
from .config import (
    EmbeddingConfig,
    ModelConfig,
    PoolingConfig,
    ClusteringConfig,
    get_preset_config,
    create_custom_config
)
from .pooling_strategies import get_pooling_strategy
from .clustering_methods import get_clustering_method

__all__ = [
    # Base classes
    "EmbeddingExtractor",
    "PoolingStrategy",
    "ClusteringMethod",
    # Extractors
    "DINOv2Extractor",
    "CLIPExtractor",
    "SigLIPExtractor",
    # Pooling strategies
    "CLSPooling",
    "MeanPooling",
    "MaxPooling",
    "GeMLPooling",
    "LastKLayerPooling",
    "SpectralSharpeningPooling",
    "get_pooling_strategy",
    # Clustering methods
    "HDBSCANMethod",
    "KMeansMethod",
    "KNNGraphMethod",
    "SpectralMethod",
    "get_clustering_method",
    "CacheManager",
    # Configuration
    "EmbeddingConfig",
    "ModelConfig",
    "PoolingConfig",
    "ClusteringConfig",
    "get_preset_config",
    "create_custom_config",
]
