"""Embedding extraction and caching infrastructure for feature learning.

This package provides a comprehensive framework for extracting image embeddings
using various foundation models and alternative methods. It includes a factory
pattern for creating extractors and a sophisticated caching system for persistent
storage.

Components
----------

**CacheManager**
    Unified disk-based caching system supporting batch mode (ad-hoc queries)
    and dataset mode (large-scale datasets with subset retrieval). Includes
    in-memory caching for performance.

**ModelRegistry**
    Factory pattern registry for creating embedding extractors. Automatically
    discovers registered extractors and provides a unified creation interface.

**EmbeddingExtractor** (Abstract Base)
    Base class for all extractors. Handles model loading, batched inference,
    proxy support, and cache integration.

**Pre-trained Extractors**
    - **InceptionExtractor**: InceptionV3 for FID calculations (2048-dim)
    - **DINOv2Extractor**: Self-supervised ViT (384-1024-dim, 4 variants)
    - **CLIPExtractor**: Vision-language model (512-768-dim, multiple variants)
    - **SigLIPExtractor**: Enhanced CLIP (768-1152-dim)
    - **MAEExtractor**: Masked autoencoder (768-1024-dim)

**Untrained/Alternative Extractors**
    - **RawPixelResizerExtractor**: Flattened resized pixels
    - **SparseRandomProjectionExtractor**: Random projections
    - **UntrainedAutoEncoderExtractor**: Random encoder weights
    - **WaveletScatteringExtractor**: Wavelet transform features

Usage Example
-------------

Basic workflow with caching:

    >>> from embedding import ModelRegistry, CacheManager
    >>>
    >>> # Setup cache
    >>> cache = CacheManager(cache_dir=".cache")
    >>>
    >>> # Create extractor
    >>> extractor = ModelRegistry.create(
    ...     "dinov2-base",
    ...     device="cuda",
    ...     batch_size=64,
    ...     cache_manager=cache
    ... )
    >>>
    >>> # Extract with automatic caching
    >>> embeddings = extractor.extract_embeddings(
    ...     images=["img1.jpg", "img2.jpg"],
    ...     use_cache=True
    ... )
    >>> embeddings.shape)  # (2, 768

See Also
--------

- embedding.embedding_extractors: Detailed extractor documentation
- embedding.cache_manager: Caching system documentation
- evaluation.metrics: Metrics using these embeddings
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CacheManager": ("embedding.cache_manager", "CacheManager"),
    "ModelRegistry": ("embedding.embedding_extractors", "ModelRegistry"),
    "EmbeddingExtractor": ("embedding.embedding_extractors", "EmbeddingExtractor"),
    "InceptionExtractor": ("embedding.embedding_extractors", "InceptionExtractor"),
    "DINOv2Extractor": ("embedding.embedding_extractors", "DINOv2Extractor"),
    "CLIPExtractor": ("embedding.embedding_extractors", "CLIPExtractor"),
    "SigLIPExtractor": ("embedding.embedding_extractors", "SigLIPExtractor"),
    "MAEExtractor": ("embedding.embedding_extractors", "MAEExtractor"),
    "RawPixelResizerExtractor": ("embedding.embedding_extractors", "RawPixelResizerExtractor"),
    "SparseRandomProjectionResizerExtractor": (
        "embedding.embedding_extractors",
        "SparseRandomProjectionResizerExtractor",
    ),
    "UntrainedAutoEncoderExtractor": (
        "embedding.embedding_extractors",
        "UntrainedAutoEncoderExtractor",
    ),
    "WaveletScatteringExtractor": (
        "embedding.embedding_extractors",
        "WaveletScatteringExtractor",
    ),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))

__all__ = [
    # Cache Management
    "CacheManager",

    # Registry and Base
    "ModelRegistry",
    "EmbeddingExtractor",

    # Pre-trained Models
    "InceptionExtractor",
    "DINOv2Extractor",
    "CLIPExtractor",
    "SigLIPExtractor",
    "MAEExtractor",

    # Untrained/Alternative Methods
    "RawPixelResizerExtractor",
    "SparseRandomProjectionResizerExtractor",
    "UntrainedAutoEncoderExtractor",
    "WaveletScatteringExtractor",
]
