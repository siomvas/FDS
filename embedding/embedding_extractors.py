"""Embedding extraction infrastructure using foundation models.

This module provides a unified framework for extracting image embeddings using
various pre-trained and untrained models. It implements a factory pattern (ModelRegistry)
for creating extractors and supports caching for efficient re-use of computed embeddings.

Architecture
------------

**ModelRegistry** (Factory Pattern)
    Central registry for creating embedding extractors. Automatically discovers
    and registers extractor classes decorated with @ModelRegistry.register().

**EmbeddingExtractor** (Abstract Base Class)
    Base class defining the interface for all embedding extractors. Handles
    common functionality like model loading, batched inference, centralized proxy
    configuration (via utils.proxy_config), and cache integration.

**Specific Extractors** (Concrete Implementations)
    - **InceptionExtractor**: InceptionV3 (PyTorch, 2048-dim)
    - **DINOv2Extractor**: Self-supervised ViT (384-1024-dim depending on variant)
    - **CLIPExtractor**: Contrastive vision-language model (512-768-dim)
    - **SigLIPExtractor**: Enhanced CLIP variant (768-1152-dim)
    - **MAEExtractor**: Masked autoencoder (768-1024-dim)
    - **RawPixelResizerExtractor**: Flattened raw pixels (no training)
    - **SparseRandomProjectionExtractor**: Random projections (no training)
    - **UntrainedAutoEncoderExtractor**: Random untrained encoder
    - **WaveletScatteringExtractor**: Wavelet transform (no training)

Key Features
------------

1. **Unified Interface**: All extractors expose `extract_embeddings()` method
2. **Automatic Caching**: Integrates with CacheManager for persistent storage
3. **Batched Processing**: Efficient GPU utilization with DataLoader
4. **Centralized Proxy Support**: Downloads models through utils.proxy_config on HPC
5. **Local Model Storage**: Save/load models from local directories
6. **Token Selection**: Choose specific tokens (CLS, all, k-last layers)
7. **Flexible Configuration**: Device selection, batch size, workers

Usage Examples
--------------

Basic usage with automatic caching:

    >>> from embedding import ModelRegistry, CacheManager
    >>>
    >>> cache = CacheManager(cache_dir=".cache")
    >>> extractor = ModelRegistry.create(
    ...     "dinov2-base",
    ...     device="cuda",
    ...     batch_size=32,
    ...     cache_manager=cache
    ... )
    >>>
    >>> image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    >>> embeddings = extractor.extract_embeddings(
    ...     images=image_paths,
    ...     use_cache=True
    ... )
    >>> embeddings.shape)  # (3, 768

Using different models:

    >>> # CLIP for vision-language tasks
    >>> clip = ModelRegistry.create("clip-vit-b-32", cache_manager=cache)
    >>> clip_emb = clip.extract_embeddings(image_paths)
    >>>
    >>> # InceptionV3 for FID calculations
    >>> inception = ModelRegistry.create("inception_v3", cache_manager=cache)
    >>> inception_emb = inception.extract_embeddings(image_paths)
    >>>
    >>> # Untrained random features for baselines
    >>> random = ModelRegistry.create("sparse_random_projection",
    ...                               cache_manager=cache,
    ...                               output_dim=512)
    >>> random_emb = random.extract_embeddings(image_paths)

Advanced: Custom token selection (for ViT models):

    >>> # Extract CLS token from last 4 layers
    >>> dinov2 = ModelRegistry.create("dinov2-large", cache_manager=cache)
    >>> emb = dinov2.extract_embeddings(
    ...     images=image_paths,
    ...     return_which_tokens="k_last_cls",
    ...     k_last=4
    ... )

Using proxy for model downloads:

    >>> extractor = ModelRegistry.create(
    ...     "dinov2-base",
    ...     proxy="http://proxy.example.com:3128",
    ...     model_dir="/local/models/"  # Save downloaded models
    ... )

Model Naming Convention
-----------------------

Models follow the pattern: `<family>-<variant>`

Examples:
    - `inception_v3` (no variant)
    - `dinov2-small`, `dinov2-base`, `dinov2-large`, `dinov2-giant`
    - `clip-vit-b-32`, `clip-vit-l-14`
    - `siglip-so400m`
    - `mae-base`, `mae-large`

See Also
--------

- CacheManager: Persistent caching system for embeddings
- ImageDataset: Dataset wrapper for loading images
- ModelRegistry: Factory pattern documentation
"""

from abc import ABC, abstractmethod
import logging
import warnings
from typing import Dict, List, Optional, Type, Union, Any
import os
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from data_loading.dataset import to_image_dataset
from embedding.cache_manager import CacheManager
from utils.proxy_config import hpc_proxy

try:
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        CLIPVisionModel,
        CLIPImageProcessor,
        SiglipVisionModel,
        SiglipImageProcessor,
        ViTMAEModel,
    )
except ImportError as exc:
    AutoImageProcessor = None
    AutoModel = None
    CLIPVisionModel = None
    CLIPImageProcessor = None
    SiglipVisionModel = None
    SiglipImageProcessor = None
    ViTMAEModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _require_transformers() -> None:
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "Transformer-backed embedding extractors require the optional dependency "
            "'transformers' to be installed."
        ) from _TRANSFORMERS_IMPORT_ERROR


class ModelRegistry:
    """Factory pattern registry for creating embedding extractors.

    Provides a centralized registry for discovering and instantiating embedding
    extractor classes. Extractors self-register using the `@ModelRegistry.register()`
    decorator, enabling automatic discovery and flexible creation.

    The registry supports a hierarchical naming scheme with model families and
    variants (e.g., "dinov2-base", "clip-vit-l-14").

    Attributes:
        _REGISTRY: Class-level dict mapping model family names to extractor classes.

    Examples:
        Register a custom extractor:

        >>> @ModelRegistry.register("my_model")
        ... class MyExtractor(EmbeddingExtractor):
        ...     def __init__(self, variant="default", **kwargs):
        ...         super().__init__(model_name=f"my_model-{variant}", **kwargs)
        ...     # ... implement abstract methods

        Create extractors:

        >>> # Built-in models
        >>> dinov2 = ModelRegistry.create("dinov2-base", device="cuda")
        >>> clip = ModelRegistry.create("clip-vit-b-32", batch_size=64)
        >>>
        >>> # Custom model
        >>> my_model = ModelRegistry.create("my_model-v2", custom_param=True)

        List available models:

        >>> available = ModelRegistry.list_models()
        >>> available
        ['inception_v3', 'dinov2', 'clip', 'siglip', ...]
    """

    _REGISTRY: Dict[str, Type["EmbeddingExtractor"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an extractor class with the registry.

        Args:
            name: Family name for the model (e.g., "dinov2", "clip").
                Will be used as the prefix in model identifiers.

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Examples:
            >>> @ModelRegistry.register("my_encoder")
            ... class MyEncoder(EmbeddingExtractor):
            ...     pass
        """
        def decorator(model_cls):
            cls._REGISTRY[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, model_identifier: str, **kwargs: Any) -> "EmbeddingExtractor":
        """Create an embedding extractor from a model identifier.

        Parses the model identifier into family and variant, looks up the
        appropriate extractor class, and instantiates it with provided kwargs.

        Args:
            model_identifier: Model specification in format "<family>-<variant>"
                or just "<family>" for default variant.
                Examples: "dinov2-base", "clip-vit-l-14", "inception_v3"
            **kwargs: Additional arguments passed to the extractor constructor.
                Common options:
                - device: str | torch.device
                - batch_size: int
                - num_workers: int
                - proxy: str
                - model_dir: str
                - cache_manager: CacheManager

        Returns:
            Instance of the appropriate EmbeddingExtractor subclass.

        Raises:
            ValueError: If the model family is not registered.

        Examples:
            >>> # Create with defaults
            >>> extractor = ModelRegistry.create("dinov2-base")
            >>>
            >>> # Create with custom configuration
            >>> extractor = ModelRegistry.create(
            ...     "clip-vit-b-32",
            ...     device="cuda:1",
            ...     batch_size=128,
            ...     cache_manager=my_cache
            ... )
        """
        # Handle cases without variants (e.g. "wavelet_scattering")
        if "-" in model_identifier:
            family, variant = model_identifier.split("-", 1)
        else:
            family, variant = model_identifier, "default"

        if family not in cls._REGISTRY:
             raise ValueError(f"Unknown model family '{family}'. Available: {list(cls._REGISTRY.keys())}")

        return cls._REGISTRY[family](variant=variant, **kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model families.

        Returns:
            List of registered model family names.

        Examples:
            >>> models = ModelRegistry.list_models()
            >>> models
            ['inception_v3', 'dinov2', 'clip', 'siglip', ...]
        """
        return list(cls._REGISTRY.keys())


def collate_fn(batch: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Custom collate function for DataLoader that handles corrupt images.

    Filters out None values (from images that failed to load) before stacking.
    This prevents batch processing from crashing due to individual corrupt files.

    Args:
        batch: List of tensors or None values from the DataLoader.

    Returns:
        Stacked tensor of valid images, or None if all images were corrupt.

    Examples:
        >>> # Used automatically by DataLoader
        >>> loader = DataLoader(dataset, collate_fn=collate_fn)
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.stack(batch)


class EmbeddingExtractor(ABC):
    """Abstract base class for embedding extractors.

    Defines the interface and common functionality for all embedding extractors.
    Subclasses must implement `load_model()` and `get_embedding_dim()` to define
    specific model architectures and dimensions.

    This class handles:
    - Model loading with proxy support
    - Batched inference using DataLoader
    - Integration with CacheManager
    - Device management (CPU/GPU)
    - Local model storage/retrieval

    Attributes:
        model_name: Full model identifier (e.g., "dinov2-base").
        batch_size: Number of images to process per batch.
        num_workers: Number of worker processes for data loading.
        proxy: HTTP proxy URL for model downloads (optional).
        model_dir: Local directory for storing/loading models (optional).
        cache_manager: CacheManager instance for embedding persistence (optional).
        device: Torch device for computation (cuda or cpu).
        model: The loaded neural network model (None until load_model() called).
        processor: HuggingFace processor for image preprocessing (model-specific).
        transform: Torchvision transform for image preprocessing (model-specific).

    Examples:
        Subclass implementation:

        >>> class MyExtractor(EmbeddingExtractor):
        ...     def __init__(self, variant="default", **kwargs):
        ...         super().__init__(model_name=f"mymodel-{variant}", **kwargs)
        ...
        ...     def load_model(self):
        ...         self.model = MyModel.from_pretrained(self.model_name)
        ...         self.model.to(self.device)
        ...         self.model.eval()
        ...         self.transform = transforms.Compose([...])
        ...
        ...     def get_embedding_dim(self) -> int:
        ...         return 768

        Usage:

        >>> extractor = MyExtractor(
        ...     variant="large",
        ...     device="cuda",
        ...     batch_size=32,
        ...     cache_manager=cache
        ... )
        >>> embeddings = extractor.extract_embeddings(image_paths)
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        proxy: Optional[str] = None,
        model_dir: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        """Initialize the embedding extractor.

        Args:
            model_name: Full model identifier (e.g., "dinov2-base", "clip-vit-b-32").
            device: Computation device. If None, auto-selects cuda if available.
            batch_size: Number of images per batch during inference.
            num_workers: Number of DataLoader worker processes.
            proxy: HTTP proxy URL for downloading models (e.g., "http://proxy:3128").
            model_dir: Directory to save/load models locally. If provided, models
                are cached to this directory after first download.
            cache_manager: CacheManager instance for persistent embedding storage.
                If None, embeddings won't be cached across runs.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.proxy = proxy
        self.model_dir = Path(model_dir) if model_dir else None

        # Use provided CacheManager or create a local one if None is passed but we want caching
        self.cache_manager = cache_manager
        self._proxy_context: Optional[Any] = None

        self.device = self._resolve_device(device)

        self.model: Optional[nn.Module] = None
        self.processor: Any = None
        self.transform: Any = None

    def _resolve_device(self, requested: Optional[Union[str, torch.device]]) -> torch.device:
        if requested is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(requested)
        if device.type != "cuda":
            return device
        if torch.cuda.is_available():
            return device

        message = (
            f"CUDA was requested for '{self.model_name}' but is unavailable in this environment. "
            "Falling back to CPU; embedding extraction may take a long time."
        )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return torch.device("cpu")

    def _setup_proxy_for_download(self) -> None:
        self._proxy_context = hpc_proxy(self.proxy)
        self._proxy_context.__enter__()

    def _cleanup_proxy_after_download(self) -> None:
        if self._proxy_context is None:
            return
        self._proxy_context.__exit__(None, None, None)
        self._proxy_context = None

    def _download_model_with_proxy(self, download_func, *args, **kwargs) -> Any:
        """Wrapper for model downloads with centralized proxy configuration.

        Uses centralized proxy configuration from utils.proxy_config for safe
        proxy management during model downloads on HPC cluster.

        Args:
            download_func: Function to call for downloading the model
            *args: Positional arguments to pass to download_func
            **kwargs: Keyword arguments to pass to download_func

        Returns:
            Downloaded model object
        """
        self._setup_proxy_for_download()
        try:
            if self.model_dir:
                model_path = self.model_dir / self.model_name.replace("/", "--")
                if model_path.exists():
                    logger.info("Loading model from local directory: %s", model_path)
                    kwargs["local_files_only"] = True
                    return download_func(str(model_path), *args[1:], **kwargs)
                logger.info("Model not found in %s. Will try to download.", model_path)

            return download_func(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model assets for '{self.model_name}'. "
                f"proxy={self.proxy!r}, model_dir={str(self.model_dir) if self.model_dir else None}."
            ) from exc
        finally:
            self._cleanup_proxy_after_download()

    def _save_model_locally(self, model, processor) -> None:
        if self.model_dir:
            model_path = self.model_dir / self.model_name.replace("/", "--")
            model_path.mkdir(parents=True, exist_ok=True)
            if not hasattr(model, "save_pretrained"):
                raise TypeError(
                    f"Model '{self.model_name}' does not support local save_pretrained() persistence."
                )
            model.save_pretrained(str(model_path))
            if processor and hasattr(processor, "save_pretrained"):
                processor.save_pretrained(str(model_path))
            logger.info("Model saved locally to: %s", model_path)

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and initialize preprocessors.

        Must be implemented by subclasses to load the specific model architecture,
        move it to the appropriate device, set it to evaluation mode, and initialize
        either self.transform (for torchvision models) or self.processor (for
        HuggingFace models).

        Side Effects:
            - Sets self.model to the loaded nn.Module
            - Sets self.transform or self.processor for image preprocessing
            - Model is moved to self.device and set to eval() mode

        Examples:
            >>> def load_model(self):
            ...     self.model = models.inception_v3(pretrained=True)
            ...     self.model.to(self.device)
            ...     self.model.eval()
            ...     self.transform = transforms.Compose([
            ...         transforms.Resize(299),
            ...         transforms.ToTensor(),
            ...     ])
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings produced by this extractor.

        Must be implemented by subclasses to return the embedding dimension.
        This value depends on the model architecture and any token selection
        parameters.

        Returns:
            Integer dimension of the embedding vectors.

        Examples:
            >>> def get_embedding_dim(self) -> int:
            ...     return 2048  # InceptionV3 feature dimension
        """
        pass

    def _get_dataloader(self, images: List[str]) -> DataLoader:
        if not (self.transform or self.processor):
            raise RuntimeError("Processor or Transform not initialized. Call load_model() first.")

        dataset = to_image_dataset(images, transform=self.transform, processor=self.processor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=(self.device.type == "cuda"),
        )

    def extract_embeddings(
        self, 
        images: List[str], 
        return_which_tokens: str = "cls",
        k_last: int = -1,
        use_cache: bool = True,
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        valid_token_modes = {"cls", "all_last", "k_last_cls", "k_last_all"}
        if return_which_tokens not in valid_token_modes:
            raise ValueError(f"Invalid return_which_tokens '{return_which_tokens}'")
        
        # Define cache parameters
        cache_params = {
            "return_which_tokens": return_which_tokens,
            "k_last": k_last,
            # **kwargs
        }

        # 1. Check Unified Cache
        if use_cache and self.cache_manager:
            cached_data = self.cache_manager.get(
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                **cache_params
            )
            if cached_data is not None:
                # logger.debug(f"  ✓ Cache HIT: {self.model_name} ({len(cached_data)} samples)")
                return cached_data
            else:
                logger.info("Cache miss for %s; computing embeddings.", self.model_name)

        # 2. Compute if not cached
        if self.model is None:
            self.load_model()

        all_embeddings = []
        dataloader = self._get_dataloader(images)
        output_hidden = "k_last" in return_which_tokens

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting embeddings with {self.model_name}", unit="batch", leave=False):
                if batch is None: 
                    continue
                batch = batch.to(self.device)
                
                if output_hidden:
                    outputs = self.model(pixel_values=batch, output_hidden_states=True)
                    if return_which_tokens == "k_last_cls":
                        selected_states = outputs.hidden_states[-k_last:]
                        cls_embeddings = [state[:, 0, :] for state in selected_states]
                        emb = torch.cat(cls_embeddings, dim=-1)
                    elif return_which_tokens == "k_last_all":
                        selected_states = outputs.hidden_states[-k_last:]
                        emb = torch.cat(selected_states, dim=-1)
                    else:
                        raise ValueError(f"Invalid return_which_tokens '{return_which_tokens}'")
                else:
                    outputs = self.model(pixel_values=batch)
                    if return_which_tokens == "cls":
                        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                            emb = outputs.pooler_output
                        else:
                            emb = outputs.last_hidden_state[:, 0, :]
                    elif return_which_tokens == "all_last":
                        emb = outputs.last_hidden_state
                    else:
                        raise ValueError(f"Invalid return_which_tokens '{return_which_tokens}'")
                all_embeddings.append(emb.cpu().numpy())

        if not all_embeddings:
            result = np.array([])
        else:
            result = np.concatenate(all_embeddings, axis=0)
            
        # 3. Save to Unified Cache
        if use_cache and self.cache_manager:
            self.cache_manager.put(
                embeddings=result,
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                **cache_params
            )
            
        return result


@ModelRegistry.register("inception_v3")
class InceptionExtractor(EmbeddingExtractor):
    """InceptionV3 feature extractor for FID-compatible embeddings.

    Extracts 2048-dimensional features from the penultimate layer of
    InceptionV3, commonly used for Frechet Inception Distance (FID) computation.

    Parameters
    ----------
    variant : str, optional
        Variant identifier (unused, InceptionV3 has only one variant).
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    model : nn.Module
        InceptionV3 model with final FC layer replaced by Identity.

    Notes
    -----
    - Output dimension: 2048
    - Uses ImageNet normalization
    - Input size: 299x299 (auto-resized)
    """

    def __init__(self, variant: str = "default", **kwargs):
        super().__init__(model_name="inception_v3", **kwargs)

    def load_model(self) -> None:
        logger.info("Loading InceptionV3.")
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def _load_inception(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise ValueError(
                    "InceptionExtractor does not support loading from model_dir; "
                    "torchvision weights are managed by torch's own cache."
                )
            return models.inception_v3(pretrained=True, transform_input=False)

        self.model = self._download_model_with_proxy(_load_inception)
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        self.processor = None

    def get_embedding_dim(self) -> int:
        return 2048

    def extract_embeddings(
        self, 
        images: List[str], 
        use_cache: bool = True,
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:

        cache_params = {**kwargs}
        # cache_params = {} # Inception does not have token selection params

        if use_cache and self.cache_manager:
            cached_data = self.cache_manager.get(
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                **cache_params
            )
            if cached_data is not None:
                # logger.debug(f"  ✓ Cache HIT: {self.model_name} ({len(cached_data)} samples)")
                return cached_data
            else:
                logger.info(
                    "Cache miss for %s (dataset=%s, base_path=%s, use_cache=%s); computing embeddings.",
                    self.model_name,
                    dataset_name,
                    base_path,
                    use_cache,
                )
                logger.debug("Cache params for %s: %s", self.model_name, cache_params)

        if self.model is None:
            self.load_model()

        all_embeddings = []
        dataloader = self._get_dataloader(images)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting embeddings with {self.model_name}", unit="batch", leave=False):
                if batch is None: continue
                batch = batch.to(self.device)
                emb = self.model(batch)
                all_embeddings.append(emb.cpu().numpy())

        if not all_embeddings:
            result = np.array([])
        else:
            result = np.concatenate(all_embeddings, axis=0)

        if use_cache and self.cache_manager:
            logger.debug("Saving embeddings to cache for %s with params: %s", self.model_name, cache_params)
            self.cache_manager.put(
                embeddings=result,
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                **cache_params
            )
            
        return result


@ModelRegistry.register("mae")
class MAEExtractor(EmbeddingExtractor):
    """Masked Autoencoder (MAE) feature extractor.

    Extracts features from the ViT-MAE model encoder. MAE is trained via
    masked image modeling, learning rich visual representations.

    Parameters
    ----------
    variant : str, optional
        Model size variant. One of "base", "large", "huge". Default: "base".
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    VARIANTS : dict
        Mapping of variant names to HuggingFace model paths.
    hf_model_path : str
        The resolved HuggingFace model path for the selected variant.

    Notes
    -----
    - Output dimensions: base=768, large=1024, huge=1280
    - Uses HuggingFace transformers library
    - Supports CLS token or all-tokens extraction
    """

    VARIANTS = {
        "base": "facebook/vit-mae-base",
        "large": "facebook/vit-mae-large",
        "huge": "facebook/vit-mae-huge",
    }

    def __init__(self, variant: str = "base", **kwargs):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown MAE variant '{variant}'. Options: {list(self.VARIANTS.keys())}")
        super().__init__(f"mae-{variant}", **kwargs)
        self.variant = variant
        self.hf_model_path = self.VARIANTS[variant]

    def load_model(self) -> None:
        _require_transformers()
        logger.info("Loading MAE %s.", self.variant)
        self.processor = self._download_model_with_proxy(AutoImageProcessor.from_pretrained, self.hf_model_path)
        self.model = self._download_model_with_proxy(ViTMAEModel.from_pretrained, self.hf_model_path)
        if self.model_dir: self._save_model_locally(self.model, self.processor)
        self.model.to(self.device)
        self.model.eval()
    def get_embedding_dim(self) -> int:
        return {"base": 768, "large": 1024, "huge": 1280}[self.variant]

@ModelRegistry.register("dino")
@ModelRegistry.register("dinov2")
class DINOv2Extractor(EmbeddingExtractor):
    """DINOv2 self-supervised Vision Transformer feature extractor.

    Extracts rich semantic features from Meta's DINOv2 model, trained with
    self-distillation on large-scale unlabeled image data.

    Parameters
    ----------
    variant : str, optional
        Model size variant. One of "small", "base", "large", "giant".
        Default: "base".
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    VARIANTS : dict
        Mapping of variant names to HuggingFace model paths.

    Notes
    -----
    - Output dimensions: small=384, base=768, large=1024, giant=1536
    - Excellent for semantic similarity and clustering tasks
    - Recommended default for embedding-based partitioning
    """

    VARIANTS = {"small": "facebook/dinov2-small", "base": "facebook/dinov2-base", "large": "facebook/dinov2-large", "giant": "facebook/dinov2-giant"}

    def __init__(self, variant: str = "base", **kwargs):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown DINOv2 variant '{variant}'. Options: {list(self.VARIANTS.keys())}")
        super().__init__(f"dinov2-{variant}", **kwargs)
        self.variant = variant
        self.hf_model_path = self.VARIANTS[variant]
    def load_model(self) -> None:
        _require_transformers()
        logger.info("Loading DINOv2 %s.", self.variant)
        self.processor = self._download_model_with_proxy(AutoImageProcessor.from_pretrained, self.hf_model_path)
        self.model = self._download_model_with_proxy(AutoModel.from_pretrained, self.hf_model_path)
        if self.model_dir: self._save_model_locally(self.model, self.processor)
        self.model.to(self.device)
        self.model.eval()
    def get_embedding_dim(self) -> int:
        return {"small": 384, "base": 768, "large": 1024, "giant": 1536}[self.variant]

@ModelRegistry.register("clip")
class CLIPExtractor(EmbeddingExtractor):
    """CLIP (Contrastive Language-Image Pre-training) vision encoder.

    Extracts visual features from OpenAI's CLIP model, trained to align
    image and text representations in a shared embedding space.

    Parameters
    ----------
    variant : str, optional
        Model variant. One of "base-patch32", "base-patch16", "large-patch14".
        Default: "base-patch16".
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    VARIANTS : dict
        Mapping of variant names to HuggingFace model paths.

    Notes
    -----
    - Output dimensions: base variants=768, large=1024
    - Good for cross-modal tasks and semantic understanding
    - Vision-only encoder (text encoder not used here)
    """

    VARIANTS = {
        "base-patch32": "openai/clip-vit-base-patch32",
        "base-patch16": "openai/clip-vit-base-patch16",
        "large-patch14": "openai/clip-vit-large-patch14",
    }

    def __init__(self, variant: str = "base-patch16", **kwargs):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown CLIP variant '{variant}'. Options: {list(self.VARIANTS.keys())}")
        super().__init__(f"clip-{variant}", **kwargs)
        self.variant = variant
        self.hf_model_path = self.VARIANTS[variant]
    def load_model(self) -> None:
        _require_transformers()
        logger.info("Loading CLIP %s.", self.variant)
        self.processor = self._download_model_with_proxy(CLIPImageProcessor.from_pretrained, self.hf_model_path)
        self.model = self._download_model_with_proxy(CLIPVisionModel.from_pretrained, self.hf_model_path)
        if self.model_dir: self._save_model_locally(self.model, self.processor)
        self.model.to(self.device)
        self.model.eval()
    def get_embedding_dim(self) -> int:
        return {"base-patch32": 768, "base-patch16": 768, "large-patch14": 1024}[self.variant]


@ModelRegistry.register("siglip")
class SigLIPExtractor(EmbeddingExtractor):
    """SigLIP (Sigmoid Loss for Language-Image Pre-training) vision encoder.

    Extracts visual features from Google's SigLIP model, an improved CLIP
    variant using sigmoid loss instead of softmax for better efficiency.

    Parameters
    ----------
    variant : str, optional
        Model variant. One of "base-patch16-224", "base-patch16-384",
        "large-patch16-256". Default: "base-patch16-224".
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    VARIANTS : dict
        Mapping of variant names to HuggingFace model paths.

    Notes
    -----
    - Output dimensions: base=768, large=1024
    - Often outperforms CLIP on retrieval tasks
    - Different input resolutions available (224, 256, 384)
    """

    VARIANTS = {
        "base-patch16-224": "google/siglip-base-patch16-224",
        "base-patch16-384": "google/siglip-base-patch16-384",
        "large-patch16-256": "google/siglip-large-patch16-256",
    }

    def __init__(self, variant: str = "base-patch16-224", **kwargs):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown SigLIP variant '{variant}'. Options: {list(self.VARIANTS.keys())}")
        super().__init__(f"siglip-{variant}", **kwargs)
        self.variant = variant
        self.hf_model_path = self.VARIANTS[variant]
    def load_model(self) -> None:
        _require_transformers()
        logger.info("Loading SigLIP %s.", self.variant)
        self.processor = self._download_model_with_proxy(SiglipImageProcessor.from_pretrained, self.hf_model_path)
        self.model = self._download_model_with_proxy(SiglipVisionModel.from_pretrained, self.hf_model_path)
        if self.model_dir: self._save_model_locally(self.model, self.processor)
        self.model.to(self.device)
        self.model.eval()
    def get_embedding_dim(self) -> int:
        return 1024 if "large" in self.variant else 768


@ModelRegistry.register("raw_pixel_resizer")
class RawPixelResizerExtractor(EmbeddingExtractor):
    """Raw pixel feature extractor (no learned features).

    Resizes images and flattens them into vectors. Provides a simple baseline
    that uses no learned representations - just raw pixel values.

    Parameters
    ----------
    variant : str, optional
        Target image size (e.g., "64" for 64x64). Default: "64".
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    target_size : int
        The target height and width for resizing.

    Notes
    -----
    - Output dimension: target_size * target_size * 3 (RGB channels)
    - No neural network inference required
    - Useful as a baseline for comparing learned features
    - Example: 64x64 produces 12,288-dimensional vectors
    """

    def __init__(self, variant: str = "64", **kwargs):
        super().__init__(model_name=f"raw_pixel_resizer-{variant}", **kwargs)
        try:
            self.target_size = int(variant)
        except ValueError:
            raise ValueError(f"Invalid target size '{variant}' for RawPixelResizerExtractor.")

    def load_model(self) -> None:
        logger.info(
            "Using Raw Pixel Resizer with target size %dx%d.",
            self.target_size,
            self.target_size,
        )
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
        ])
        self.model = None

    def get_embedding_dim(self) -> int:
        return self.target_size * self.target_size * 3
    
    def extract_embeddings(
        self,
        images: List[str],
        return_which_tokens: str = "cls",
        k_last: int = 5,
        use_cache: bool = True,
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:

        # 1. Check Cache
        if use_cache and self.cache_manager:
            cached_data = self.cache_manager.get(
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                target_size=self.target_size
            )
            if cached_data is not None:
                return cached_data

        if self.model is None:
            self.load_model()

        all_embeddings = []

        dataloader = self._get_dataloader(images)
        for batch in dataloader:
            if batch is None: continue
            batch = batch.to(self.device)
            batch_size = batch.size(0)
            emb = batch.view(batch_size, -1)
            all_embeddings.append(emb.cpu().numpy())

        if not all_embeddings:
            result = np.array([])
        else:
            result = np.concatenate(all_embeddings, axis=0)

        # 2. Save Cache
        if use_cache and self.cache_manager:
            self.cache_manager.put(
                embeddings=result,
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                target_size=self.target_size
            )

        return result


@ModelRegistry.register("sparse_random_projection_resizer")
class SparseRandomProjectionResizerExtractor(EmbeddingExtractor):
    """Sparse random projection feature extractor (no learned features).

    Resizes images, flattens them, and projects to lower dimensions using
    sparse random projections. Based on Johnson-Lindenstrauss lemma for
    dimensionality reduction while approximately preserving distances.

    Parameters
    ----------
    variant : str, optional
        Format: "<size>-<target_dim>" (e.g., "64-1000" for 64x64 images
        projected to 1000 dimensions). Default: "64-1000".
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    target_size : int
        The target height and width for resizing.
    target_dim : int
        The output embedding dimension after projection.
    transformer : SparseRandomProjection
        The sklearn sparse random projection transformer.

    Notes
    -----
    - Uses sklearn's SparseRandomProjection with random_state=42
    - Projection matrix is fit on the first batch
    - Useful as a baseline or when computational resources are limited
    - Approximately preserves pairwise distances (JL lemma)
    """

    def __init__(self, variant: str = "64-1000", **kwargs):
        if variant == "default":
            variant = "64-1000"
        try:
            size_str, target_dim_str = variant.split("-")
            self.target_size = int(size_str)
            self.target_dim = int(target_dim_str)
        except ValueError:
            raise ValueError(f"Invalid variant '{variant}' for RandomProjectionResizerExtractor. Use format '<size>-<target_dim>'.")
        super().__init__(model_name=f"sparse_random_projection_resizer-{variant}", **kwargs)
        self.transformer: Optional[Any] = None

    def load_model(self) -> None:
        logger.info(
            "Using Sparse Random Projection Resizer with target size %dx%d and target dim %d.",
            self.target_size,
            self.target_size,
            self.target_dim,
        )
        from sklearn.random_projection import SparseRandomProjection
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
        ])
        self.transformer = SparseRandomProjection(
            n_components=self.target_dim,
            random_state=42,
            dense_output=True
        )
        self.model = None

    def get_embedding_dim(self) -> int:
        return self.target_dim
    
    def extract_embeddings(
        self,
        images: List[str],
        use_cache: bool = True,
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:

        # 1. Check Cache
        if use_cache and self.cache_manager:
            cached_data = self.cache_manager.get(
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                target_size=self.target_size,
                target_dim=self.target_dim
            )
            if cached_data is not None:
                return cached_data

        if self.transformer is None:
            self.load_model()
            
        all_embeddings = []
        dataloader = self._get_dataloader(images)
        
        # Fit transformer on first batch if not already fitted
        fitted = False
        
        for batch in dataloader:
            if batch is None: continue
            batch = batch.to(self.device)
            batch_size = batch.size(0)
            flat_batch = batch.view(batch_size, -1).cpu().numpy()
            
            if not fitted:
                try:
                    self.transformer.fit(flat_batch)
                    fitted = True
                except (AttributeError, ValueError) as e:
                    raise RuntimeError(f"Failed to fit random projection transformer: {e}")
            
            try:
                projected = self.transformer.transform(flat_batch)
                all_embeddings.append(projected)
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                raise RuntimeError(f"Failed to transform batch with random projection: {e}")

        if not all_embeddings:
            result = np.array([])
        else:
            result = np.concatenate(all_embeddings, axis=0)

        # 2. Save Cache
        if use_cache and self.cache_manager:
            self.cache_manager.put(
                embeddings=result,
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                target_size=self.target_size,
                target_dim=self.target_dim
            )

        return result


@ModelRegistry.register("UAE")
class UntrainedAutoEncoderExtractor(EmbeddingExtractor):
    """Untrained AutoEncoder feature extractor (random initialization).

    Extracts features using a randomly initialized convolutional autoencoder
    encoder. The network is never trained - it uses random weights to produce
    deterministic but arbitrary features.

    Parameters
    ----------
    variant : str, optional
        Must be "default". Only one variant is supported.
    input_size : int, optional
        Input image size (square). Default: 64.
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    input_size : int
        The input image size used.
    autoencoder : nn.Module
        The randomly initialized convolutional encoder.

    Notes
    -----
    - Output dimension: 16 * (input_size // 8)^2
    - Example: input_size=64 produces 1024-dimensional vectors
    - Useful as a baseline to compare against trained embeddings
    - Architecture: 3 conv layers with max pooling
    """

    def __init__(self, variant: str = "default", input_size: int = 64, **kwargs):
        if variant != "default":
            raise ValueError(f"Unknown variant '{variant}' for UntrainedAutoEncoderExtractor. Only 'default' is supported.")
        self.input_size = input_size
        super().__init__(model_name="untrained_autoencoder", **kwargs)
        self.autoencoder: Optional[nn.Module] = None

    def load_model(self) -> None:
        logger.info("Using Untrained AutoEncoder with input size %dx%d.", self.input_size, self.input_size)
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.autoencoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
        ).to(self.device)
        
        # Also set self.model for compatibility with base class
        self.model = self.autoencoder

    def get_embedding_dim(self) -> int:
        return 16 * (self.input_size // 8) * (self.input_size // 8)
    
    def extract_embeddings(
        self,
        images: List[str],
        return_which_tokens: str = "cls",
        k_last: int = 5,
        use_cache: bool = True,
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:

        # 1. Check Cache
        if use_cache and self.cache_manager:
            cached_data = self.cache_manager.get(
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                input_size=self.input_size
            )
            if cached_data is not None:
                return cached_data

        if self.autoencoder is None:
            self.load_model()
            
        all_embeddings = []

        dataloader = self._get_dataloader(images)
        with torch.no_grad():
            for batch in dataloader:
                if batch is None: 
                    continue
                batch = batch.to(self.device)
                encoded = self.autoencoder(batch)
                emb = encoded.view(encoded.size(0), -1)
                all_embeddings.append(emb.cpu().numpy())
                
        if not all_embeddings:
            result = np.array([])
        else:
            result = np.concatenate(all_embeddings, axis=0)

        # 2. Save Cache
        if use_cache and self.cache_manager:
            self.cache_manager.put(
                embeddings=result,
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                input_size=self.input_size
            )

        return result


@ModelRegistry.register("wavelet_scattering")
class WaveletScatteringExtractor(EmbeddingExtractor):
    """Wavelet scattering transform feature extractor (no learned features).

    Computes wavelet scattering coefficients using the kymatio library.
    Scattering transforms provide stable, invariant representations without
    any training, based on cascaded wavelet convolutions.

    Parameters
    ----------
    variant : str, optional
        Format: "<size>-<J>" where size is input resolution and J is the
        number of scattering scales. Default: "224-3".
    **kwargs
        Passed to :class:`EmbeddingExtractor` (device, batch_size, etc.).

    Attributes
    ----------
    input_size : int
        The input image size used.
    J : int
        Number of wavelet scales (larger J = more invariance).
    scattering : kymatio.torch.Scattering2D
        The scattering transform module.

    Notes
    -----
    - Requires kymatio package: ``pip install kymatio``
    - Output dimension depends on input_size and J
    - Provides translation and deformation invariance
    - No training required - purely mathematical transform

    References
    ----------
    Mallat, S. "Group Invariant Scattering" (2012)
    """

    def __init__(self, variant: str = "224-3", **kwargs):
        if variant == "default":
            variant = "224-3"
        try:
            size_str, J_str = variant.split("-")
            self.input_size = int(size_str)
            self.J = int(J_str)
        except ValueError:
            raise ValueError(f"Invalid variant '{variant}' for WaveletScatteringExtractor. Use format '<size>-<J>'.")

        super().__init__(model_name=f"wavelet_scattering-{variant}", **kwargs)
        self.scattering: Optional[Any] = None

    def load_model(self) -> None:
        try:
            from kymatio.torch import Scattering2D
        except ImportError:
            raise ImportError("kymatio is required for WaveletScatteringExtractor. Please install it via 'pip install kymatio'.")
        
        logger.info(
            "Using Wavelet Scattering with input size %dx%d and J=%d.",
            self.input_size,
            self.input_size,
            self.J,
        )

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Handle potential downloads for kymatio filters with an explicit proxy when requested.
        self._setup_proxy_for_download()
        try:
            self.scattering = Scattering2D(J=self.J, shape=(self.input_size, self.input_size)).to(self.device)
        finally:
            self._cleanup_proxy_after_download()
            
        self.model = None

    def get_embedding_dim(self) -> int:
        if self.scattering is None:
            self.load_model()
        dummy_input = torch.zeros(1, 3, self.input_size, self.input_size).to(self.device)
        with torch.no_grad():
            output = self.scattering(dummy_input)
        return output.numel()
    
    def extract_embeddings(
        self,
        images: List[str],
        return_which_tokens: str = "cls",
        k_last: int = 5,
        use_cache: bool = True,
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:

        # 1. Check Cache
        if use_cache and self.cache_manager:
            cached_data = self.cache_manager.get(
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                input_size=self.input_size,
                J=self.J
            )
            if cached_data is not None:
                return cached_data

        if self.scattering is None:
            self.load_model()
            
        all_embeddings = []

        dataloader = self._get_dataloader(images)
        with torch.no_grad():
            for batch in dataloader:
                if batch is None: 
                    continue
                batch = batch.to(self.device)
                scatter_coeffs = self.scattering(batch)
                emb = scatter_coeffs.view(scatter_coeffs.size(0), -1)
                all_embeddings.append(emb.cpu().numpy())
                
        if not all_embeddings:
            result = np.array([])
        else:
            result = np.concatenate(all_embeddings, axis=0)

        # 2. Save Cache
        if use_cache and self.cache_manager:
            self.cache_manager.put(
                embeddings=result,
                model_name=self.model_name,
                image_paths=images,
                dataset_name=dataset_name,
                base_path=base_path,
                input_size=self.input_size,
                J=self.J
            )

        return result
