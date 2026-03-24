import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import zipfile
import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """Persistent caching system for embedding storage and retrieval.

    Provides intelligent caching of image embeddings to disk with support for
    two operating modes: batch mode (for ad-hoc computations) and dataset mode
    (for large-scale datasets with subset access). Includes in-memory caching
    for fast repeated access and supports relative paths for shared HPC environments.

    **Operating Modes**:

    1. **Batch Mode** (default):
       - Caches specific lists of images based on content hash
       - Cache key depends on exact image list
       - Use when processing arbitrary batches of images

    2. **Dataset Mode**:
       - Caches entire datasets indexed by image paths
       - Supports efficient subset retrieval
       - Cache key independent of specific queried subset
       - Use when working with named datasets (e.g., "imagenet_train")

    **Features**:
    - **Content-based hashing**: Cache keys derived from MD5/SHA256 hashes
    - **In-memory caching**: Loaded .npz files cached in memory for speed
    - **Relative paths**: Convert absolute paths to relative for portability
    - **Metadata tracking**: JSON metadata file tracks all cache entries
    - **Model-specific clearing**: Clear cache by model name
    - **Compression**: Numpy compressed format (.npz) for space efficiency

    Attributes:
        cache_dir: Directory for storing cache files.
        enabled: Whether caching is active. If False, all operations are no-ops.
        metadata: Dict tracking all cache entries (loaded from metadata.json).
        _memory_cache: In-memory dict of loaded cache data for fast access.

    Examples:
        Basic usage with embedding extractors:

        >>> cache = CacheManager(cache_dir=".cache")
        >>>
        >>> # Batch mode - specific images
        >>> embeddings = cache.get(
        ...     model_name="dinov2-base",
        ...     image_paths=["img1.jpg", "img2.jpg"]
        ... )
        >>>
        >>> if embeddings is None:
        ...     # Compute embeddings
        ...     embeddings = extractor.extract_embeddings(image_paths)
        ...     # Store for future use
        ...     cache.put(
        ...         embeddings=embeddings,
        ...         model_name="dinov2-base",
        ...         image_paths=image_paths
        ...     )

        Dataset mode for large-scale caching:

        >>> # Cache entire dataset
        >>> all_paths = ["img1.jpg", "img2.jpg", ..., "img10000.jpg"]
        >>> all_embeddings = extractor.extract_embeddings(all_paths)
        >>> cache.put(
        ...     embeddings=all_embeddings,
        ...     model_name="dinov2-base",
        ...     image_paths=all_paths,
        ...     dataset_name="imagenet_train",
        ...     base_path="/data/imagenet/"  # For relative paths
        ... )
        >>>
        >>> # Retrieve subset efficiently
        >>> subset_paths = ["img5.jpg", "img100.jpg", "img500.jpg"]
        >>> subset_embs = cache.get(
        ...     model_name="dinov2-base",
        ...     image_paths=subset_paths,
        ...     dataset_name="imagenet_train",
        ...     base_path="/data/imagenet/"
        ... )

        Cache management:

        >>> # Get cache statistics
        >>> info = cache.get_cache_info()
        >>> f"Cache size: {info['total_size_mb']:.2f} MB"
        >>> f"Entries: {info['num_entries']}"
        >>>
        >>> # Clear specific model's cache
        >>> cache.clear(model_name="dinov2-base")
        >>>
        >>> # Clear all cache
        >>> cache.clear()
    """

    def __init__(
        self,
        cache_dir: Union[str, Path, None] = None,
        enabled: bool = True,
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory path for storing cache files. Created if doesn't exist.
            enabled: Whether caching is active. If False, get() always returns None
                and put() does nothing.
        """
        resolved_cache_dir = cache_dir or os.environ.get("FDS_CACHE_DIR", ".embedding_cache")
        self.cache_dir = Path(resolved_cache_dir)
        self.enabled = enabled
        self._memory_cache = {}  # In-memory store for loaded .npz content

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._init_metadata()

    def _init_metadata(self) -> None:
        """Initialize or load cache metadata."""
        self.metadata_file = self.cache_dir / "metadata.json"

        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                try:
                    self.metadata = json.load(f)
                except json.JSONDecodeError:
                    self.metadata = {"version": "1.0", "entries": {}}
                    self._save_metadata()
        else:
            self.metadata = {"version": "1.0", "entries": {}}
            self._save_metadata()

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def _evict_cache_entry(self, cache_key: str) -> None:
        """Remove a corrupted cache entry from memory, metadata, and disk."""
        self._memory_cache.pop(cache_key, None)
        entry = self.metadata["entries"].pop(cache_key, None)
        cache_file = self.cache_dir / (entry["file"] if entry and entry.get("file") else f"{cache_key}.npz")
        if cache_file.exists():
            try:
                cache_file.unlink()
            except OSError as exc:
                logger.warning("Failed to remove cache file %s during eviction: %s", cache_file, exc)
        self._save_metadata()

    def _to_relative(self, paths: List[str], base_path: Optional[str]) -> List[str]:
        """Convert paths to relative if base_path is provided."""
        if not base_path:
            return [str(p) for p in paths]
        
        rel_paths = []
        for p in paths:
            if isinstance(p, str) and "://" in p:
                rel_paths.append(str(p))
                continue
            try:
                # Resolve paths to handle ../ etc, though usually inputs are clean
                rel = os.path.relpath(str(p), start=str(base_path))
                rel_paths.append(rel)
            except ValueError:
                # Fallback if paths are on different drives etc.
                rel_paths.append(str(p))
        return rel_paths

    def _compute_cache_key(
        self,
        model_name: str,
        image_paths: List[str],
        dataset_name: Optional[str] = None,
        pooling_strategy: Optional[str] = None,
        layers: Optional[List[int]] = None,
        **kwargs,
    ) -> str:
        """
        Compute cache key.
        If dataset_name is provided, uses a fixed key structure for that dataset
        to allow subset retrieval.
        """
        # Create a unique key for the configuration, allow for non-transformer models to not have transformer-related params
        # by only including non-None params.
        config_dict = {
            "model_name": model_name,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        if pooling_strategy is not None:
            config_dict["pooling_strategy"] = pooling_strategy
        if layers is not None:
            config_dict["layers"] = layers

        # Sort keys for consistency
        config_str = json.dumps(config_dict, sort_keys=True)
        # Print the config string for verification/debugging
        # logger.debug(f"Computing cache key with config: {config_str}")
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        if dataset_name:
            # Dataset Mode: Key depends on dataset name + config, NOT specific image paths
            return f"dataset_{dataset_name}_{model_name}_{config_hash}"
        else:
            # Batch Mode: Key depends on the specific list of images
            sorted_paths = sorted(image_paths)
            paths_str = json.dumps(sorted_paths)
            paths_hash = hashlib.sha256(paths_str.encode()).hexdigest()
            return f"batch_{model_name}_{config_hash}_{paths_hash[:16]}"

    def get(
        self,
        model_name: str,
        image_paths: List[str],
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        pooling_strategy: Optional[str] = None,
        layers: Optional[List[int]] = None,
        **kwargs,
    ) -> Optional[Union[np.ndarray, Dict[int, np.ndarray]]]:
        """Retrieve cached embeddings if available.

        Checks cache (memory then disk) for embeddings matching the specified
        configuration. Returns None if not found, triggering recomputation.

        Args:
            model_name: Model identifier (e.g., "dinov2-base").
            image_paths: List of image file paths to retrieve embeddings for.
            dataset_name: Optional dataset name for dataset mode. If provided,
                enables subset retrieval from larger cached datasets.
            base_path: Optional base directory for converting paths to relative.
                Useful for shared HPC environments where absolute paths vary.
            pooling_strategy: Optional pooling method used during extraction
                (e.g., "cls", "avg"). Part of cache key.
            layers: Optional list of layer indices used during extraction.
                Part of cache key.
            **kwargs: Additional configuration parameters that affect the cache
                key (e.g., return_which_tokens, k_last).

        Returns:
            - np.ndarray: Cached embeddings (shape: [N, D]) if found
            - Dict[int, np.ndarray]: Multi-layer embeddings if that's how they
              were stored
            - None: If no matching cache entry exists

        Examples:
            >>> # Batch mode
            >>> embs = cache.get(
            ...     model_name="clip-vit-b-32",
            ...     image_paths=["a.jpg", "b.jpg"]
            ... )
            >>>
            >>> # Dataset mode with subset retrieval
            >>> embs = cache.get(
            ...     model_name="dinov2-large",
            ...     image_paths=["train/001.jpg", "train/100.jpg"],
            ...     dataset_name="imagenet",
            ...     base_path="/data/imagenet"
            ... )
        """
        if not self.enabled:
            return None

        # Normalize paths
        query_paths = self._to_relative(image_paths, base_path)
        
        cache_key = self._compute_cache_key(
            model_name, query_paths if not dataset_name else [], 
            dataset_name, pooling_strategy, layers, **kwargs
        )
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            data = self._memory_cache[cache_key]
        else:
            cache_file = self.cache_dir / f"{cache_key}.npz"
            if not cache_file.exists():
                return None
            try:
                data = np.load(cache_file, allow_pickle=True)
                # Store in memory for future lookups (optional: implement LRU here if memory is tight)
                # We store the loaded dict-like object.
                # Note: np.load with mmap_mode=None loads into memory.
                self._memory_cache[cache_key] = dict(data)
                data = self._memory_cache[cache_key]
            except (OSError, ValueError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as e:
                logger.warning("Failed to load cache entry %s: %s", cache_key, e)
                self._evict_cache_entry(cache_key)
                return None

        # Retrieve specific data
        try:
            if dataset_name:
                # Dataset Mode: Load specific subset
                if "is_path_dict" in data and data["is_path_dict"]:
                    # Create lookup if not already created in memory cache
                    if "path_to_idx" not in data:
                        stored_paths = data["image_paths"]
                        data["path_to_idx"] = {str(p): i for i, p in enumerate(stored_paths)}
                    
                    path_to_idx = data["path_to_idx"]
                    stored_embeddings = data["embeddings"]
                    
                    indices = []
                    missing = []
                    for qp in query_paths:
                        if qp in path_to_idx:
                            indices.append(path_to_idx[qp])
                        else:
                            missing.append(qp)
                    
                    if missing:
                        # If a subset is missing, we must recompute (or implement partial return logic)
                        # Returning None forces recomputation of the requested batch.
                        return None
                        
                    return stored_embeddings[indices]
                else:
                    return None

            else:
                # Batch Mode: Return exactly what was saved
                if "is_dict" in data and data["is_dict"]:
                    result = {}
                    # Reconstruct keys from stored 'keys' array
                    keys = data["keys"]
                    for k in keys:
                        # Handle potential numpy types in keys
                        key_val = int(k) if isinstance(k, (int, np.integer, float, np.floating)) else str(k)
                        result[key_val] = data[f"layer_{k}"]
                    return result
                elif "embeddings" in data:
                    return data["embeddings"]
                else:
                    return None

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.warning("Error processing cached data for %s: %s", cache_key, e)
            return None

    def put(
        self,
        embeddings: Union[np.ndarray, Dict[str, np.ndarray], Dict[int, np.ndarray]],
        model_name: str,
        image_paths: List[str],
        dataset_name: Optional[str] = None,
        base_path: Optional[str] = None,
        pooling_strategy: Optional[str] = None,
        layers: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Store embeddings to cache.

        Saves embeddings to disk in compressed .npz format and updates in-memory
        cache and metadata. Supports multiple storage formats (array, dict, multi-layer).

        Args:
            embeddings: Embeddings to store. Can be:
                - np.ndarray: Simple array of shape [N, D]
                - Dict[str, np.ndarray]: Path-to-embedding mapping
                - Dict[int, np.ndarray]: Multi-layer embeddings (layer_id -> array)
            model_name: Model identifier (e.g., "dinov2-base").
            image_paths: List of image paths corresponding to embeddings.
            dataset_name: Optional dataset name for dataset mode.
            base_path: Optional base directory for converting to relative paths.
            pooling_strategy: Optional pooling method. Part of cache key.
            layers: Optional layer indices. Part of cache key.
            **kwargs: Additional configuration affecting cache key.

        Side Effects:
            - Writes .npz file to cache_dir
            - Updates in-memory cache
            - Updates metadata.json file
            - Prints warning if operation fails

        Examples:
            >>> # Store simple array
            >>> cache.put(
            ...     embeddings=np.random.randn(10, 768),
            ...     model_name="dinov2-base",
            ...     image_paths=["a.jpg", "b.jpg", ...]
            ... )
            >>>
            >>> # Store with dataset mode
            >>> cache.put(
            ...     embeddings=all_embeddings,
            ...     model_name="clip-vit-l-14",
            ...     image_paths=all_paths,
            ...     dataset_name="coco_train",
            ...     base_path="/data/coco"
            ... )
        """
        if not self.enabled:
            return

        # Normalize paths
        rel_paths = self._to_relative(image_paths, base_path)

        cache_key = self._compute_cache_key(
            model_name, rel_paths if not dataset_name else [], 
            dataset_name, pooling_strategy, layers, **kwargs
        )
        cache_file = self.cache_dir / f"{cache_key}.npz"

        config_entry = {
            "pooling_strategy": pooling_strategy,
            "layers": layers,
            **kwargs
        }

        is_path_dict_input = isinstance(embeddings, dict) and all(
            isinstance(k, str) for k in embeddings.keys()
        )
        if ((dataset_name is not None) or is_path_dict_input) and not isinstance(embeddings, dict):
            if len(embeddings) != len(rel_paths):
                raise ValueError(
                    "Cache put requires matching numbers of embeddings and image paths "
                    f"in dataset mode, got embeddings={len(embeddings)} and paths={len(rel_paths)}."
                )

        try:
            save_dict = {}
            
            # 1. Dataset Mode OR Dictionary of Path->Embedding provided
            if (dataset_name is not None) or is_path_dict_input:

                # If it's a list/array, we align it with rel_paths
                if not isinstance(embeddings, dict):
                    # Create pairs
                    path_emb_pairs = zip(rel_paths, embeddings)
                else:
                    # It's already a dict with string keys as paths
                    # Convert dict keys to relative paths for consistent storage
                    orig_paths = list(embeddings.keys())
                    rel_paths_for_dict = self._to_relative(orig_paths, base_path)
                    # Create pairs with relative paths
                    path_emb_pairs = [(rel_p, embeddings[orig_p]) for rel_p, orig_p in zip(rel_paths_for_dict, orig_paths)]

                # Sort by path for deterministic storage
                sorted_pairs = sorted(path_emb_pairs, key=lambda x: x[0])
                sorted_paths, sorted_embs = zip(*sorted_pairs)
                
                save_dict["is_path_dict"] = True
                save_dict["image_paths"] = np.array(sorted_paths, dtype=object)
                save_dict["embeddings"] = np.vstack(sorted_embs)

            # 2. Multi-layer dictionary (int keys)
            elif isinstance(embeddings, dict):
                save_dict["is_dict"] = True
                save_dict["keys"] = list(embeddings.keys())
                for key, array in embeddings.items():
                    save_dict[f"layer_{key}"] = array

            # 3. Simple Array (Batch mode)
            else:
                save_dict["is_dict"] = False
                save_dict["embeddings"] = embeddings

            # Save compressed
            np.savez_compressed(cache_file, **save_dict)

            # Update in-memory cache immediately
            self._memory_cache[cache_key] = save_dict

            # Update Metadata
            self.metadata["entries"][cache_key] = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "num_images": len(image_paths) if image_paths else 0,
                "base_path": str(base_path) if base_path else None,
                "file": cache_file.name,
                "config": config_entry,
                "created_at": os.path.getmtime(cache_file),
            }
            self._save_metadata()

        except (OSError, TypeError, ValueError, KeyError) as e:
            logger.warning("Failed to save cache entry %s: %s", cache_key, e)

    def clear(self, model_name: Optional[str] = None) -> None:
        if not self.enabled:
            return

        keys_to_remove = []
        if model_name is None:
            # Clear all
            for cache_file in self.cache_dir.glob("*.npz"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
            self.metadata["entries"] = {}
            self._memory_cache = {}
        else:
            # Clear specific model
            for cache_key, entry in self.metadata["entries"].items():
                if entry.get("model_name") == model_name:
                    cache_file = self.cache_dir / entry["file"]
                    if cache_file.exists():
                        try:
                            cache_file.unlink()
                        except OSError:
                            pass
                    keys_to_remove.append(cache_key)
                    self._memory_cache.pop(cache_key, None)
            
            for key in keys_to_remove:
                del self.metadata["entries"][key]
                
        self._save_metadata()

    def get_cache_info(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}

        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.npz"))
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "num_entries": len(self.metadata["entries"]),
            "total_size_mb": total_size / (1024 * 1024),
            "entries": self.metadata["entries"],
        }
