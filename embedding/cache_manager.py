import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class CacheManager:
    """
    Manages caching of embeddings to disk.

    Uses content-based hashing to identify cached embeddings.
    Supports both numpy arrays and dictionaries of arrays.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "./.embedding_cache",
        enabled: bool = True,
    ):
        """
        Args:
            cache_dir: Directory to store cached embeddings
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._init_metadata()

    def _init_metadata(self) -> None:
        """Initialize or load cache metadata."""
        self.metadata_file = self.cache_dir / "metadata.json"

        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "version": "1.0",
                "entries": {},
            }
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_cache_key(
        self,
        model_name: str,
        image_paths: List[str],
        pooling_strategy: Optional[str] = None,
        layers: Optional[List[int]] = None,
        **kwargs,
    ) -> str:
        """
        Compute cache key based on inputs.

        Args:
            model_name: Name of the model
            image_paths: List of image paths
            pooling_strategy: Pooling strategy used (if any)
            layers: Layer indices extracted (if any)
            **kwargs: Additional parameters that affect the cache

        Returns:
            Hash string as cache key
        """
        # Sort image paths for consistency
        sorted_paths = sorted(image_paths)

        # Create a dictionary of all parameters
        cache_params = {
            "model_name": model_name,
            "image_paths": sorted_paths,
            "pooling_strategy": pooling_strategy,
            "layers": layers,
            **kwargs,
        }

        # Convert to JSON string (sorted keys for consistency)
        cache_string = json.dumps(cache_params, sort_keys=True)

        # Compute SHA256 hash
        cache_hash = hashlib.sha256(cache_string.encode()).hexdigest()

        return cache_hash

    def get(
        self,
        model_name: str,
        image_paths: List[str],
        pooling_strategy: Optional[str] = None,
        layers: Optional[List[int]] = None,
        **kwargs,
    ) -> Optional[Union[np.ndarray, Dict[int, np.ndarray], Dict[str, np.ndarray]]]:
        """
        Retrieve cached embeddings if available.

        Args:
            model_name: Name of the model
            image_paths: List of image paths
            pooling_strategy: Pooling strategy used
            layers: Layer indices extracted
            **kwargs: Additional parameters

        Returns:
            Cached embeddings if found, None otherwise
            Can be: np.ndarray, Dict[int, np.ndarray], or Dict[str, np.ndarray]
        """
        if not self.enabled:
            return None

        cache_key = self._compute_cache_key(
            model_name, image_paths, pooling_strategy, layers, **kwargs
        )

        cache_file = self.cache_dir / f"{cache_key}.npz"

        if not cache_file.exists():
            return None

        # Load from cache
        try:
            data = np.load(cache_file, allow_pickle=True)

            # Check if it's a path-indexed dictionary (image_path → embedding)
            if "is_path_dict" in data and data["is_path_dict"]:
                # Reconstruct dictionary: path → embedding
                paths = data["image_paths"]
                embeddings = data["embeddings"]
                return {str(p): emb for p, emb in zip(paths, embeddings)}

            # Check if it's a dictionary of arrays (multi-layer)
            elif "is_dict" in data and data["is_dict"]:
                # Reconstruct dictionary: layer_id → array
                result = {}
                for key in data["keys"]:
                    result[int(key)] = data[f"layer_{key}"]
                return result
            else:
                # Single array
                return data["embeddings"]

        except Exception as e:
            print(f"Warning: Failed to load cache {cache_key}: {e}")
            return None

    def put(
        self,
        embeddings: Union[np.ndarray, Dict[int, np.ndarray], Dict[str, np.ndarray]],
        model_name: str,
        image_paths: List[str],
        pooling_strategy: Optional[str] = None,
        layers: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """
        Store embeddings in cache.

        Args:
            embeddings: Embeddings to cache (array, dict of int→arrays, or dict of str→arrays)
            model_name: Name of the model
            image_paths: List of image paths
            pooling_strategy: Pooling strategy used
            layers: Layer indices extracted
            **kwargs: Additional parameters
        """
        if not self.enabled:
            return

        cache_key = self._compute_cache_key(
            model_name, image_paths, pooling_strategy, layers, **kwargs
        )

        cache_file = self.cache_dir / f"{cache_key}.npz"

        # Save to cache
        try:
            if isinstance(embeddings, dict):
                # Check if dict has string keys (path-indexed)
                if embeddings and all(isinstance(k, str) for k in embeddings.keys()):
                    # Dictionary of path → embedding
                    sorted_paths = sorted(embeddings.keys())
                    embeddings_array = np.vstack([embeddings[p] for p in sorted_paths])

                    np.savez_compressed(
                        cache_file,
                        is_path_dict=True,
                        image_paths=np.array(sorted_paths, dtype=object),
                        embeddings=embeddings_array,
                    )
                else:
                    # Dictionary of int → arrays (multi-layer)
                    save_dict = {
                        "is_dict": True,
                        "keys": list(embeddings.keys()),
                    }
                    for key, array in embeddings.items():
                        save_dict[f"layer_{key}"] = array
                    np.savez_compressed(cache_file, **save_dict)
            else:
                # Single array
                np.savez_compressed(cache_file, embeddings=embeddings, is_dict=False)

            # Update metadata
            self.metadata["entries"][cache_key] = {
                "model_name": model_name,
                "num_images": len(image_paths),
                "pooling_strategy": pooling_strategy,
                "layers": layers,
                "file": cache_file.name,
            }
            self._save_metadata()

        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key}: {e}")

    def clear(self, model_name: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            model_name: If provided, only clear entries for this model.
                       If None, clear all cache.
        """
        if not self.enabled:
            return

        if model_name is None:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.npz"):
                cache_file.unlink()
            self.metadata["entries"] = {}
            self._save_metadata()
        else:
            # Clear specific model cache
            keys_to_remove = []
            for cache_key, entry in self.metadata["entries"].items():
                if entry["model_name"] == model_name:
                    cache_file = self.cache_dir / entry["file"]
                    if cache_file.exists():
                        cache_file.unlink()
                    keys_to_remove.append(cache_key)

            for key in keys_to_remove:
                del self.metadata["entries"][key]
            self._save_metadata()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {"enabled": False}

        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.npz")
        )

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "num_entries": len(self.metadata["entries"]),
            "total_size_mb": total_size / (1024 * 1024),
            "entries": self.metadata["entries"],
        }

    def __repr__(self) -> str:
        if self.enabled:
            return f"CacheManager(cache_dir={self.cache_dir}, enabled=True)"
        else:
            return "CacheManager(enabled=False)"
