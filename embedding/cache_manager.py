import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class CacheManager:
    """
    Manages caching of embeddings to disk.
    
    Features:
    - Content-based hashing (batch mode).
    - Dataset-based lookups (dataset mode).
    - Relative path support for shared HPC environments.
    - In-memory caching for faster repeated access.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "./.embedding_cache",
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._memory_cache = {}  # In-memory store for loaded .npz content

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._init_metadata()

    def _init_metadata(self) -> None:
        """Initialize or load cache metadata."""
        self.metadata_file = self.cache_dir / "metadata.json"

        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                try:
                    self.metadata = json.load(f)
                except json.JSONDecodeError:
                    self.metadata = {"version": "1.0", "entries": {}}
        else:
            self.metadata = {"version": "1.0", "entries": {}}
            self._save_metadata()

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _to_relative(self, paths: List[str], base_path: Optional[str]) -> List[str]:
        """Convert paths to relative if base_path is provided."""
        if not base_path:
            return [str(p) for p in paths]
        
        rel_paths = []
        for p in paths:
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
        # Create a unique key for the configuration
        config_dict = {
            "model_name": model_name,
            "pooling_strategy": pooling_strategy,
            "layers": layers,
            **kwargs
        }
        # Sort keys for consistency
        config_str = json.dumps(config_dict, sort_keys=True)
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
        """
        Retrieve cached embeddings.
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
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_key}: {e}")
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

        except Exception as e:
            print(f"Warning: Error processing cached data for {cache_key}: {e}")
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
        """
        Store embeddings.
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

        try:
            save_dict = {}
            
            # 1. Dataset Mode OR Dictionary of Path->Embedding provided
            if (dataset_name is not None) or (isinstance(embeddings, dict) and all(isinstance(k, str) for k in embeddings.keys())):
                
                # If it's a list/array, we align it with rel_paths
                if not isinstance(embeddings, dict):
                    if len(embeddings) != len(rel_paths):
                        print(f"Warning: Cache put failed. Embeddings {len(embeddings)} != Paths {len(rel_paths)}")
                        return
                    # Create pairs
                    path_emb_pairs = zip(rel_paths, embeddings)
                else:
                    # It's already a dict, assuming keys are compatible (metrics.py usually passes arrays, only specialized extractors pass dicts)
                    path_emb_pairs = embeddings.items()

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
            }
            self._save_metadata()

        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key}: {e}")

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