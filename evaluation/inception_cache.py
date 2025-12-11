"""
InceptionV3 Embedding Cache

Provides dataset-level caching of InceptionV3 features using the
CacheManager infrastructure from embedding_pipeline.
"""

import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

from data_partitioning.embedding_pipeline.cache_manager import CacheManager


class InceptionV3EmbeddingCache:
    """Cache for InceptionV3 embeddings at the dataset level."""

    def __init__(self, cache_dir: str = "./.inception_cache", enabled: bool = True):
        """
        Initialize the InceptionV3 embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
            enabled: Whether caching is enabled
        """
        self.cache_manager = CacheManager(cache_dir=cache_dir, enabled=enabled)
        self.model = None
        self.device = None
        self.transform = None

    def _init_model(self):
        """Lazy initialization of InceptionV3 model."""
        if self.model is not None:
            return

        # Set proxy for model download
        os.environ['https_proxy'] = 'http://proxy.example.org:3128'
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        os.environ.pop('https_proxy', None)

        # Replace final FC layer to get 2048-dim features
        self.model.fc = torch.nn.Identity()

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print(f"✓ InceptionV3 loaded on {self.device}")

    def get_embeddings(
        self,
        image_paths: List[str],
        batch_size: int = 50,
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Get InceptionV3 embeddings for image paths.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for extraction
            force_recompute: Ignore cache and recompute

        Returns:
            Dict mapping image_path → embedding (2048,)
        """
        # Try cache first
        if not force_recompute:
            cached = self._load_from_cache(image_paths)
            if cached is not None:
                print(f"✓ Loaded {len(image_paths)} embeddings from cache")
                return cached

        # Extract fresh
        print(f"Computing InceptionV3 embeddings for {len(image_paths)} images...")
        embeddings_dict = self._extract_embeddings(image_paths, batch_size)

        # Save to cache
        self._save_to_cache(image_paths, embeddings_dict)

        return embeddings_dict

    def _load_from_cache(self, image_paths: List[str]) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings from cache."""
        cache_key_params = {
            "model_name": "InceptionV3",
            "image_paths": sorted(image_paths),  # Sorted for consistent hashing
        }

        cached = self.cache_manager.get(**cache_key_params)

        if cached is not None:
            # Cached data is dict: {image_path: embedding}
            return cached

        return None

    def _save_to_cache(self, image_paths: List[str], embeddings_dict: Dict[str, np.ndarray]):
        """Save embeddings to cache."""
        cache_key_params = {
            "model_name": "InceptionV3",
            "image_paths": sorted(image_paths),
        }

        self.cache_manager.put(
            embeddings=embeddings_dict,
            **cache_key_params
        )

    def _extract_embeddings(
        self,
        image_paths: List[str],
        batch_size: int
    ) -> Dict[str, np.ndarray]:
        """Extract fresh embeddings from images."""
        self._init_model()

        embeddings_dict = {}

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]

            # Load and transform images
            batch_images = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                img_tensor = self.transform(img)
                batch_images.append(img_tensor)

            batch_tensor = torch.stack(batch_images).to(self.device)

            # Extract features
            with torch.no_grad():
                batch_features = self.model(batch_tensor).cpu().numpy()

            # Store in dict
            for path, features in zip(batch_paths, batch_features):
                embeddings_dict[path] = features

        return embeddings_dict

    def clear_cache(self):
        """Clear all cached embeddings."""
        self.cache_manager.clear(model_name="InceptionV3")
