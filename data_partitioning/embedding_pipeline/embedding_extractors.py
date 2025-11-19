"""
Foundation model wrappers for embedding extraction.

Provides abstract base class and concrete implementations for various
foundation models (DINO, CLIP, SigLIP, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)
import os


class EmbeddingExtractor(ABC):
    """Abstract base class for embedding extractors."""

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Args:
            model_name: Name/identifier of the model
            device: Device to run model on (cuda/cpu)
            batch_size: Batch size for extraction
            num_workers: Number of dataloader workers
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model: Optional[nn.Module] = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the foundation model."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of the embeddings."""
        pass

    @abstractmethod
    def extract_embeddings(
        self,
        images: List[str],
        return_all_tokens: bool = False,
    ) -> np.ndarray:
        """
        Extract embeddings from images.

        Args:
            images: List of image paths
            return_all_tokens: If True, return all token embeddings, not just CLS

        Returns:
            Embeddings as numpy array of shape:
            - (n_images, embed_dim) if return_all_tokens=False
            - (n_images, n_tokens, embed_dim) if return_all_tokens=True
        """
        pass

    def get_layer_embeddings(
        self,
        images: List[str],
        layers: List[int],
    ) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from specific layers.

        Args:
            images: List of image paths
            layers: List of layer indices to extract from

        Returns:
            Dictionary mapping layer index to embeddings
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support multi-layer extraction"
        )

    def _prepare_dataloader(
        self,
        images: List[str],
        transform: Optional[T.Compose] = None,
    ) -> DataLoader:
        """Create a dataloader for batch processing."""

        class ImageDataset(Dataset):
            def __init__(self, image_paths, transform=None):
                self.image_paths = image_paths
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, idx

        dataset = ImageDataset(images, transform)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )


class DINOv2Extractor(EmbeddingExtractor):
    """DINOv2 embedding extractor."""

    VARIANTS = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }

    def __init__(
        self,
        variant: str = "base",
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        proxy: Optional[str] = None,
    ):
        """
        Args:
            variant: DINOv2 variant (small/base/large/giant)
            device: Device to run on
            batch_size: Batch size for extraction
            num_workers: Number of workers
            proxy: HTTP proxy for downloading models (e.g., "http://proxy:3128")
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant {variant}. Choose from {list(self.VARIANTS.keys())}")

        model_name = self.VARIANTS[variant]
        super().__init__(model_name, device, batch_size, num_workers)
        self.variant = variant
        self.proxy = proxy

        # Set proxy if provided
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy

        self.load_model()

    def load_model(self) -> None:
        """Load DINOv2 model via HuggingFace."""
        print(f"Loading DINOv2 {self.variant}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ DINOv2 {self.variant} loaded on {self.device}")

    def get_embedding_dim(self) -> int:
        """Return embedding dimension based on variant."""
        dims = {"small": 384, "base": 768, "large": 1024, "giant": 1536}
        return dims[self.variant]

    def extract_embeddings(
        self,
        images: List[str],
        return_all_tokens: bool = False,
    ) -> np.ndarray:
        """Extract DINOv2 embeddings."""
        all_embeddings = []

        # Create dataloader with appropriate transform
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataloader = self._prepare_dataloader(images, transform)

        with torch.no_grad():
            for batch_imgs, _ in dataloader:
                batch_imgs = batch_imgs.to(self.device)

                # Forward pass
                outputs = self.model(batch_imgs, output_hidden_states=True)

                if return_all_tokens:
                    # Return all token embeddings (including CLS)
                    # Shape: (batch_size, n_tokens, embed_dim)
                    embeddings = outputs.last_hidden_state.cpu().numpy()
                else:
                    # Return only CLS token (first token)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                all_embeddings.append(embeddings)

        # Concatenate all batches
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings

    def get_layer_embeddings(
        self,
        images: List[str],
        layers: List[int],
    ) -> Dict[int, np.ndarray]:
        """Extract embeddings from specific layers."""
        layer_embeddings = {layer: [] for layer in layers}

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataloader = self._prepare_dataloader(images, transform)

        with torch.no_grad():
            for batch_imgs, _ in dataloader:
                batch_imgs = batch_imgs.to(self.device)

                outputs = self.model(batch_imgs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of all layer outputs

                for layer_idx in layers:
                    # Extract CLS token from specified layer
                    layer_output = hidden_states[layer_idx][:, 0, :].cpu().numpy()
                    layer_embeddings[layer_idx].append(layer_output)

        # Concatenate batches for each layer
        for layer_idx in layers:
            layer_embeddings[layer_idx] = np.concatenate(layer_embeddings[layer_idx], axis=0)

        return layer_embeddings


class CLIPExtractor(EmbeddingExtractor):
    """CLIP embedding extractor (vision encoder only)."""

    VARIANTS = {
        "base-patch32": "openai/clip-vit-base-patch32",
        "base-patch16": "openai/clip-vit-base-patch16",
        "large-patch14": "openai/clip-vit-large-patch14",
    }

    def __init__(
        self,
        variant: str = "base-patch16",
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Args:
            variant: CLIP variant (base-patch32/base-patch16/large-patch14)
            device: Device to run on
            batch_size: Batch size
            num_workers: Number of workers
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant {variant}. Choose from {list(self.VARIANTS.keys())}")

        model_name = self.VARIANTS[variant]
        super().__init__(model_name, device, batch_size, num_workers)
        self.variant = variant
        self.load_model()

    def load_model(self) -> None:
        """Load CLIP model."""
        print(f"Loading CLIP {self.variant}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ CLIP {self.variant} loaded on {self.device}")

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        dims = {"base-patch32": 768, "base-patch16": 768, "large-patch14": 1024}
        return dims[self.variant]

    def extract_embeddings(
        self,
        images: List[str],
        return_all_tokens: bool = False,
    ) -> np.ndarray:
        """Extract CLIP vision embeddings."""
        all_embeddings = []

        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_paths = images[i : i + self.batch_size]
            batch_images = [Image.open(path).convert("RGB") for path in batch_paths]

            # Process with CLIP processor
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Get vision model outputs
                vision_outputs = self.model.vision_model(**inputs, output_hidden_states=True)

                if return_all_tokens:
                    # Return all patch embeddings
                    embeddings = vision_outputs.last_hidden_state.cpu().numpy()
                else:
                    # Return pooled output (CLS token equivalent)
                    embeddings = vision_outputs.pooler_output.cpu().numpy()

                all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings


class SigLIPExtractor(EmbeddingExtractor):
    """SigLIP embedding extractor (improved CLIP from Google)."""

    VARIANTS = {
        "base-patch16-224": "google/siglip-base-patch16-224",
        "base-patch16-384": "google/siglip-base-patch16-384",
        "large-patch16-256": "google/siglip-large-patch16-256",
    }

    def __init__(
        self,
        variant: str = "base-patch16-224",
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Args:
            variant: SigLIP variant
            device: Device to run on
            batch_size: Batch size
            num_workers: Number of workers
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant {variant}. Choose from {list(self.VARIANTS.keys())}")

        model_name = self.VARIANTS[variant]
        super().__init__(model_name, device, batch_size, num_workers)
        self.variant = variant
        self.load_model()

    def load_model(self) -> None:
        """Load SigLIP model."""
        print(f"Loading SigLIP {self.variant}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ SigLIP {self.variant} loaded on {self.device}")

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        # SigLIP uses same dimensions as CLIP variants
        if "base" in self.variant:
            return 768
        elif "large" in self.variant:
            return 1024
        else:
            return 768  # Default

    def extract_embeddings(
        self,
        images: List[str],
        return_all_tokens: bool = False,
    ) -> np.ndarray:
        """Extract SigLIP vision embeddings."""
        all_embeddings = []

        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_paths = images[i : i + self.batch_size]
            batch_images = [Image.open(path).convert("RGB") for path in batch_paths]

            # Process with SigLIP processor
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.vision_model(**inputs, output_hidden_states=True)

                if return_all_tokens:
                    # Return all token embeddings
                    embeddings = outputs.last_hidden_state.cpu().numpy()
                else:
                    # Return pooled output
                    embeddings = outputs.pooler_output.cpu().numpy()

                all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings
