from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Union, Any
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loading.dataset import ImageDataset

from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPVisionModel,
    CLIPImageProcessor,
    SiglipVisionModel,
    SiglipImageProcessor,
    ViTMAEModel,
)

class ModelRegistry:
    _REGISTRY: Dict[str, Type["EmbeddingExtractor"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls):
            cls._REGISTRY[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, model_identifier: str, **kwargs: Any) -> "EmbeddingExtractor":
        # Handle cases without variants (e.g. "wavelet_scattering")
        if "-" in model_identifier:
            family, variant = model_identifier.split("-", 1)
        else:
            family, variant = model_identifier, "default"
        
        if family not in cls._REGISTRY:
             raise ValueError(f"Unknown model family '{family}'. Available: {list(cls._REGISTRY.keys())}")
             
        return cls._REGISTRY[family](variant=variant, **kwargs)


def collate_fn(batch: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Filters out None values (corrupt images) from the batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.stack(batch)


class EmbeddingExtractor(ABC):
    def __init__(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        proxy: Optional[str] = None,
        model_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.proxy = proxy
        self.model_dir = model_dir
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[nn.Module] = None
        self.processor: Any = None
        self.transform: Any = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the specific model and processor."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output dimension size."""
        pass

    def _get_dataloader(self, images: List[str]) -> DataLoader:
        if not (self.transform or self.processor):
            raise RuntimeError("Processor or Transform not initialized. Call load_model() first.")

        dataset = ImageDataset(images, transform=self.transform, processor=self.processor)
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
        k_last: int = 4,
    ) -> np.ndarray:
        """
        Default extraction for transformer-based models.
        Override in subclasses if needed.
        return_which_tokens: 'cls' or 'all_last' or 'k_last_cls' or 'k_last_all'
        """
        if not self.model:
            self.load_model()

        all_embeddings = []

        dataloader = self._get_dataloader(images)
        output_hidden = "k_last" in return_which_tokens

        with torch.no_grad():
            for batch in dataloader:
                if batch is None: continue
                batch = batch.to(self.device)
                
                if output_hidden:
                    outputs = self.model(pixel_values=batch, output_hidden_states=True)
                    if return_which_tokens == "k_last_cls":
                        selected_states = outputs.hidden_states[-k_last:]
                        cls_embeddings = [state[:, 0, :] for state in selected_states]
                        emb = torch.cat(cls_embeddings, dim=1)
                    elif return_which_tokens == "k_last_all":
                        selected_states = outputs.hidden_states[-k_last:]
                        emb = torch.cat(selected_states, dim=2)
                    else:
                        raise ValueError(f"Invalid return_which_tokens '{return_which_tokens}' for k_last option.")
                else:
                    outputs = self.model(pixel_values=batch)
                    if return_which_tokens == "cls":
                        # try pooler first
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
            return np.array([])
            
        return np.concatenate(all_embeddings, axis=0)


@ModelRegistry.register("mae")
class MAEExtractor(EmbeddingExtractor):
    VARIANTS = {
        "base": "facebook/vit-mae-base",
        "large": "facebook/vit-mae-large",
        "huge": "facebook/vit-mae-huge",
    }

    def __init__(self, variant: str = "base", **kwargs):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown MAE variant '{variant}'. Options: {list(self.VARIANTS.keys())}")
        super().__init__(self.VARIANTS[variant], **kwargs)
        self.variant = variant
        self.proxy = kwargs.get("proxy", None)

    def load_model(self) -> None:
        print(f"Loading MAE {self.variant}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = ViTMAEModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_embedding_dim(self) -> int:
        return {"base": 768, "large": 1024, "huge": 1280}[self.variant]


@ModelRegistry.register("dino")
class DINOv2Extractor(EmbeddingExtractor):
    VARIANTS = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }

    def __init__(self, variant: str = "base", **kwargs):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown DINOv2 variant '{variant}'")
        super().__init__(self.VARIANTS[variant], **kwargs)
        self.variant = variant

    def load_model(self) -> None:
        print(f"Loading DINOv2 {self.variant}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_embedding_dim(self) -> int:
        return {"small": 384, "base": 768, "large": 1024, "giant": 1536}[self.variant]


@ModelRegistry.register("clip")
class CLIPExtractor(EmbeddingExtractor):
    VARIANTS = {
        "base-patch32": "openai/clip-vit-base-patch32",
        "base-patch16": "openai/clip-vit-base-patch16",
        "large-patch14": "openai/clip-vit-large-patch14",
    }

    def __init__(self, variant: str = "base-patch16", **kwargs):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown CLIP variant '{variant}'")
        super().__init__(self.VARIANTS[variant], **kwargs)
        self.variant = variant

    def load_model(self) -> None:
        print(f"Loading CLIP {self.variant} (Vision)...")
        self.processor = CLIPImageProcessor.from_pretrained(self.model_name)
        self.model = CLIPVisionModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_embedding_dim(self) -> int:
        return {"base-patch32": 768, "base-patch16": 768, "large-patch14": 1024}[self.variant]


@ModelRegistry.register("siglip")
class SigLIPExtractor(EmbeddingExtractor):
    VARIANTS = {
        "base-patch16-224": "google/siglip-base-patch16-224",
        "base-patch16-384": "google/siglip-base-patch16-384",
        "large-patch16-256": "google/siglip-large-patch16-256",
    }

    def __init__(self, variant: str = "base-patch16-224", **kwargs):
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown SigLIP variant '{variant}'")
        super().__init__(self.VARIANTS[variant], **kwargs)
        self.variant = variant

    def load_model(self) -> None:
        print(f"Loading SigLIP {self.variant} (Vision)...")
        self.processor = SiglipImageProcessor.from_pretrained(self.model_name)
        self.model = SiglipVisionModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_embedding_dim(self) -> int:
        return 1024 if "large" in self.variant else 768

@ModelRegistry.register("raw_pixel_resizer")
class RawPixelResizerExtractor(EmbeddingExtractor):
    def __init__(self, variant: str = "64", **kwargs):
        super().__init__(model_name="raw_pixel_resizer", **kwargs)
        try:
            self.target_size = int(variant)
        except ValueError:
            raise ValueError(f"Invalid target size '{variant}' for RawPixelResizerExtractor.")

    def load_model(self) -> None:
        print(f"Using Raw Pixel Resizer with target size {self.target_size}x{self.target_size}...")
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
    ) -> np.ndarray:
        all_embeddings = []

        dataloader = self._get_dataloader(images)
        for batch in dataloader:
            if batch is None: continue
            batch_size = batch.size(0)
            emb = batch.view(batch_size, -1)
            all_embeddings.append(emb.cpu().numpy())
        if not all_embeddings:
            return np.array([])
        return np.concatenate(all_embeddings, axis=0)
    
@ModelRegistry.register("sparse_random_projection_resizer")
class SparseRandomProjectionResizerExtractor(EmbeddingExtractor):
    def __init__(self, variant: str = "64-1000", **kwargs):
        if variant == "default":
            variant = "64-1000"
        try:
            size_str, target_dim_str = variant.split("-")
            self.target_size = int(size_str)
            self.target_dim = int(target_dim_str)
        except ValueError:
            raise ValueError(f"Invalid variant '{variant}' for RandomProjectionResizerExtractor. Use format '<size>-<target_dim>'.")
        super().__init__(model_name="sparse_random_projection_resizer", **kwargs)
        self.transformer: Optional[Any] = None

    def load_model(self) -> None:
        print(f"Using Sparse Random Projection Resizer with target size {self.target_size}x{self.target_size} and target dim {self.target_dim}...")
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
        **kwargs: Any,
    ) -> np.ndarray:
        if self.transformer is None:
            self.load_model()
        all_embeddings = []
        dataloader = self._get_dataloader(images)
        for batch in dataloader:
            if batch is None: continue
            batch_size = batch.size(0)
            flat_batch = batch.view(batch_size, -1).cpu().numpy()
            try:
                projected = self.transformer.transform(flat_batch)
            except:
                self.transformer.fit(flat_batch)
                projected = self.transformer.transform(flat_batch)
            all_embeddings.append(projected)

        if not all_embeddings:
            return np.array([])

        return np.concatenate(all_embeddings, axis=0)


@ModelRegistry.register("UAE")
class UntrainedAutoEncoderExtractor(EmbeddingExtractor):
    def __init__(self, variant: str = "default", input_size: int = 64, **kwargs):
        if variant != "default":
            raise ValueError(f"Unknown variant '{variant}' for UntrainedAutoEncoderExtractor. Only 'default' is supported.")
        self.input_size = input_size
        super().__init__(model_name="untrained_autoencoder", **kwargs)
        self.autoencoder: Optional[nn.Module] = None

    def load_model(self) -> None:
        print(f"Using Untrained AutoEncoder with input size {self.input_size}x{self.input_size}")
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


    def get_embedding_dim(self) -> int:
        return 16 * (self.input_size // 8) * (self.input_size // 8)
    
    def extract_embeddings(
        self, 
        images: List[str], 
        return_which_tokens: str = "cls",
        k_last: int = 5,
    ) -> np.ndarray:
        if self.autoencoder is None:
            self.load_model()
        all_embeddings = []

        if self.autoencoder is None:
            self.load_model()

        dataloader = self._get_dataloader(images)
        with torch.no_grad():
            for batch in dataloader:
                if batch is None: continue
                batch = batch.to(self.device)
                encoded = self.autoencoder(batch)
                emb = encoded.view(encoded.size(0), -1)
                all_embeddings.append(emb.cpu().numpy())
        if not all_embeddings:
            return np.array([])
        return np.concatenate(all_embeddings, axis=0)

@ModelRegistry.register("wavelet_scattering")
class WaveletScatteringExtractor(EmbeddingExtractor):
    def __init__(self, variant: str = "224-3", **kwargs):
        if variant == "default":
            variant = "224-3"
        try:
            size_str, J_str = variant.split("-")
            self.input_size = int(size_str)
            self.J = int(J_str)
        except ValueError:
            raise ValueError(f"Invalid variant '{variant}' for WaveletScatteringExtractor. Use format '<size>-<J>'.")

        super().__init__(model_name="wavelet_scattering", **kwargs)
        self.scattering: Optional[Any] = None

    def load_model(self) -> None:
        try:
            from kymatio.torch import Scattering2D
        except ImportError:
            raise ImportError("kymatio is required for WaveletScatteringExtractor. Please install it via 'pip install kymatio'.")
        
        print(f"Using Wavelet Scattering with input size {self.input_size}x{self.input_size} and J={self.J}...")

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.scattering = Scattering2D(J=self.J, shape=(self.input_size, self.input_size)).to(self.device)
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
    ) -> np.ndarray:
        if self.scattering is None:
            self.load_model()
        all_embeddings = []

        dataloader = self._get_dataloader(images)
        with torch.no_grad():
            for batch in dataloader:
                if batch is None: continue
                batch = batch.to(self.device)
                scatter_coeffs = self.scattering(batch)
                emb = scatter_coeffs.view(scatter_coeffs.size(0), -1)
                all_embeddings.append(emb.cpu().numpy())
        if not all_embeddings:
            return np.array([])
        return np.concatenate(all_embeddings, axis=0)