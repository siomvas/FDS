import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from data_loading.dataset import BaseDataset


class DatasetEmbeddingCache:
    
    def __init__(self, model_name: str = "inception_v3", cache_dir: str = "./.embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_cache: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
        self.model_name = model_name
    
    def _get_paths(self, dataset_name: str):
        embeddings_path = self.cache_dir / f"{dataset_name}_embeddings.npz"
        metadata_path = self.cache_dir / f"{dataset_name}_metadata.json"
        return embeddings_path, metadata_path
    
    def exists(self, dataset_name: str) -> bool:
        embeddings_path, metadata_path = self._get_paths(dataset_name)
        return embeddings_path.exists() and metadata_path.exists()

    def _to_relative_path(self, path, base_path) -> str:
        try:
            return os.path.relpath(path, start=base_path)
        except ValueError:
            return str(Path(path).resolve())
    
    def load(self, dataset_name: str):
        if dataset_name in self._loaded_cache:
            return
        
        embeddings_path, metadata_path = self._get_paths(dataset_name)
        
        if not self.exists(dataset_name):
            raise FileNotFoundError(f"Cache not found for dataset '{dataset_name}'")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        data = np.load(embeddings_path)
        embeddings_matrix = data['embeddings']
        
        path_to_idx = metadata['path_to_index']
        
        self._loaded_cache[dataset_name] = (embeddings_matrix, path_to_idx)
        print(f"Loaded {dataset_name}: {embeddings_matrix.shape} matrix")
    
    def get(self, dataset_name: str, image_paths: List[str], base_path: str) -> np.ndarray:
        """Get embeddings for specific image paths. Returns (N, 2048) array."""
        if dataset_name not in self._loaded_cache:
            self.load(dataset_name)
        embeddings_matrix, path_to_idx = self._loaded_cache[dataset_name]
    
        indices = []
        missing = []
        for path in image_paths:
            rel_path = self._to_relative_path(path, base_path)
            if rel_path not in path_to_idx:
                missing.append(path)
            else:
                indices.append(path_to_idx[rel_path])
        if missing:
            raise KeyError(f"{len(missing)} images not found in cache. Examples: {missing[:3]}")
        
        return embeddings_matrix[indices]
    
    def precompute(
        self,
        dataset_name: str,
        image_paths: List[str],
        base_path: str,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
        force: bool = False,
        proxy: Optional[str] = None
    ) -> int:

        if self.exists(dataset_name) and not force:
            print(f"Cache already exists for '{dataset_name}'. Use force=True to recompute.")
            return 0
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Computing embeddings for {len(image_paths)} images on {device}...")

        embedding_dim_dict = {
            "inception_v3": 2048,
            'dinov2_vitb14': 768,
            'resnet50': 2048
        }
        
        # Load model
        torch.hub.set_dir(os.path.expanduser('~/.cache/torch/hub'))
        if self.model_name == "inception_v3":
            if proxy:
                os.environ['https_proxy'] = proxy
            model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
            os.environ.pop('https_proxy', None)
            model.fc = torch.nn.Identity()
            model.to(device)
            model.eval()
            transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        
        # Compute embeddings
        num_images = len(image_paths) # preallocate
        embedding_dim = embedding_dim_dict[self.model_name]
        all_embeddings = np.zeros((num_images, embedding_dim), dtype=np.float32)
        path_to_index = {}
        
        for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_end = min(batch_start + batch_size, num_images)
            batch_paths = image_paths[batch_start:batch_end]
            batch_tensors = []
            valid_path_indices = []
            for i, path in enumerate(batch_paths):
                try:
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                        batch_tensors.append(transform(img))
                        valid_path_indices.append(batch_start + i)
                        rel_path = self._to_relative_path(path, base_path)
                        path_to_index[rel_path] = batch_start + i
                except Exception as e:
                    print(f"Warning: Could not process image {path}: {e}")

            if not batch_tensors:
                continue            
            batch_stack = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                features = model(batch_stack).cpu().numpy()
            
            if len(features) == len(valid_path_indices): # no loading issues
                all_embeddings[valid_path_indices] = features
            else:
                for idx, f in zip(valid_path_indices, features):
                    all_embeddings[idx] = f

        embeddings_path, metadata_path = self._get_paths(dataset_name)        
        np.savez_compressed(embeddings_path, embeddings=all_embeddings)
        
        metadata = {
            'dataset_name': dataset_name,
            'num_images': num_images,
            'embedding_dim': embedding_dim,
            'model': self.model_name,
            'path_to_index': path_to_index
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded_cache.pop(dataset_name, None)
        
        print(f"✓ Cached {len(path_to_index)} embeddings to {embeddings_path}")
        print(f"  Size: {embeddings_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return len(path_to_index)
    
    def list_cached(self) -> List[str]:
        return [p.stem.replace("_metadata", "") for p in self.cache_dir.glob("*_metadata.json")]

    def get_info(self, dataset_name: str) -> Optional[Dict]:
        _, meta = self._get_paths(dataset_name)
        if meta.exists():
            with open(meta, 'r') as f: return json.load(f)
        return None

def main():

    parser = argparse.ArgumentParser(description="Pre-compute embeddings for a dataset")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default="./.embedding_cache")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Force recomputation of embeddings")
    parser.add_argument("--list", action="store_true", help="List cached datasets")
    parser.add_argument("--proxy", type=str, default="http://proxy.example.org:3128", help="Proxy for downloading models")
    
    args = parser.parse_args()
    cache = DatasetEmbeddingCache(cache_dir=args.cache_dir)
    
    if args.list:
        datasets = cache.list_cached()
        print(f"Cached datasets ({len(datasets)}):")
        for name in datasets:
            info = cache.get_info(name)
            # print all info nicely
            print(f"- {name}: {info['num_images']} images, model={info['model']}, dim={info['embedding_dim']}")
        return
    
    if not args.dataset or not args.base_path:
        parser.error("--dataset and --base_path are required unless using --list")
    
    print(f"Loading dataset '{args.dataset}'...")
    dataset = BaseDataset(args.dataset, args.base_path, args.split)
    
    image_paths = dataset.data
    print(f"Found {len(image_paths)} images")
    
    cache.precompute(
        args.dataset,
        image_paths,
        args.base_path,
        batch_size=args.batch_size,
        force=args.force,
        proxy=args.proxy
    )


if __name__ == "__main__":
    main()