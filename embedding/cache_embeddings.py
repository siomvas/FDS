import argparse
import sys
from pathlib import Path
from typing import List
import torch
from embedding.cache_manager import CacheManager
from embedding.embedding_extractors import ModelRegistry
from data_loading.dataset import BaseDataset

def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (key for cache)")
    parser.add_argument("--base_path", type=str, required=True, help="Root path of the dataset")
    parser.add_argument("--model", type=str, default="inception_v3", help="Model identifier (e.g., inception_v3, dinov2-base)")
    parser.add_argument("--tokens", type=str, default="cls", choices=["cls", "k_last_cls", "k_last_all", "all_last"])
    parser.add_argument("--k_last", type=int, default=4, help="Number of last layers to consider when using k_last options")
    parser.add_argument("--cache_dir", type=str, default="/path/to/gtai/FDS/embedding_cache", help="Directory to store embeddings cache")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--proxy", type=str, default='http://proxy.example.org:3128', help="HTTP/HTTPS proxy for downloads")
    
    args = parser.parse_args()
    
    print(f"Initializing CacheManager at {args.cache_dir}...")
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    
    print(f"Loading dataset '{args.dataset}' from {args.base_path}...")

    dataset = BaseDataset(dataset_name=args.dataset, base_path=args.base_path, split='train')
    images = dataset.data

    print(f"Initializing Extractor: {args.model}...")
    try:
        extractor = ModelRegistry.create(
            args.model, 
            batch_size=args.batch_size,
            proxy=args.proxy,
            cache_manager=cache_manager,
            device="cuda" if torch.cuda.is_available() else "cpu",

        )
    except ValueError as e:
        print(f"Error initializing model: {e}")
        return
    
    print("Computing embeddings... (This will automatically save to cache)")
    try:
        extractor.extract_embeddings(
            images=images,
            use_cache=True,
            dataset_name=args.dataset,
            base_path=args.base_path,
            return_which_token=args.tokens,
            k_last=args.k_last,
        )
        print("Embedding extraction and caching completed successfully.")


    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()