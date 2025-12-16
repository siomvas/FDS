import argparse
import sys
from pathlib import Path
from typing import List
from cache_manager import CacheManager
from embedding_extractors import ModelRegistry
from data_loading.dataset import BaseDataset

def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (key for cache)")
    parser.add_argument("--base_path", type=str, required=True, help="Root path of the dataset")
    parser.add_argument("--model", type=str, default="inception_v3", help="Model identifier (e.g., inception_v3, dinov2-base)")
    parser.add_argument("--cache_dir", type=str, default="./.embedding_cache")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--proxy", type=str, default=None, help="HTTP/HTTPS proxy for downloads")
    
    args = parser.parse_args()
    
    print(f"Initializing CacheManager at {args.cache_dir}...")
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    
    print(f"Loading dataset '{args.dataset}' from {args.base_path}...")

    dataset = BaseDataset(
        dataset_name=args.dataset,
        base_path=args.base_path,
        split="train"
    )
    image_paths = dataset.data

    print(f"Initializing Extractor: {args.model}...")
    try:
        extractor = ModelRegistry.create(
            args.model, 
            batch_size=args.batch_size,
            proxy=args.proxy,
            cache_manager=cache_manager
        )
    except ValueError as e:
        print(f"Error initializing model: {e}")
        return
    
    print("Computing embeddings... (This will automatically save to cache)")
    try:
        # We pass dataset_name and base_path to ensure it saves in "Dataset Mode"
        # (store relative paths, enable subset lookup)
        embeddings = extractor.extract_embeddings(
            image_paths, 
            use_cache=True, # Will check if fully cached first
            dataset_name=args.dataset,
            base_path=args.base_path
        )
        print(f"✓ Success. Embeddings shape: {embeddings.shape}")
        
        info = cache_manager.get_cache_info()
        print(f"Cache status: {info['num_entries']} entries, {info['total_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()