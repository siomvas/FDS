import argparse
import logging
import traceback
import torch
from embedding.cache_manager import CacheManager
from embedding.embedding_extractors import ModelRegistry
from data_loading.dataset import BaseDataset

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (key for cache)")
    parser.add_argument("--base_path", type=str, required=True, help="Root path of the dataset")
    parser.add_argument("--model", type=str, default="inception_v3", help="Model identifier (e.g., inception_v3, dinov2-base)")
    parser.add_argument("--tokens", type=str, default="none", choices=["none", "cls", "k_last_cls", "k_last_all", "all_last"])
    parser.add_argument("--k_last", type=int, default=0, help="Number of last layers to consider when using k_last options")
    parser.add_argument("--cache_dir", type=str, default="/path/to/gtai/FDS/embedding_cache", help="Directory to store embeddings cache")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--proxy", type=str, default='http://proxy.example.org:3128', help="HTTP/HTTPS proxy for downloads")
    
    args = parser.parse_args()
    
    logger.info("Initializing CacheManager at %s...", args.cache_dir)
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    
    logger.info("Loading dataset '%s' from %s...", args.dataset, args.base_path)

    dataset = BaseDataset(dataset_name=args.dataset, base_path=args.base_path, split='train')
    images = dataset.data

    logger.info("Initializing extractor: %s...", args.model)
    try:
        extractor = ModelRegistry.create(
            args.model, 
            batch_size=args.batch_size,
            proxy=args.proxy,
            cache_manager=cache_manager,
            device="cuda" if torch.cuda.is_available() else "cpu",

        )
    except ValueError as e:
        logger.exception("Error initializing model: %s", e)
        return
    
    logger.info("Computing embeddings (this will automatically save to cache)...")
    try:
        tokens = args.tokens if args.tokens != "none" else None
        k_last = args.k_last if args.k_last > 0 else None
        extractor.extract_embeddings(
            images=images,
            use_cache=True,
            dataset_name=args.dataset,
            base_path=args.base_path,
            return_which_token=tokens,
            k_last=k_last
        )
        logger.info("Embedding extraction and caching completed successfully.")


    except (RuntimeError, ValueError, OSError) as e:
        logger.exception("Error during extraction: %s", e)
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
