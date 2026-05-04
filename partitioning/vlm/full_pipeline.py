from partitioning.vlm import step1, step2a, step2b, step3
from partitioning.vlm.utils import util_loader
from partitioning.vlm.utils.argument import args
import gc
import torch
import logging

logger = logging.getLogger(__name__)

def main():
    # # Load model 
    # logger.debug("=============== Starting full ICTC pipeline... ========================")
    # logger.debug("--- Loading VLM ...")
    # vlm_model, vlm_tokenizer = util_loader.load_vlm(args.vlm_model)

    # logger.debug("----Loading dataset...")
    # # Load data: get dataset_name from argument when run `python script.py --dataset <name>`, default is: imagenet
    # data = util_loader.load_data(args.dataset, "train")

    # # Run pipeline
    # logger.debug("1) === Step 1...")
    # step1.main(vlm_model, vlm_tokenizer, data=data, max_samples=None, batch_size=72)

    # # Unload vlm here:
    # del vlm_model
    # del vlm_tokenizer
    # gc.collect()
    # torch.cuda.empty_cache()

    logger.info("--- VLM unloaded, loading LLM...")
    llm_model, llm_tokenizer = util_loader.load_llm(args.llama_ver)

    # # logger.debug("2a) === Step 2a...")
    # step2a.main(llm_model, llm_tokenizer, inference_batch_size=88)

    # # logger.debug("2b) === Step 2b...")
    # step2b.main(llm_model, llm_tokenizer)

    logger.info("3) === Step 3...")
    step3.main(llm_model, llm_tokenizer, inference_batch_size=88)
    
    logger.info("Full pipeline finished successfully!")

if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
