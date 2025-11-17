import gc, torch
import os
from datasets import Dataset
from typing import List, Dict, Any
import logging
import sys, os
from utils.argument import args
from utils import util_loader
from utils import helper
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from dataset import BaseDataset

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Utils for InternVL3 (referenced from Hugging Face: https://huggingface.co/OpenGVLab/InternVL3-38B) ---
# Could ignore these as it's only used for InternVL3 inference
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ---- Set up directory path ----
# from dotenv import load_dotenv, find_dotenv
# env_path = find_dotenv()
# load_dotenv(env_path)
# home_path = os.getenv("HOME_PATH")
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# print(">>> sys.path includes:", sys.path[-3:])

# ----- Configure logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ------- Main VLM pipeline functions -------------
# STEP 1: Prepare batch question, questions passed in, read from txt file
def prepare_batch_questions(data: Dataset,
                            question: str, max_samples: int = None) -> List[List[Dict[str, Any]]]:
    """
    Prepare batch questions in the format expected by MiniCPM.
    
    Args:
        data: Dataset containing images and labels
        label_set: List of label names
        question: Question to ask about each image
        max_samples: Maximum number of samples to process (None for all)
        skip_grayscale: Whether to skip grayscale images
        
    Returns:
        List of questions in batch format
    """
    questions = []
    processed_count = 0

    for i in range(len(data.data)):
        if max_samples and processed_count >= max_samples:
            break
        
        image_path = data.data[i] 
        label = data.labels[i]

        #print(f"Type of image_path[{i}]: {type(image_path)}")
        
        questions.append({
            'question': question,
            'label': label,
            'index': i,
            'filename': str(image_path)  # Original path
            #'saved_name': f"{label}_{processed_count:04}.jpg"
        })
        
        processed_count += 1
        
    logger.info(f"Prepared {len(questions)} questions for processing")
    return questions

# STEP 2: Process questions by batch through batch API
# ------------- For MiniCPM ------------------------------
def process_batch_minicpm(model, tokenizer, batch_data: List[Dict[str, Any]], 
                  temperature: float = 0.7, sampling: bool = True) -> List[Dict[str, str]]:
    """
    Process a batch of questions using MiniCPM's batch API.
    
    Args:
        batch_data: List of prepared question data
        temperature: Sampling temperature
        sampling: Whether to use sampling
        
    Returns:
        List of results with descriptions
    """
    results = []
    
    # Load images and reconstruct questions for batch processing
    questions = []
    for item in batch_data:
        # Load image
        pil_image = Image.open(item['filename']).convert('RGB')
        
        # Get question text
        question_text = item['question']
        
        # Reconstruct question with image
        question_batch = [{"role": "user", "content": [pil_image, question_text]}]
        questions.append(question_batch)
    
    try:
        # Process batch using batch API
        # Use the batch processing format: msgs=[msgs, msgs] with image=None
        res = model.chat(
            image=None,
            msgs=questions,  # Pass the list of message batches directly
            tokenizer=tokenizer,
            sampling=sampling,
            temperature=temperature,
            stream=False
        )
        
        responses = res
        
        logger.info(f"Successfully processed batch of {len(questions)} questions")
        #return responses
        
        # Ensure there's a right number of responses
        if len(responses) != len(batch_data):
            logger.warning(f"Expected {len(batch_data)} responses, got {len(responses)}")
            # Pad with error messages if needed
            while len(responses) < len(batch_data):
                responses.append("Error: No response received")
        
        # Combine results with metadata
        for i, (batch_item, response) in enumerate(zip(batch_data, responses)):
            results.append({
                "image": batch_item['filename'],
                "label": batch_item['label'],
                "description": response,
                "original_index": batch_item['index']
            })
            
    except Exception as e:
        logger.error(f"Batch processing failed...: {e}")
        
    return results

# ------------ For InternVL3 & 3.5 --------------------------------
def process_batch_internvl(model, tokenizer, batch_data):
    """Process with InternVL in batch"""
    #print_gpu_memory()
    results = []
    generation_config = dict(max_new_tokens=512, do_sample=False)
    pixel_values_list = []
    num_patches_list = []

    # Load and process all images in the batch
    for batch_item in batch_data:    
        # Load image
        pixel_values = load_image(batch_item['filename'], input_size=448, max_num=12).to(torch.bfloat16).cuda()
        
        # Track number of patches
        num_patches_list.append(pixel_values.size(0))
        pixel_values_list.append(pixel_values)

    # Concatenate all pixel values
    pixel_values = torch.cat(pixel_values_list, dim=0)

    # Extract question text
    question_text = batch_data[0]['question']
    question_with_image = f'<image>\n{question_text}'
    questions = [question_with_image] * len(num_patches_list)
    
    # Batch inference
    with torch.no_grad():
        responses = model.batch_chat(
            tokenizer, 
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config)

    #print_gpu_memory()

    # Combine results with metadata
    for i, (batch_item, response) in enumerate(zip(batch_data, responses)):
        results.append({
            "image": batch_item['filename'],
            "label": batch_item['label'],
            "description": response,
            "original_index": batch_item['index']})

    # Cleanup
    del pixel_values, pixel_values_list
    gc.collect()
    torch.cuda.empty_cache()

    return results

# ---------------------- For Qwen3 -------------------------
# def process_batch_qwen(model, processor, batch_data):
#     """
#     Batch inference with Qwen3-VL model.

#     Args:
#         model: Qwen3VLForConditionalGeneration model.
#         processor: AutoProcessor for Qwen3VL.
#         batch_data: list of dicts like:
#             {
#                 "filename": str,  # image file path
#                 "question": str,  # text prompt
#                 "label": str,     # optional label
#                 "index": int,     # index for reference
#             }
#         max_new_tokens: int, number of tokens to generate.

#     Returns:
#         List of dicts with keys ["image", "label", "description", "original_index"].
#     """

#     results = []

#     # Prepare all message pairs for batch
#     messages = []
#     question = batch_data[0]['question']
#     for item in batch_data:
#         messages.append([
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": item['filename']},
#                 {"type": "text", "text": question},
#             ],
#         }])

#     # Apply processor to batch
#     inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_dict=True,
#         padding=True, 
#         return_tensors="pt"
#     ).to(model.device)

#     # Generate batched responses
#     with torch.no_grad():
#         generated_ids = model.generate(**inputs, max_new_tokens=50)

#     # Decode each output
#     generated_ids_trimmed = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]

#     responses = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )

#     # Combine results with metadata
#     for i, (batch_item, response) in enumerate(zip(batch_data, responses)):
#         results.append({
#             "image": batch_item['filename'],
#             "label": batch_item['label'],
#             "description": response.strip(),
#             "original_index": batch_item['index']
#         })

#     #finally:
#     # CRITICAL: Clean up ALL intermediate tensors
#     if 'inputs' in locals():
#         del inputs
#     if 'generated_ids' in locals():
#         del generated_ids
#     if 'generated_ids_trimmed' in locals():
#         del generated_ids_trimmed
#     if 'messages' in locals():
#         del messages
        
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()

#     return results

def clear_model_cache(model):
    """Comprehensive cache clearing for Qwen models"""
    
    # Clear KV cache
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None
    
    # Clear vision encoder cache
    if hasattr(model, 'visual'):
        if hasattr(model.visual, 'cache_clear'):
            model.visual.cache_clear()
        model.visual.zero_grad(set_to_none=True)
    
    # Clear transformer layer caches
    if hasattr(model, 'transformer'):
        for layer in model.transformer.h:
            if hasattr(layer, 'attn'):
                if hasattr(layer.attn, 'past_key_value'):
                    layer.attn.past_key_value = None
    
    # Clear attention buffers
    for module in model.modules():
        if hasattr(module, '_buffers'):
            for buf_name in list(module._buffers.keys()):
                if any(kw in buf_name.lower() for kw in ['cache', 'past', 'hidden']):
                    module._buffers[buf_name] = None


def process_batch_qwen_optimized(model, processor, batch_data):
    """
    Memory-optimized Qwen batch processing with aggressive cleanup.
    
    This version addresses multiple memory leak sources:
    1. Vision encoder feature accumulation
    2. KV cache persistence
    3. Attention mask accumulation
    4. Image processor hidden copies
    5. Gradient graph creation
    """
    results = []
    images_loaded = []
    
    try:
        # Phase 1: Load images with explicit cleanup
        logger.debug(f"Loading {len(batch_data)} images...")
        for item in batch_data:
            with Image.open(item['filename']) as img:
                # Create copy and immediately close file
                img_copy = img.convert('RGB').copy()
                images_loaded.append(img_copy)
        
        # Phase 2: Build messages
        messages = []
        question = batch_data[0]['question']
        for i, item in enumerate(batch_data):
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": images_loaded[i]},
                    {"type": "text", "text": question},
                ],
            }])
        
        # Phase 3: Process with comprehensive no_grad
        with torch.no_grad():
            # Tokenize and prepare inputs
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate with explicit cache management
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                use_cache=True,  # Enable for efficiency
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            
            # Immediately clear model caches
            clear_model_cache(model)
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            responses = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        
        # Phase 4: Build results (outside no_grad, no tensors)
        for batch_item, response in zip(batch_data, responses):
            results.append({
                "image": batch_item['filename'],
                "label": batch_item['label'],
                "description": response.strip(),
                "original_index": batch_item['index']
            })
        
        logger.debug(f"Successfully processed batch of {len(results)} samples")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise
        
    finally:
        # Phase 5: Aggressive cleanup
        
        # Close and delete images
        for img in images_loaded:
            try:
                img.close()
            except:
                pass
        del images_loaded
        
        # Delete all intermediate variables
        if 'messages' in locals():
            del messages
        if 'inputs' in locals():
            # Delete each tensor in inputs dict
            for k in list(inputs.keys()):
                if isinstance(inputs[k], torch.Tensor):
                    del inputs[k]
            del inputs
        if 'generated_ids' in locals():
            del generated_ids
        if 'generated_ids_trimmed' in locals():
            del generated_ids_trimmed
        if 'responses' in locals():
            del responses
        
        # Force CUDA operations to complete
        torch.cuda.synchronize()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Log memory status
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.debug(f"GPU {i} after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return results


def process_with_periodic_deep_cleanup(model, processor, all_batches, cleanup_interval=10):
    """
    Process batches with periodic deep cleanup.
    
    Args:
        model: Qwen model
        processor: Qwen processor
        all_batches: List of batch_data
        cleanup_interval: Perform deep cleanup every N batches
    """
    all_results = []
    
    for batch_idx, batch_data in enumerate(all_batches):
        batch_num = batch_idx + 1
        
        # Process batch
        logger.info(f"Processing batch {batch_num}/{len(all_batches)}")
        batch_results = process_batch_qwen_optimized(model, processor, batch_data)
        all_results.extend(batch_results)
        
        # Periodic deep cleanup
        if batch_num % cleanup_interval == 0:
            logger.info(f"Performing deep cleanup at batch {batch_num}...")
            
            # Clear all model caches
            clear_model_cache(model)
            
            # Multiple rounds of cleanup
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Reset CUDA memory allocator
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            # Log detailed memory stats
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                logger.info(f"GPU {i}: {allocated:.2f}GB allocated, "
                          f"{reserved:.2f}GB reserved, "
                          f"{max_allocated:.2f}GB peak")
    
    return all_results


# Example usage in your main function
def main_with_memory_optimization(model, tokenizer, data, max_samples=None, batch_size=4):
    """Updated main function with memory optimization"""
    
    logger.info("Starting dataset processing with memory optimization...")
    
    # Prepare all data
    question = "Your question here"
    prepared_data = prepare_batch_questions(data, question, max_samples)
    
    # Split into batches
    batches = [prepared_data[i:i + batch_size] 
               for i in range(0, len(prepared_data), batch_size)]
    
    # Process with periodic cleanup
    all_results = process_with_periodic_deep_cleanup(
        model=model,
        processor=tokenizer,  # processor in your case
        all_batches=batches,
        cleanup_interval=5  # Deep cleanup every 5 batches
    )
    
    logger.info(f"Processing complete! Processed {len(all_results)} samples")
    return all_results

#================================= ENTIRE FLOW FOR STEP 1 ========================================
def main(model, tokenizer, data: BaseDataset,
        max_samples: int = 6, 
        batch_size: int = 4) -> None:
    """
    Main method to process the entire dataset.
    
    Args:
        model, tokenizer: VLM
        max_samples: Maximum number of samples to process
        batch_size: Number of samples to process in each batch
        dataset_split: Dataset split to use
    """
    #all_results = []
    logger.info("Starting dataset processing...")
    question = helper.read_file_to_string(args.step1_prompt_path)
    question = question.replace("[__CRITERION__]", str((args.prompt_label).lower()))

    # Prepare questions
    print("Preparing batch question.... ")
    print(f"Question: {question}")
    prepared_data = prepare_batch_questions(data, question, max_samples)

    jsonl_path = os.path.expanduser(args.step1_result_path)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        pass  # Create empty file
    
    total_processed = 0
    # Process in batches
    for i in range(0, len(prepared_data), batch_size):
        batch = prepared_data[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(prepared_data) + batch_size - 1)//batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")

        # Different inference method for different model
        # todo: refactor this to not use if else
        if "minicpm" in args.vlm_model.lower():
            batch_results = process_batch_minicpm(model=model, tokenizer=tokenizer, batch_data=batch)
        if "internvl" in args.vlm_model.lower():
            batch_results = process_batch_internvl(model=model, tokenizer=tokenizer, batch_data=batch)
        if "qwen" in args.vlm_model.lower():
            batch_results = process_batch_qwen_optimized(model=model, processor=tokenizer, batch_data=batch)
        else:
            logger.error("Model does not registered in step 1 yet...")
    
        #all_results.extend(batch_results)
        # Write results immediately to JSONL and free memory
        helper.append_batch_to_jsonl(jsonl_path, batch_results)
        total_processed += len(batch_results)

        del batch, batch_results
        gc.collect()
        torch.cuda.empty_cache()
        
        # Log progress
        logger.info(f"Batch {batch_num} complete. Total processed: {total_processed}/{len(prepared_data)}")
        
        if batch_num % 5 == 0:  # Log every 5 batches
            logger.info(f"Progress update: {total_processed} samples processed so far...")

    # Done all batches
    logger.info(f"Processing complete! Processed {total_processed} samples. Saved to {jsonl_path}")


if __name__ == "__main__":
    """Main execution function."""

    #model, tokenizer = util_loader.load_vlm(args.vlm_model)
    data = util_loader.load_data(args.dataset)
    # Run this on your ADE20K data
    util_loader.analyze_dataset_images(data, sample_size=100)
    
    # Process dataset
    #main(model=model, tokenizer=tokenizer, data=data, max_samples=None, batch_size=144)



