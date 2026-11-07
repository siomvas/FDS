from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import Qwen3VLMoeForConditionalGeneration
import os
import torch
import logging
from pathlib import Path
from data_loading.dataset import BaseDataset
import math
import torchvision.transforms as T
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from partitioning.vlm.utils.constants import FOOD101, INTERNVL3_38B, LLAMA33_70B, LLAMA31_8B_INSTRUCT, MINICPM26
from partitioning.vlm.utils.constants import INTERNVL35_8B, INTERNVL3_8B, QWEN3_8B, QWEN3_30B

log = logging.getLogger(__name__)

# used with InternVL3 38B to split models across multiple GPUs
def split_model(path):
    device_map = {}
    world_size = torch.cuda.device_count()
    log.info("Num of GPU: %s", world_size)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0

    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.output'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def analyze_dataset_images(data, sample_size=100):
    """Analyze image dimensions and sizes in dataset"""
    import random
    from PIL import Image, UnidentifiedImageError
    
    sample_paths = random.sample(data.data, min(sample_size, len(data.data)))
    
    resolutions = []
    file_sizes = []
    aspect_ratios = []
    
    for path in sample_paths:
        try:
            # Get file size
            import os
            file_size = os.path.getsize(path) / 1024  # KB
            file_sizes.append(file_size)
            
            # Get dimensions
            with Image.open(path) as img:
                width, height = img.size
                resolutions.append((width, height))
                aspect_ratios.append(width / height)
                
        except (OSError, ValueError, UnidentifiedImageError) as e:
            log.warning("Error while analyzing image %s: %s", path, e)
    
    import numpy as np
    
    log.info("=== Dataset Analysis (n=%d) ===", len(resolutions))
    log.info("Resolutions:")
    log.info("  Min: %s", min(resolutions, key=lambda x: x[0] * x[1]))
    log.info("  Max: %s", max(resolutions, key=lambda x: x[0] * x[1]))
    avg_side = int(np.mean([w * h for w, h in resolutions]) ** 0.5)
    log.info("  Avg: %dx%d", avg_side, avg_side)
    
    log.info("File Sizes:")
    log.info("  Min: %.1f KB", min(file_sizes))
    log.info("  Max: %.1f KB", max(file_sizes))
    log.info("  Avg: %.1f KB", float(np.mean(file_sizes)))
    log.info("  Median: %.1f KB", float(np.median(file_sizes)))
    
    log.info("Aspect Ratios:")
    log.info("  Min: %.2f", min(aspect_ratios))
    log.info("  Max: %.2f", max(aspect_ratios))
    log.info("  Avg: %.2f", float(np.mean(aspect_ratios)))
    
    # Estimate patches for Qwen
    patches_per_image = []
    for w, h in resolutions:
        # Simulate Qwen's dynamic preprocessing
        aspect_ratio = w / h
        image_size = 448
        max_num = 12
        
        # Rough estimation of patches
        target_ratios = [(i, j) for i in range(1, 5) for j in range(1, 5) if i*j <= max_num]
        best_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0]/r[1]))
        num_patches = best_ratio[0] * best_ratio[1]
        patches_per_image.append(num_patches)
    
    log.info("Estimated Qwen Patches per Image:")
    log.info("  Min: %d patches", min(patches_per_image))
    log.info("  Max: %d patches", max(patches_per_image))
    log.info("  Avg: %.1f patches", float(np.mean(patches_per_image)))
    
    log.info("Memory Estimate for batch_size=6:")
    avg_patches = np.mean(patches_per_image)
    total_patches = 6 * avg_patches
    memory_gb = total_patches * (448*448*3*4) / 1024**3
    log.info("  Total patches: %.0f", total_patches)
    log.info("  Raw pixel memory: %.2f GB", memory_gb)
    log.info("  With model overhead (3x): %.2f GB", memory_gb * 3)

# model and tokenizer
#----- Load model ----
# Model paths. Set FDS_VLM_MODEL_ROOT to the local directory that contains these
# manually downloaded model checkpoints.
model_path = Path(os.environ.get("FDS_VLM_MODEL_ROOT", "/path/to/local/vlm_models"))
MODEL_PATHS = {
    # vlm
    # minicpm & internVL
    MINICPM26: str(model_path / "minicpm26"),
    INTERNVL3_38B: str(model_path / "internVl3"),
    INTERNVL3_8B: str(model_path / "internVl3_8B"),
    INTERNVL35_8B: str(model_path / "internVl3_5"),

    # qwen
    QWEN3_8B: str(model_path / "qwen3_8B"),
    QWEN3_30B: str(model_path / "qwen3_30B"),

    # llm
    # llama
    LLAMA31_8B_INSTRUCT: str(model_path / "llama3.1-instruct"),
    LLAMA33_70B: str(model_path / "llama3.3")
}

# Data path
data_path = Path(os.environ.get("DATASETS_ROOT", "/path/to/datasets"))

# Model Registry
MODEL_REGISTRY = {}

def register_model(name: str):
    """Decorator to register a model loader function."""
    def decorator(load_func):
        MODEL_REGISTRY[name] = load_func
        return load_func
    return decorator

def get_model_loader(name: str):
    """Get the registered model loader function."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]

# Register VLM models
@register_model(MINICPM26)
def load_minicpm():
    """Load MiniCPM 2.6"""
    log.info("Loading %s...", MINICPM26)
    model_path = MODEL_PATHS[MINICPM26]
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to(device='cuda', dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

# Register VLM models
@register_model(QWEN3_8B)
def load_qwen_8b():
    """Load Qwen3 VL 8B"""
    log.info("Loading %s...", QWEN3_8B)
    model_path = MODEL_PATHS[QWEN3_8B]
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
                                                            dtype=torch.bfloat16,
                                                            attn_implementation="flash_attention_2",
                                                            device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer  # get the inner tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # safe default

    tokenizer.padding_side = "left"
    return model, processor

@register_model(QWEN3_30B)
def load_qwen3_30b():
    """Load Qwen3 VL 30B"""
    log.info("Loading %s...", QWEN3_30B)
    model_path = MODEL_PATHS[QWEN3_30B]
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path,
                                                            dtype=torch.bfloat16,
                                                            attn_implementation="eager",
                                                            device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer  # get the inner tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # safe default

    tokenizer.padding_side = "left"
    return model, processor

@register_model(INTERNVL3_38B)
def load_internvl3_38b():
    """Load InternVL3-38B with custom device map"""
    log.info("Loading %s...", INTERNVL3_38B)
    model_path = MODEL_PATHS[INTERNVL3_38B]
    device_map = split_model(model_path)
    
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False)
    
    return model, tokenizer

@register_model(INTERNVL3_8B)
def load_internvl3_8b():
    """Load InternVL3-8B with custom device map"""
    log.info("Loading %s...", INTERNVL3_8B)
    model_path = MODEL_PATHS[INTERNVL3_8B]
    device_map = split_model(model_path)
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False)
    
    return model, tokenizer

@register_model(INTERNVL35_8B)
def load_internvl3_5_8b():
    """Load InternVL3.5-8B with auto device map"""
    log.info("Loading %s...", INTERNVL35_8B)
    model_path = MODEL_PATHS[INTERNVL35_8B]
    #device_map = split_model(model_path)
    
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False)
    
    return model, tokenizer

# Register LLM models
@register_model(LLAMA31_8B_INSTRUCT)
def load_llama_8b():
    """Load LLaMA 3.1-Instruct 8B"""
    log.info("Loading %s...", LLAMA31_8B_INSTRUCT)
    model_path = MODEL_PATHS[LLAMA31_8B_INSTRUCT]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

@register_model(LLAMA33_70B)
def load_llama_70b():
    """Load LLaMA 3.3 70B"""
    log.info("Loading %s...", LLAMA33_70B)
    model_path = MODEL_PATHS[LLAMA33_70B]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

# Main loading functions
def load_vlm(ver: str = INTERNVL35_8B):
    """Load the VLM model and tokenizer."""
    loader_func = get_model_loader(ver)
    return loader_func()

def load_llm(ver: str = LLAMA33_70B):
    """Load the LLM model and tokenizer."""
    loader_func = get_model_loader(ver)
    return loader_func()

# Dataset loading
def load_data(dataset: str, split: str="train"):
    """Load dataset based on params set when run script."""
    # split = "val"
    # if dataset == FOOD101:
    #     split = "test"
    
    log.info("Dataset: %s", dataset)
    data = BaseDataset(dataset, str(data_path), split)
    return data

# Optional: Print available models
def list_available_models():
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())
