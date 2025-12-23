from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import Qwen3VLMoeForConditionalGeneration
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dataset import BaseDataset
import math
import torchvision.transforms as T
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from utils.constants import FOOD101, INTERNVL3_38B, LLAMA33_70B, LLAMA31_8B_INSTRUCT, MINICPM26
from utils.constants import INTERNVL35_8B, INTERNVL3_8B, QWEN3_8B, QWEN3_30B

# used with InternVL3 38B to split models across multiple GPUs
def split_model(path):
    device_map = {}
    world_size = torch.cuda.device_count()
    print("Num of GPU: ", world_size)
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
    from PIL import Image
    
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
                
        except Exception as e:
            print(f"Error with {path}: {e}")
    
    import numpy as np
    
    print(f"\n=== Dataset Analysis (n={len(resolutions)}) ===")
    print(f"\nResolutions:")
    print(f"  Min: {min(resolutions, key=lambda x: x[0]*x[1])}")
    print(f"  Max: {max(resolutions, key=lambda x: x[0]*x[1])}")
    print(f"  Avg: {int(np.mean([w*h for w,h in resolutions])**0.5)}×{int(np.mean([w*h for w,h in resolutions])**0.5)}")
    
    print(f"\nFile Sizes:")
    print(f"  Min: {min(file_sizes):.1f} KB")
    print(f"  Max: {max(file_sizes):.1f} KB")
    print(f"  Avg: {np.mean(file_sizes):.1f} KB")
    print(f"  Median: {np.median(file_sizes):.1f} KB")
    
    print(f"\nAspect Ratios:")
    print(f"  Min: {min(aspect_ratios):.2f}")
    print(f"  Max: {max(aspect_ratios):.2f}")
    print(f"  Avg: {np.mean(aspect_ratios):.2f}")
    
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
    
    print(f"\nEstimated Qwen Patches per Image:")
    print(f"  Min: {min(patches_per_image)} patches")
    print(f"  Max: {max(patches_per_image)} patches")
    print(f"  Avg: {np.mean(patches_per_image):.1f} patches")
    
    print(f"\nMemory Estimate for batch_size=6:")
    avg_patches = np.mean(patches_per_image)
    total_patches = 6 * avg_patches
    memory_gb = total_patches * (448*448*3*4) / 1024**3
    print(f"  Total patches: {total_patches:.0f}")
    print(f"  Raw pixel memory: {memory_gb:.2f} GB")
    print(f"  With model overhead (3x): {memory_gb*3:.2f} GB")

# model and tokenizer
#----- Load model ----
# note: change to custom path when run
# Model paths
model_path = "/path/to/archive/vision-saved"
MODEL_PATHS = {
    # vlm
    # minicpm & internVL
    MINICPM26: f"{model_path}/minicpm26",
    INTERNVL3_38B: f"{model_path}/internVl3",
    INTERNVL3_8B: f"{model_path}/internVl3_8B",
    INTERNVL35_8B: f"{model_path}/internVl3_5",

    # qwen
    QWEN3_8B: f"{model_path}/qwen3_8B",
    QWEN3_30B: f"{model_path}/qwen3_30B",

    # llm
    # llama
    LLAMA31_8B_INSTRUCT: f"{model_path}/llama3.1-instruct",
    LLAMA33_70B: f"{model_path}/llama3.3"
}

# Data path
data_path = "/path/to/gtai/FDS"

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
    print(f"Loading {MINICPM26}...")
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
    print(f"Loading {QWEN3_8B}...")
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
    print(f"Loading {QWEN3_30B}...")
    model_path = MODEL_PATHS[QWEN3_30B]
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path,
                                                            dtype=torch.bfloat16,
                                                            attn_implementation="flash_attention_2",
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
    print(f"Loading {INTERNVL3_38B}...")
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
    print(f"Loading {INTERNVL3_8B}...")
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
    print(f"Loading {INTERNVL35_8B}...")
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
    print(f"Loading {LLAMA31_8B_INSTRUCT}...")
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
    print(f"Loading {LLAMA33_70B}...")
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
def load_data(dataset: str, split: str="test"):
    """Load dataset based on params set when run script."""
    # split = "val"
    # if dataset == FOOD101:
    #     split = "test"
    
    print("Dataset: ", dataset)
    data = BaseDataset(dataset, data_path, split)
    return data

# Optional: Print available models
def list_available_models():
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())
