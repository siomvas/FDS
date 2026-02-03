import logging
from typing import Any, Dict
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForSemanticSegmentation
)
from utils.proxy_config import hpc_proxy

log = logging.getLogger(__name__)

def create_model(name: str, num_classes: int, task_type: str, **kwargs) -> nn.Module:
    """
    Adapter for Hugging Face Transformers.
    
    Args:
        name (str): The HF Hub ID (e.g., 'google/vit-base-patch16-224', 'facebook/detr-resnet-50').
        num_classes (int): Number of target classes.
        task_type (str): 'classification', 'detection', or 'segmentation'.
        **kwargs: Extra args passed to from_pretrained (e.g. revision, cache_dir).
    """
    
    # HF uses 'num_labels' instead of 'num_classes'
    # 'ignore_mismatched_sizes' is critical: it allows loading weights 
    # from a 1000-class model into a 10-class model by discarding the head.
    hf_args = {
        "num_labels": num_classes,
        "ignore_mismatched_sizes": True
    }
    
    # Merge with user kwargs (user args override defaults if collision)
    hf_args.update(kwargs)

    log.info(f"Initializing HF Model: {name} | Task: {task_type} | Args: {hf_args}")
    
    with hpc_proxy():
        try:
            if task_type == "classification":
                return AutoModelForImageClassification.from_pretrained(name, **hf_args)
            
            elif task_type == "detection":
                return AutoModelForObjectDetection.from_pretrained(name, **hf_args)
            
            elif task_type == "segmentation":
                return AutoModelForSemanticSegmentation.from_pretrained(name, **hf_args)
                
            else:
                raise ValueError(f"Task {task_type} not supported by HF Adapter")

        except OSError as e:
            log.error(f"Could not load model '{name}' from Hugging Face Hub. Check spelling or internet connection.")
            raise e