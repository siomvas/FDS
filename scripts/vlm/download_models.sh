#!/bin/bash
export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TRANSFORMERS_OFFLINE=0
export http_proxy=http://proxy.example.org:3128
export HTTP_PROXY=http://proxy.example.org:3128

# echo "Downloading CLIP safetensors..."
# python -c "
# from transformers import CLIPVisionModel, CLIPImageProcessor
# CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16', use_safetensors=True)
# CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch16')
# print('CLIP downloaded successfully.')
# "

# echo "Downloading inception_v3 weights..."
# python -c "
# from torchvision import models
# models.inception_v3(weights='IMAGENET1K_V1')
# print('inception_v3 downloaded successfully.')
# "

echo "Downloading resnet18 pretrained weights (timm)..."
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='timm/resnet18.a1_in1k', filename='model.safetensors')
print('resnet18 downloaded successfully.')
"

echo "All models downloaded."
