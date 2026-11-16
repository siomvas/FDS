"""InternVL3.5 image preprocessing and batched caption generation.

This adapter follows the model card's dynamic high-resolution preprocessing
and ``batch_chat`` API while keeping the CLEVR-4 pipeline independent of the
repository's import-time global VLM arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Sequence


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class InternVLRuntime:
    """Tokenizer and preprocessing settings paired with an InternVL model."""

    tokenizer: Any
    max_tiles: int = 12
    input_size: int = 448


def resolve_device_map(device: str, visible_gpu_count: int):
    """Return the Accelerate placement used for the requested CUDA topology."""

    if device == "auto":
        return "auto"
    if device.startswith("cuda") and visible_gpu_count > 1:
        return "balanced"
    if device.startswith("cuda"):
        return {"": device}
    return None


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Select the dynamic-tile grid used by the official InternVL example."""

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            tiled_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * tiled_area:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Any,
    *,
    min_tiles: int = 1,
    max_tiles: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Any]:
    """Resize an image into InternVL's aspect-ratio-preserving tile grid."""

    if min_tiles <= 0 or max_tiles < min_tiles:
        raise ValueError("InternVL tile limits must satisfy 1 <= min_tiles <= max_tiles.")
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (columns, rows)
            for tile_count in range(min_tiles, max_tiles + 1)
            for columns in range(1, tile_count + 1)
            for rows in range(1, tile_count + 1)
            if min_tiles <= columns * rows <= max_tiles
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )
    resized = image.resize((image_size * columns, image_size * rows))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(
    image_path: str | Path,
    *,
    input_size: int = 448,
    max_tiles: int = 12,
):
    """Load and normalize one image into a stack of InternVL tiles."""

    import numpy as np
    import torch
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        tiles = dynamic_preprocess(
            image,
            image_size=input_size,
            max_tiles=max_tiles,
            use_thumbnail=True,
        )
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensors = []
    for tile in tiles:
        array = np.asarray(tile, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append((tensor - mean) / std)
    return torch.stack(tensors)


def load_model(
    model_name: str,
    revision: str | None,
    device: str,
    *,
    max_tiles: int,
) -> tuple[Any, InternVLRuntime]:
    """Load an InternVL custom-code checkpoint on one or more visible GPUs."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    if max_tiles <= 0:
        raise ValueError("internvl_max_tiles must be positive.")
    device_map = resolve_device_map(device, torch.cuda.device_count())
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if importlib.util.find_spec("flash_attn") is not None:
        model_kwargs["use_flash_attn"] = True
    if device_map is not None:
        # Stream weights directly to the only visible GPU. Loading the 38B
        # checkpoint on CPU and calling ``model.to`` afterwards needlessly
        # faults the full 76.8 GB checkpoint through host memory first.
        model_kwargs["device_map"] = device_map
    model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
    if "device_map" not in model_kwargs:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        use_fast=False,
    )
    return model, InternVLRuntime(tokenizer=tokenizer, max_tiles=max_tiles)


def _vision_device(model: Any, fallback: str):
    import torch

    for attribute in ("vision_model", "vision_tower"):
        module = getattr(model, attribute, None)
        if module is None:
            continue
        try:
            return next(module.parameters()).device
        except StopIteration:
            continue
    model_device = getattr(model, "device", None)
    return model_device if model_device is not None else torch.device(fallback)


def generate_batch(
    model: Any,
    runtime: InternVLRuntime,
    image_paths: Sequence[str],
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
) -> list[str]:
    """Generate one criterion-specific description for every supplied image."""

    import torch

    pixel_values_list = [
        load_image(
            image_path,
            input_size=runtime.input_size,
            max_tiles=runtime.max_tiles,
        )
        for image_path in image_paths
    ]
    num_patches_list = [values.size(0) for values in pixel_values_list]
    pixel_values = torch.cat(pixel_values_list).to(
        device=_vision_device(model, device), dtype=torch.bfloat16
    )
    questions = [f"<image>\n{prompt}"] * len(image_paths)
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    with torch.inference_mode():
        responses = model.batch_chat(
            runtime.tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )
    if isinstance(responses, str):
        responses = [responses]
    if len(responses) != len(image_paths):
        raise RuntimeError(
            f"InternVL returned {len(responses)} responses for {len(image_paths)} images."
        )
    del pixel_values, pixel_values_list
    return [str(response).strip() for response in responses]
