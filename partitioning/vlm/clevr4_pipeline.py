"""Resumable four-stage semantic annotation pipeline for CLEVR-4.

The pipeline mirrors the ICTC flow used by the existing VLM implementation:

1. A VLM describes each image with respect to one semantic criterion.
2. A text LLM reduces each description to a short free-form label.
3. The LLM consolidates the observed labels into a fixed taxonomy.
4. The LLM assigns every description to one consolidated category.

CLEVR-4 ground-truth annotations are deliberately excluded from all model
prompts.  They are loaded only after step 4 to measure whether each discovered
taxonomy aligns with the corresponding true taxonomy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import gc
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

CLEVR4_CRITERIA = ("texture", "shape", "color", "count")
DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_VLM_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LLM_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_MAX_PIXELS = {
    # Fine surface patterns need the native 640x480 image budget.  Use a
    # lower-cost 100k-pixel resize for the other criteria.
    "texture": 28 * 28 * 512,
    "shape": 28 * 28 * 128,
    "color": 28 * 28 * 128,
    "count": 28 * 28 * 128,
}
CRITERION_CONFIG: dict[str, dict[str, str | int]] = {
    "texture": {
        "prompt_label": "shared foreground-object surface texture or material",
        "guidance": (
            "Describe the most discriminative visible surface texture or material "
            "shared by the foreground objects. If a repeated motif, tiling, or printed "
            "marking is present, describe its visual structure precisely rather than "
            "stopping at generic finish words such as smooth, matte, or shiny. Use "
            "material or finish as the primary description only for genuinely "
            "unpatterned surfaces. Ignore object color, the floor, walls, shadows, and "
            "background."
        ),
        "num_classes": 10,
    },
    "shape": {
        "prompt_label": "shared foreground-object shape",
        "guidance": (
            "Describe the geometric or recognizable shape shared by the foreground "
            "objects. Ignore viewpoint, size, color, texture, and background."
        ),
        "num_classes": 10,
    },
    "color": {
        "prompt_label": "shared foreground-object color",
        "guidance": (
            "Describe the color shared by the foreground objects. Ignore the floor, "
            "walls, lighting cast, shadows, and background colors."
        ),
        "num_classes": 10,
    },
    "count": {
        "prompt_label": "number of foreground objects",
        "guidance": (
            "Count all distinct foreground objects, including partially occluded ones. "
            "Ignore the floor, walls, and shadows."
        ),
        "num_classes": 10,
    },
}


@dataclass(frozen=True)
class Clevr4Record:
    """One CLEVR-4 image and its held-out evaluation annotations."""

    image_id: str
    image: str
    split: str
    texture: str
    shape: str
    color: str
    count: str


@dataclass(frozen=True)
class PipelinePaths:
    """Stable artifact paths for one criterion run."""

    output_dir: Path
    step1: Path
    step2a: Path
    step2b: Path
    step2b_metadata: Path
    step3: Path
    validation: Path
    manifest: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def criterion_paths(output_root: Path, criterion: str) -> PipelinePaths:
    output_dir = output_root / f"output_{criterion}_10k"
    return PipelinePaths(
        output_dir=output_dir,
        step1=output_dir / "step1_result.jsonl",
        step2a=output_dir / "step2a_result.jsonl",
        step2b=output_dir / "step2b_result.txt",
        step2b_metadata=output_dir / "step2b_metadata.json",
        step3=output_dir / "step3_result.jsonl",
        validation=output_dir / "validation.json",
        manifest=output_dir / "run_manifest.json",
    )


def _single_match(paths: Sequence[Path], description: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"Could not find {description}.")
    if len(paths) > 1:
        rendered = ", ".join(str(path) for path in paths)
        raise ValueError(f"Found multiple candidates for {description}: {rendered}")
    return paths[0]


def load_clevr4_records(
    dataset_root: str | Path,
    *,
    splits: Sequence[str] | None = None,
    expected_samples: int | None = 10_531,
) -> list[Clevr4Record]:
    """Load the official CLEVR-4 image/annotation layout strictly.

    ``dataset_root`` may point either at the extracted dataset itself or at a
    parent directory containing it.  Every annotation must resolve to one PNG.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CLEVR-4 dataset root does not exist: {root}")

    direct_annotation = root / "clevr_4_annots.json"
    if direct_annotation.is_file():
        annotation_path = direct_annotation
    else:
        shallow_matches = sorted(root.glob("*/clevr_4_annots.json"))
        annotation_path = _single_match(
            shallow_matches or sorted(root.rglob("clevr_4_annots.json")),
            "clevr_4_annots.json under the dataset root",
        )
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, dict):
        raise ValueError(f"Expected an annotation dictionary in {annotation_path}.")

    images_dir = annotation_path.parent / "images"
    if images_dir.is_dir():
        image_paths = sorted(
            (Path(entry.path) for entry in os.scandir(images_dir) if entry.name.endswith(".png")),
            key=lambda path: path.name,
        )
    else:
        image_paths = sorted(annotation_path.parent.rglob("*.png"))
    image_candidates: dict[str, list[Path]] = {}
    for image_path in image_paths:
        image_candidates.setdefault(image_path.stem, []).append(image_path)

    requested_splits = {value.lower() for value in splits} if splits else None
    records: list[Clevr4Record] = []
    missing_images: list[str] = []
    invalid_annotations: list[str] = []
    for image_id, annotation in sorted(annotations.items()):
        if not isinstance(annotation, dict):
            invalid_annotations.append(str(image_id))
            continue
        if any(key not in annotation for key in (*CLEVR4_CRITERIA, "split")):
            invalid_annotations.append(str(image_id))
            continue
        split = str(annotation["split"]).lower()
        if requested_splits is not None and split not in requested_splits:
            continue
        candidates = image_candidates.get(str(image_id), [])
        if len(candidates) != 1:
            missing_images.append(str(image_id))
            continue
        records.append(
            Clevr4Record(
                image_id=str(image_id),
                image=str(candidates[0].absolute()),
                split=split,
                texture=str(annotation["texture"]).strip().lower(),
                shape=str(annotation["shape"]).strip().lower(),
                color=str(annotation["color"]).strip().lower(),
                count=str(annotation["count"]).strip().lower(),
            )
        )

    if invalid_annotations:
        raise ValueError(
            f"{len(invalid_annotations)} annotations are missing CLEVR-4 fields; "
            f"first ids: {invalid_annotations[:5]}"
        )
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} annotations did not resolve to exactly one PNG; "
            f"first ids: {missing_images[:5]}"
        )
    if expected_samples is not None and len(records) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} CLEVR-4 records but loaded {len(records)} "
            f"from {annotation_path}."
        )
    return records


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(row)
    return rows


def completed_image_ids(path: str | Path) -> set[str]:
    return {
        str(row["image_id"])
        for row in read_jsonl(path)
        if row.get("image_id") is not None
    }


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    if not payload:
        return
    original_size = path.stat().st_size if path.exists() else 0
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("JSONL append made no forward progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
        except OSError:
            LOGGER.exception(
                "Could not roll back failed JSONL append for %s to %d bytes",
                path,
                original_size,
            )
        raise
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def atomic_write_text(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value.rstrip() + "\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def batched(values: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _normalise_label(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"^assistant\s*:?[\s\n]*", "", text, flags=re.IGNORECASE)
    text = text.splitlines()[0] if text.splitlines() else text
    text = text.strip(" \t\n\r\"'`.,;:[]{}()")
    text = re.sub(r"\s+", " ", text)
    return text


def clean_step2a_label(response: str) -> str:
    """Normalize a short free-form step-2a label without guessing its class."""

    text = _normalise_label(response)
    text = re.sub(r"^(?:the\s+)?(?:answer|label)\s+is\s+", "", text)
    return text.strip(" \t\n\r\"'`.,;:")


def parse_categories(response: str, expected: int = 10) -> list[str]:
    """Extract unique ``Answer N: category`` values from step 2b output."""

    indexed: list[tuple[int, str]] = []
    pattern = re.compile(r"^\s*Answer\s+(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
    for line in response.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = _normalise_label(match.group(2))
        if value:
            indexed.append((int(match.group(1)), value))
    indexed.sort(key=lambda item: item[0])
    categories = list(dict.fromkeys(value for _, value in indexed))
    if len(categories) != expected:
        raise ValueError(
            f"Expected {expected} unique step-2b categories, found {len(categories)}: "
            f"{categories}"
        )
    return categories


def match_category(response: str, categories: Sequence[str]) -> tuple[str, str]:
    """Map a model response onto a supplied category while recording the method."""

    cleaned = _normalise_label(response)
    normalized = {_normalise_label(category): category for category in categories}
    if cleaned in normalized:
        return normalized[cleaned], "exact"

    contained = [
        category
        for key, category in normalized.items()
        if key and (key in cleaned or cleaned in key)
    ]
    if len(contained) == 1:
        return contained[0], "contained"

    scores = [
        (SequenceMatcher(None, cleaned, key).ratio(), category)
        for key, category in normalized.items()
    ]
    score, category = max(scores, default=(0.0, "Unknown"))
    if score >= 0.60:
        return category, "fuzzy"
    return "Unknown", "unmatched"


def render_prompt(prompt_dir: Path, filename: str, replacements: Mapping[str, Any]) -> str:
    prompt = (prompt_dir / filename).read_text(encoding="utf-8")
    for key, value in replacements.items():
        prompt = prompt.replace(f"[__{key}__]", str(value))
    unresolved = re.findall(r"\[__[A-Z0-9_]+__\]", prompt)
    if unresolved:
        raise ValueError(f"Unresolved prompt placeholders in {filename}: {unresolved}")
    return prompt


def _torch_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_vlm(
    model_name: str,
    revision: str | None,
    device: str,
    max_pixels: int,
    *,
    internvl_max_tiles: int = 12,
):
    if "internvl" in model_name.lower():
        from partitioning.vlm.internvl_adapter import load_model

        return load_model(
            model_name,
            revision,
            device,
            max_tiles=internvl_max_tiles,
        )

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_name,
        revision=revision,
        min_pixels=28 * 28 * 128,
        max_pixels=max_pixels,
    )
    return model, processor


def resolve_llm_device_map(device: str, visible_gpu_count: int) -> str | None:
    """Choose a generation-safe placement for a text model.

    Accelerate's ``auto`` strategy fills GPU 0 before GPU 1, which leaves
    little KV-cache headroom for a model that nearly fills the first device.
    ``balanced`` spreads a multi-GPU model evenly while preserving the
    single-GPU behavior used by the original four-worker pipeline.
    """

    if device == "auto":
        return "auto"
    if device.startswith("cuda") and visible_gpu_count > 1:
        return "balanced"
    return None


def resolve_llm_attention_backend() -> str:
    """Use FlashAttention 2 when its tested runtime package is available."""

    return (
        "flash_attention_2"
        if importlib.util.find_spec("flash_attn") is not None
        else "sdpa"
    )


def _load_llm(model_name: str, revision: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": torch.bfloat16,
        "attn_implementation": resolve_llm_attention_backend(),
        "low_cpu_mem_usage": True,
    }
    device_map = resolve_llm_device_map(device, torch.cuda.device_count())
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if "device_map" not in model_kwargs:
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def _generate_vlm(
    model: Any,
    processor: Any,
    image_paths: Sequence[str],
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
) -> list[str]:
    from partitioning.vlm.internvl_adapter import InternVLRuntime

    if isinstance(processor, InternVLRuntime):
        from partitioning.vlm.internvl_adapter import generate_batch

        return generate_batch(
            model,
            processor,
            image_paths,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    import torch

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image_path in image_paths
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    trimmed = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]
    responses = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    del inputs, generated, trimmed
    return [response.strip() for response in responses]


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: str,
    max_new_tokens: int,
    do_sample: bool = False,
) -> list[str]:
    import torch

    rendered = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Follow the requested output format exactly."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(temperature=0.7, top_p=0.8, top_k=20)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    trimmed = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, outputs)
    ]
    responses = tokenizer.batch_decode(trimmed, skip_special_tokens=True)
    del inputs, outputs, trimmed, rendered
    return [response.strip() for response in responses]


def _run_adaptive_batches(
    items: Sequence[Any],
    initial_batch_size: int,
    generate: Any,
    consume: Any,
    *,
    stage_name: str,
) -> None:
    """Run batches with automatic CUDA-OOM backoff and durable consumption."""

    import torch

    index = 0
    batch_size = initial_batch_size
    while index < len(items):
        batch = items[index : index + batch_size]
        try:
            outputs = generate(batch)
        except torch.cuda.OutOfMemoryError:
            _torch_cleanup()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            LOGGER.warning("%s OOM; reducing batch size to %d", stage_name, batch_size)
            continue
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"{stage_name} returned {len(outputs)} outputs for {len(batch)} inputs."
            )
        consume(batch, outputs)
        index += len(batch)
        if index == len(items) or index % max(100, batch_size * 10) == 0:
            LOGGER.info("%s progress: %d/%d", stage_name, index, len(items))
        del outputs
        _torch_cleanup()


def run_step1(
    records: Sequence[Clevr4Record],
    criterion: str,
    paths: PipelinePaths,
    prompt_dir: Path,
    model: Any,
    processor: Any,
    *,
    device: str,
    batch_size: int,
) -> None:
    completed = completed_image_ids(paths.step1)
    pending = [record for record in records if record.image_id not in completed]
    if not pending:
        LOGGER.info("Step 1 already complete for %s (%d rows).", criterion, len(completed))
        return
    config = CRITERION_CONFIG[criterion]
    prompt = render_prompt(
        prompt_dir,
        "step1_prompt.txt",
        {
            "CRITERION": config["prompt_label"],
            "CRITERION_GUIDANCE": config["guidance"],
        },
    )

    def generate(batch: Sequence[Clevr4Record]) -> list[str]:
        return _generate_vlm(
            model,
            processor,
            [record.image for record in batch],
            prompt,
            device=device,
            max_new_tokens=24,
        )

    def consume(batch: Sequence[Clevr4Record], outputs: Sequence[str]) -> None:
        append_jsonl(
            paths.step1,
            (
                {
                    "image_id": record.image_id,
                    "image": record.image,
                    "split": record.split,
                    "criterion": criterion,
                    "description": response.strip(),
                }
                for record, response in zip(batch, outputs)
            ),
        )

    _run_adaptive_batches(
        pending,
        batch_size,
        generate,
        consume,
        stage_name=f"{criterion}/step1",
    )


def run_step2a(
    criterion: str,
    paths: PipelinePaths,
    prompt_dir: Path,
    model: Any,
    tokenizer: Any,
    *,
    device: str,
    batch_size: int,
) -> None:
    source = read_jsonl(paths.step1)
    completed = completed_image_ids(paths.step2a)
    pending = [row for row in source if str(row["image_id"]) not in completed]
    if not pending:
        LOGGER.info("Step 2a already complete for %s (%d rows).", criterion, len(completed))
        return
    config = CRITERION_CONFIG[criterion]
    template = render_prompt(
        prompt_dir,
        "step2a_prompt.txt",
        {
            "CRITERION": config["prompt_label"],
            "CRITERION_GUIDANCE": config["guidance"],
        },
    )

    def generate(batch: Sequence[dict[str, Any]]) -> list[str]:
        prompts = [
            f'{template}\n\nImage description: "{row["description"]}"\nYour response:'
            for row in batch
        ]
        return _generate_text(
            model,
            tokenizer,
            prompts,
            device=device,
            max_new_tokens=16,
        )

    def consume(batch: Sequence[dict[str, Any]], outputs: Sequence[str]) -> None:
        append_jsonl(
            paths.step2a,
            (
                {
                    **row,
                    "label": clean_step2a_label(response),
                    "step2a_raw_response": response,
                }
                for row, response in zip(batch, outputs)
            ),
        )

    _run_adaptive_batches(
        pending,
        batch_size,
        generate,
        consume,
        stage_name=f"{criterion}/step2a",
    )


def run_step2b(
    criterion: str,
    paths: PipelinePaths,
    prompt_dir: Path,
    model: Any,
    tokenizer: Any,
    *,
    device: str,
    min_label_frequency: int,
) -> list[str]:
    if paths.step2b.exists():
        try:
            categories = parse_categories(paths.step2b.read_text(encoding="utf-8"))
            LOGGER.info("Step 2b already complete for %s: %s", criterion, categories)
            return categories
        except ValueError as exc:
            classified_ids = completed_image_ids(paths.step3)
            if classified_ids:
                raise RuntimeError(
                    f"Existing step 2b output for {criterion} is invalid, but step 3 "
                    f"already contains {len(classified_ids)} rows. Refusing to regenerate "
                    "the taxonomy because that would mix class vocabularies."
                ) from exc
            LOGGER.warning("Existing step 2b output is invalid; regenerating it.")

    rows = read_jsonl(paths.step2a)
    counts = Counter(clean_step2a_label(str(row.get("label", ""))) for row in rows)
    counts.pop("", None)
    config = CRITERION_CONFIG[criterion]
    retained = {
        label: count
        for label, count in counts.most_common()
        if count >= min_label_frequency
    }
    retained_below_threshold: dict[str, int] = {}
    minimum_observed_labels = int(config["num_classes"])
    if len(retained) < minimum_observed_labels:
        # A fixed-size taxonomy should be grounded in as many observed labels
        # as categories whenever the VLM produced that many distinct labels.
        # Otherwise step 2b has to invent categories merely because a real but
        # infrequent label fell below the generic frequency cutoff.
        for label, count in counts.most_common():
            if label in retained:
                continue
            retained[label] = count
            retained_below_threshold[label] = count
            if len(retained) >= minimum_observed_labels:
                break
        if retained_below_threshold:
            LOGGER.warning(
                "Only %d labels met frequency %d for %s; retaining %d additional "
                "observed labels to ground the %d-category taxonomy: %s",
                len(retained) - len(retained_below_threshold),
                min_label_frequency,
                criterion,
                len(retained_below_threshold),
                minimum_observed_labels,
                retained_below_threshold,
            )
    if not retained:
        raise ValueError(
            f"No step-2a labels meet min frequency {min_label_frequency}; "
            "cannot discover categories."
        )
    template = render_prompt(
        prompt_dir,
        "step2b_prompt.txt",
        {
            "NUM_CLASSES_CLUSTER": config["num_classes"],
            "LEN": len(retained),
            "CRITERION": config["prompt_label"],
        },
    )
    labels_json = json.dumps(retained, ensure_ascii=False)
    response = ""
    categories: list[str] | None = None
    errors: list[str] = []
    for attempt in range(1, 4):
        correction = ""
        if errors:
            correction = (
                "\nThe previous response was invalid because: "
                + errors[-1]
                + " Return exactly ten unique Answer lines."
            )
        prompt = f"{template}\n\nObserved labels and counts: {labels_json}{correction}"
        response = _generate_text(
            model,
            tokenizer,
            [prompt],
            device=device,
            max_new_tokens=768,
            do_sample=attempt > 1,
        )[0]
        try:
            categories = parse_categories(response, expected=10)
            break
        except ValueError as exc:
            errors.append(str(exc))
            LOGGER.warning("Step 2b attempt %d failed: %s", attempt, exc)
    if categories is None:
        raise RuntimeError(f"Step 2b failed after three attempts: {errors}")

    atomic_write_text(paths.step2b, response)
    atomic_write_json(
        paths.step2b_metadata,
        {
            "criterion": criterion,
            "categories": categories,
            "min_label_frequency": min_label_frequency,
            "minimum_observed_labels_requested": minimum_observed_labels,
            "retained_below_frequency_threshold": retained_below_threshold,
            "all_label_counts": dict(counts.most_common()),
            "retained_label_counts": retained,
            "created_at": utc_now(),
        },
    )
    LOGGER.info("Step 2b categories for %s: %s", criterion, categories)
    return categories


def run_step3(
    criterion: str,
    paths: PipelinePaths,
    prompt_dir: Path,
    categories: Sequence[str],
    model: Any,
    tokenizer: Any,
    *,
    device: str,
    batch_size: int,
) -> None:
    source = read_jsonl(paths.step1)
    completed = completed_image_ids(paths.step3)
    pending = [row for row in source if str(row["image_id"]) not in completed]
    if not pending:
        LOGGER.info("Step 3 already complete for %s (%d rows).", criterion, len(completed))
        return
    config = CRITERION_CONFIG[criterion]
    template = render_prompt(
        prompt_dir,
        "step3_prompt.txt",
        {
            "CRITERION": config["prompt_label"],
            "CLASSES": json.dumps(list(categories), ensure_ascii=False),
        },
    )

    def generate(batch: Sequence[dict[str, Any]]) -> list[str]:
        prompts = [
            f'{template}\n\nImage description: "{row["description"]}"\nYour response:'
            for row in batch
        ]
        return _generate_text(
            model,
            tokenizer,
            prompts,
            device=device,
            max_new_tokens=24,
        )

    def consume(batch: Sequence[dict[str, Any]], outputs: Sequence[str]) -> None:
        rows = []
        for row, response in zip(batch, outputs):
            category, match_method = match_category(response, categories)
            rows.append(
                {
                    **row,
                    "class": category,
                    "step3_raw_response": response,
                    "step3_match_method": match_method,
                }
            )
        append_jsonl(paths.step3, rows)

    _run_adaptive_batches(
        pending,
        batch_size,
        generate,
        consume,
        stage_name=f"{criterion}/step3",
    )


def _taxonomy_metrics(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        v_measure_score,
    )
    from sklearn.metrics.cluster import contingency_matrix

    contingency = contingency_matrix(truth, predicted)
    row_indices, column_indices = linear_sum_assignment(-contingency)
    matched = int(contingency[row_indices, column_indices].sum())
    true_labels = sorted(set(truth))
    predicted_labels = sorted(set(predicted))
    hungarian_mapping = {
        predicted_labels[column]: true_labels[row]
        for row, column in zip(row_indices.tolist(), column_indices.tolist())
    }
    return {
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ari": float(adjusted_rand_score(truth, predicted)),
        "homogeneity": float(homogeneity_score(truth, predicted)),
        "completeness": float(completeness_score(truth, predicted)),
        "v_measure": float(v_measure_score(truth, predicted)),
        "hungarian_accuracy": float(matched / len(truth)) if truth else 0.0,
        "hungarian_matched_count": matched,
        "hungarian_predicted_to_true": hungarian_mapping,
    }


def validate_criterion(
    records: Sequence[Clevr4Record],
    criterion: str,
    paths: PipelinePaths,
    *,
    expected_samples: int | None,
) -> dict[str, Any]:
    """Compare discovered labels with every true CLEVR-4 taxonomy."""

    rows = read_jsonl(paths.step3)
    by_id = {str(row["image_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate image ids in {paths.step3}.")
    if expected_samples is not None and len(by_id) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} step-3 rows for {criterion}, found {len(by_id)}."
        )
    missing = [record.image_id for record in records if record.image_id not in by_id]
    if missing:
        raise ValueError(f"Step 3 is missing {len(missing)} records; first: {missing[:5]}")

    ordered_records = [record for record in records if record.image_id in by_id]
    predicted = [str(by_id[record.image_id].get("class", "Unknown")) for record in ordered_records]
    comparisons: dict[str, dict[str, Any]] = {}
    for taxonomy in CLEVR4_CRITERIA:
        truth = [str(getattr(record, taxonomy)) for record in ordered_records]
        comparisons[taxonomy] = _taxonomy_metrics(truth, predicted)

    target_nmi = comparisons[criterion]["nmi"]
    best_nmi = max(metrics["nmi"] for metrics in comparisons.values())
    methods = Counter(
        str(by_id[record.image_id].get("step3_match_method", "missing"))
        for record in ordered_records
    )
    true_target = [str(getattr(record, criterion)) for record in ordered_records]
    exact_semantic_accuracy = sum(
        _normalise_label(prediction) == _normalise_label(truth)
        for prediction, truth in zip(predicted, true_target)
    ) / len(predicted)
    report: dict[str, Any] = {
        "criterion": criterion,
        "sample_count": len(predicted),
        "predicted_unique_count": len(set(predicted)),
        "predicted_categories": sorted(set(predicted)),
        "true_unique_count": len(set(true_target)),
        "true_categories": sorted(set(true_target)),
        "exact_semantic_accuracy": exact_semantic_accuracy,
        "match_methods": dict(methods),
        "unknown_count": sum(value == "Unknown" for value in predicted),
        "taxonomy_comparisons": comparisons,
        "target_is_best_by_nmi": bool(target_nmi >= best_nmi - 1e-12),
        "best_matching_taxonomy": max(
            comparisons, key=lambda name: comparisons[name]["nmi"]
        ),
        "ground_truth_usage": "evaluation_only_after_step3",
        "created_at": utc_now(),
    }
    atomic_write_json(paths.validation, report)
    return report


def build_alignment_report(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    reports: dict[str, Any] = {}
    missing: list[str] = []
    for criterion in CLEVR4_CRITERIA:
        path = criterion_paths(root, criterion).validation
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as handle:
            reports[criterion] = json.load(handle)
    if missing:
        raise FileNotFoundError("Missing criterion validation reports: " + ", ".join(missing))
    target_alignment_passed = all(
        report.get("best_matching_taxonomy") == criterion
        and report.get("target_is_best_by_nmi")
        for criterion, report in reports.items()
    )
    combined = {
        "dataset": "clevr_4_10k_v1",
        "true_criteria": list(CLEVR4_CRITERIA),
        "criterion_reports": reports,
        "all_criteria_best_match_their_true_taxonomy": target_alignment_passed,
        "created_at": utc_now(),
    }
    atomic_write_json(root / "semantic_alignment_report.json", combined)
    return combined


def _manifest_base(args: argparse.Namespace, criterion: str, records: Sequence[Clevr4Record]) -> dict[str, Any]:
    try:
        import torch

        runtime = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count_visible": torch.cuda.device_count(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        runtime = {"torch": None}
    try:
        import transformers

        runtime["transformers"] = transformers.__version__
    except ImportError:
        runtime["transformers"] = None
    try:
        from importlib.metadata import version

        runtime["flash_attn"] = version("flash-attn")
    except Exception:
        runtime["flash_attn"] = None
    prompt_dir = Path(args.prompt_dir).expanduser().resolve()
    annotation_path = Path(records[0].image).parent.parent / "clevr_4_annots.json"
    source_hashes = {"pipeline": sha256_file(__file__)}
    if "internvl" in str(args.vlm_model).lower():
        from partitioning.vlm import internvl_adapter

        source_hashes["internvl_adapter"] = sha256_file(internvl_adapter.__file__)
    return {
        "dataset": "clevr_4_10k_v1",
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "sample_count": len(records),
        "criterion": criterion,
        "criterion_config": CRITERION_CONFIG[criterion],
        "true_criteria": list(CLEVR4_CRITERIA),
        "vlm_model": args.vlm_model,
        "vlm_revision": args.vlm_revision,
        "llm_model": args.llm_model,
        "llm_revision": args.llm_revision,
        "device": args.device,
        "vlm_batch_size_requested": args.vlm_batch_size,
        "llm_batch_size_requested": args.llm_batch_size,
        "max_pixels": args.max_pixels,
        "internvl_max_tiles": getattr(args, "internvl_max_tiles", 12),
        "vlm_attention_backend": (
            "flash_attention_2"
            if "internvl" in str(args.vlm_model).lower()
            and runtime["flash_attn"] is not None
            else "model_default"
        ),
        "llm_attention_backend": (
            "flash_attention_2"
            if runtime["flash_attn"] is not None
            else "sdpa"
        ),
        "min_label_frequency": args.min_label_frequency,
        "seed": args.seed,
        "source_sha256": source_hashes["pipeline"],
        "source_sha256s": source_hashes,
        "annotation_sha256": sha256_file(annotation_path),
        "prompt_sha256": {
            name: sha256_file(prompt_dir / name)
            for name in (
                "step1_prompt.txt",
                "step2a_prompt.txt",
                "step2b_prompt.txt",
                "step3_prompt.txt",
            )
        },
        "runtime": runtime,
        "ground_truth_usage": "evaluation_only_after_step3",
        "started_at": utc_now(),
        "status": "running",
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any] | None:
    criterion = args.criterion
    if criterion not in CLEVR4_CRITERIA:
        raise ValueError(f"Unknown CLEVR-4 criterion: {criterion}")
    if args.max_pixels <= 0:
        args.max_pixels = DEFAULT_MAX_PIXELS[criterion]
    records = load_clevr4_records(
        args.dataset_root,
        splits=args.splits,
        expected_samples=args.expected_samples,
    )
    paths = criterion_paths(Path(args.output_root).expanduser().resolve(), criterion)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = Path(args.prompt_dir).expanduser().resolve()
    manifest = _manifest_base(args, criterion, records)
    if paths.manifest.exists():
        try:
            previous = json.loads(paths.manifest.read_text(encoding="utf-8"))
            manifest["original_started_at"] = previous.get(
                "original_started_at", previous.get("started_at")
            )
        except (json.JSONDecodeError, OSError):
            pass
    atomic_write_json(paths.manifest, manifest)

    try:
        if args.stage in {"all", "step1"}:
            LOGGER.info("Loading VLM %s for %s", args.vlm_model, criterion)
            vlm, processor = _load_vlm(
                args.vlm_model,
                args.vlm_revision,
                args.device,
                args.max_pixels,
                internvl_max_tiles=args.internvl_max_tiles,
            )
            run_step1(
                records,
                criterion,
                paths,
                prompt_dir,
                vlm,
                processor,
                device=args.device,
                batch_size=args.vlm_batch_size,
            )
            del vlm, processor
            _torch_cleanup()

        categories: list[str] | None = None
        if args.stage in {"all", "step2a", "step2b", "step3"}:
            LOGGER.info("Loading LLM %s for %s", args.llm_model, criterion)
            llm, tokenizer = _load_llm(args.llm_model, args.llm_revision, args.device)
            if args.stage in {"all", "step2a"}:
                run_step2a(
                    criterion,
                    paths,
                    prompt_dir,
                    llm,
                    tokenizer,
                    device=args.device,
                    batch_size=args.llm_batch_size,
                )
            if args.stage in {"all", "step2b"}:
                categories = run_step2b(
                    criterion,
                    paths,
                    prompt_dir,
                    llm,
                    tokenizer,
                    device=args.device,
                    min_label_frequency=args.min_label_frequency,
                )
            if args.stage in {"all", "step3"}:
                if categories is None:
                    categories = parse_categories(paths.step2b.read_text(encoding="utf-8"))
                run_step3(
                    criterion,
                    paths,
                    prompt_dir,
                    categories,
                    llm,
                    tokenizer,
                    device=args.device,
                    batch_size=args.llm_batch_size,
                )
            del llm, tokenizer
            _torch_cleanup()

        report = None
        if args.stage in {"all", "validate"}:
            report = validate_criterion(
                records,
                criterion,
                paths,
                expected_samples=args.expected_samples,
            )
        manifest.update(status="complete", completed_at=utc_now())
        atomic_write_json(paths.manifest, manifest)
        return report
    except BaseException as exc:
        manifest.update(status="failed", failed_at=utc_now(), error=repr(exc))
        atomic_write_json(paths.manifest, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    default_prompt_dir = Path(__file__).resolve().parent / "data" / "clevr4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--criterion", choices=CLEVR4_CRITERIA, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "step1", "step2a", "step2b", "step3", "validate"),
        default="all",
    )
    parser.add_argument("--prompt-dir", type=Path, default=default_prompt_dir)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm-revision", default=DEFAULT_VLM_REVISION)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-revision", default=DEFAULT_LLM_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlm-batch-size", type=int, default=8)
    parser.add_argument("--llm-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help=(
            "Visual pixel budget. Zero selects a criterion-aware default: native-budget "
            "for texture and 100k pixels for shape, color, and count."
        ),
    )
    parser.add_argument("--min-label-frequency", type=int, default=31)
    parser.add_argument(
        "--internvl-max-tiles",
        type=int,
        default=12,
        help="Maximum dynamic image tiles for InternVL models; ignored by Qwen VL.",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=10_531,
        help="Official 10k-v1 archive size (the release contains 10,531 images).",
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass
    report = run_pipeline(args)
    if report is not None:
        LOGGER.info("Validation report: %s", json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
