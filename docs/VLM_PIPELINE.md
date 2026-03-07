## VLM-Based Partitioning

Located in `partitioning/vlm/`. Uses a multi-step LLM + VLM pipeline to annotate images by a chosen visual criterion (e.g. main object, color, location), then groups images into client partitions based on those annotations.

*Pipeline based on [ICTC](https://github.com/sehyunkwon/ICTC) by Kwon et al.*

### Pipeline steps

| Script | Description |
|--------|-------------|
| `step1.py` | VLM annotates each image with a raw description for the chosen criterion |
| `step2a.py` | LLM proposes a set of category labels from the annotations |
| `step2b.py` | LLM assigns each image to one of the proposed categories |
| `step3.py` | Groups images by category to form client partitions |
| `full_pipeline.py` | Runs all steps end-to-end |

### Usage

```bash
python partitioning/vlm/full_pipeline.py \
  --dataset <dataset> \
  --criteria <criterion> \
  --exp_name <name> \
  --llama_ver <llm_model> \
  --vlm_model <vlm_model>
```

### Arguments

| Argument | Default | Choices |
|----------|---------|---------|
| `--dataset` | `tiny_imagenet` | `tiny_imagenet`, `food101`, `ade20k`, `pascalvoc` |
| `--criteria` | `main_object` | see table below |
| `--exp_name` | `test` | any string |
| `--llama_ver` | `LLaMA 3.3 70B` | `LLaMA 3.3 70B`, `LLaMA 3.1 8B Instruct` |
| `--vlm_model` | `Qwen3-30B` | `Qwen3-8B`, `Qwen3-30B`, `InternVL3-8B`, `InternVL3-38B`, `MiniCPM-2.6` |

### Criteria per dataset

| Dataset | Available criteria |
|---------|-------------------|
| Tiny ImageNet | `main_object`, `size`, `time`, `color`, `location` |
| Food-101 | `main_object`, `size`, `color`, `angle`, `lighting` |
| ADE20K | `main_object`, `scene`, `color`, `light`, `size`, `camera` |
| PascalVOC | `main_object`, `location`, `color`, `light`, `size`, `camera` |

### Output

Results are saved to:
```
partitioning/vlm/data/<dataset>/output_<criteria>_<exp_name>/
```

Each step writes its own output file (`step1_result.jsonl`, `step2a_result.jsonl`, etc.).

### Setup

Before running, edit the following files to set your local paths:

- **`partitioning/vlm/utils/util_loader.py`** — paths to VLM/LLM model weights and dataset root
- **`partitioning/vlm/utils/argument.py`** — `args.home_path` (line 91) — set to your local repo path

Prompts for each dataset and step are `.txt` files located at:
```
partitioning/vlm/data/<dataset>/step{1,2a,2b,3}_prompt.txt
```
