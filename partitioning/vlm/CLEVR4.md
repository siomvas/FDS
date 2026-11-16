# Clevr-4 validation of the semantic pipeline

This is the code behind the Clevr-4 experiments of the paper (Section 3.5, Figure 1, Table 7 and Appendix F). Its outputs are in `resources/paper_artifacts/clevr4/`.

Clevr-4 (Vaze et al., NeurIPS 2023) annotates every image under four equally valid taxonomies: texture, shape, color and count, with ten classes each. The experiment runs the four-stage semantic pipeline once per criterion on the 10,531 images of the official `clevr_4_10k_v1.zip` release (8,424 train and 2,107 validation) and checks that the partition of each criterion agrees with its own taxonomy far more than with the other three.

| File | Role |
|---|---|
| `partitioning/vlm/clevr4_pipeline.py` | the four stages, resumable, one criterion per process |
| `partitioning/vlm/internvl_adapter.py` | loading and batched inference for InternVL checkpoints |
| `partitioning/vlm/data/clevr4/` | the four prompt templates |
| `scripts/launch_clevr4_semantic_pipeline.py` | one worker per GPU, one criterion each |
| `scripts/run_clevr4_large_models.py` | all criteria with InternVL3.5-38B and Llama 3.3-70B, each checkpoint loaded once |
| `scripts/run_clevr4_step1_shard.py` | splits the unfinished captions of one criterion across idle GPUs and merges them |
| `scripts/reset_clevr4_text_stages.py` | archives the text stages and resets a run to its captions |
| `scripts/verify_clevr4_semantic_run.py` | strict verification of a finished output tree |
| `scripts/run_clevr4_axis_recovery.py` | the embedding experiment of Figure 1 |

## The four stages

1. A vision-language model describes each image with respect to one criterion.
2. A language model reduces each description to a short free-form label.
3. The language model consolidates the observed labels into ten categories.
4. The language model assigns every description to one of those categories.

The annotations are read only after the last stage, to compute NMI, AMI, ARI, V-measure and Hungarian-matched accuracy against the four taxonomies. No prompt contains a class value of any taxonomy, and the verifier rejects a run whose prompts or guidance do.

## Provenance of the released labels

The three files that ran the text stages of both released runs are in this repository byte for byte, and their SHA-256 digests are recorded under `source_sha256s` in every `run_manifest.json`:

| File | SHA-256 |
|---|---|
| `partitioning/vlm/clevr4_pipeline.py` | `847e29baa526ddb0839f4f64b237f2ec893dd37b94e43d00028bcda6b5e2c4f1` |
| `scripts/run_clevr4_large_models.py` | `7f9a8e630acb0aaa8edad225b16baef60b42f796e74251d374f453768b4cd966` |
| `partitioning/vlm/internvl_adapter.py` | `69beb337b550e36d357399781753d2deb847e0e9468ff72eec05b987a33679fa` |

The captions of both runs were written by earlier revisions of `clevr4_pipeline.py`, whose digests are recorded under `stage_source_sha256s.step1`. Two things follow.

- **InternVL3.5-38B and Llama 3.3-70B run.** The caption prompt of that revision is the one in this repository. No stage of the pipeline reads an annotation, but the texture guidance was revised once during the run, after the descriptions produced so far had been clustered and scored against the texture taxonomy. The guidance in `CRITERION_CONFIG` is the revised one.
- **Qwen3-VL-8B and Qwen3-8B run.** Its captions were produced with the original texture guidance, which read: "Describe the visible surface texture or material shared by the foreground objects. Ignore the floor, walls, shadows, and background." Its manifests show the revised guidance under `criterion_config` because they were rewritten when the text stages were rerun. The texture labels of this run therefore do not isolate the effect of the models.

Earlier revisions also passed one example label per criterion to the second stage. The text stages of both runs were rerun without it, so no released label depends on such an example.

## Installation

Install a CUDA build of PyTorch and the pipeline dependencies:

```bash
python -m pip install "transformers>=4.57" "accelerate>=1.0" \
  "safetensors>=0.4" "scikit-learn>=1.3" "scipy>=1.10" \
  torchvision protobuf pillow sentencepiece einops timm
```

## One criterion per GPU

```bash
python -m scripts.launch_clevr4_semantic_pipeline \
  --dataset-root /path/to/clevr_4_10k \
  --output-root /path/to/outputs/clevr4_semantic \
  --gpu-ids 0,1,2,3
```

The defaults are `Qwen/Qwen3-VL-8B-Instruct` for the descriptions and `Qwen/Qwen3-8B` for the text stages, at pinned revisions, which fits one criterion on a 32 GB GPU. Every worker appends JSONL batches and skips completed image IDs when it is rerun. Per-criterion outputs go to `output_<criterion>_10k/`, and the cross-taxonomy report to `semantic_alignment_report.json`.

Texture uses the native image resolution, and the other criteria a resize to about 100,000 pixels. The budget is recorded per criterion in the manifest.

## InternVL3.5-38B and Llama 3.3-70B

`scripts/run_clevr4_large_models.py` runs the same stages while loading each checkpoint once. It pins `OpenGVLab/InternVL3_5-38B` and `meta-llama/Llama-3.3-70B-Instruct` to exact revisions. Caption first, so that missing access to the gated Llama repository cannot interrupt the vision work:

```bash
python -m scripts.run_clevr4_large_models \
  --dataset-root /path/to/clevr_4_10k \
  --output-root /path/to/outputs/clevr4_internvl38b_llama70b \
  --stage caption --vlm-batch-size 4
```

Then, with a Hugging Face account that has accepted the Llama 3.3 license:

```bash
python -m scripts.run_clevr4_large_models \
  --dataset-root /path/to/clevr_4_10k \
  --output-root /path/to/outputs/clevr4_internvl38b_llama70b \
  --stage text --llm-batch-size 32
```

The adapter follows the checkpoint's dynamic high-resolution tiling and its `batch_chat` interface. FlashAttention 2 is used when `flash-attn` is installed, and the language model is spread across the visible GPUs. Backends and package versions are recorded in the manifests. On GPUs that hold a full checkpoint each, captioning can run as two workers with disjoint `--criteria` lists and distinct `--worker-name` values.

## Verifying a run

```bash
python -m scripts.verify_clevr4_semantic_run \
  --dataset-root /path/to/clevr_4_10k \
  --output-root /path/to/outputs/clevr4_semantic
```

The verifier requires exact image-ID coverage without duplicates at stages 1, 2 and 4, ten valid categories, final classes drawn only from them, completed manifests, prompt hashes that match the templates, and a best match of every criterion to its own taxonomy. It writes `artifact_verification.json` and exits nonzero if any check fails. It needs the full output tree of a run, including the descriptions, which are not part of the released artifacts.

## Embedding experiment (Figure 1)

`scripts/run_clevr4_axis_recovery.py` clusters embeddings of the same images with k-means (K = 10) and measures the agreement of the clusters with each taxonomy after the top p principal components are removed. It takes the image order from a finished caption file of the semantic pipeline.

```bash
for mode in protocol extract evaluate controls finalize; do
  python -m scripts.run_clevr4_axis_recovery --mode "$mode" \
    --output-root /path/to/outputs/clevr4_axis_recovery \
    --semantic-step1 /path/to/outputs/clevr4_semantic/output_shape_10k/step1_result.jsonl \
    --annotations /path/to/clevr_4_10k/clevr_4_annots.json \
    --configs dinov2-base mae-base
done
```

The MAE encoder is evaluated on intact images, with its inference mask ratio set to zero.

## Tests

```bash
python -m pytest tests/test_clevr4_semantic_pipeline.py tests/test_clevr4_internvl_adapter.py \
  tests/test_clevr4_large_models.py tests/test_clevr4_step1_shard.py \
  tests/test_clevr4_text_reset.py tests/test_clevr4_artifact_verifier.py
```

The tests run without a GPU and without the dataset.
