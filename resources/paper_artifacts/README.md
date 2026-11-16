# Paper artifacts

Artifacts behind the validation experiments of the paper (Section 3.5, Figure 1 and Appendix F). The criterion labels of the four main datasets are in `resources/vlm_labels/`.

## Clevr-4

- `clevr4/labels/internvl35_38b_llama33_70b/`: criterion labels from InternVL3.5-38B and Llama 3.3-70B for the 10,531 Clevr-4 images, one folder per criterion (texture, shape, color, count). Each holds `step3_result.jsonl` (image path relative to the Clevr-4 root, and label), `step2b_result.txt` (the label set), `run_manifest.json` (models and revisions, criterion guidance, prompt hashes and seed) and `validation.json` (agreement with the four taxonomies, computed after labelling). `prompts/` holds the prompts of the run. No stage of the pipeline reads an annotation, but the texture guidance of this run was revised once during the run, after the descriptions produced so far had been clustered and scored against the texture taxonomy.
- `clevr4/labels/qwen3vl_8b_qwen3_8b/`: the same for Qwen3-VL-8B and Qwen3-8B. The captions of this run were produced with the original texture guidance, so its texture labels do not isolate the effect of the models. Its manifests show the revised guidance under `criterion_config` because they were rewritten when the text stages were rerun, and the revision of the pipeline that wrote the captions is identified by `stage_source_sha256s.step1`.
- `clevr4/analysis/agreement_matrix/`: NMI, AMI and ARI between each criterion partition and each taxonomy, and the two controls (criterion against criterion, taxonomy against taxonomy), for both runs (Table 7).
- `clevr4/analysis/axis_recovery/`: the data of Figure 1, NMI between k-means partitions (K = 10) of DINOv2-base and MAE-base embeddings and each taxonomy after removing the top p principal components (`results.csv`), with the client sizes of every partition (`cluster_sizes.json`), PCA metadata, protocol and summary.

The code that produced these artifacts, and its provenance, are described in `partitioning/vlm/CLEVR4.md`.

The Clevr-4 images and annotations come from Vaze et al., *No Representation Rules Them All in Category Discovery* (NeurIPS 2023), and are not redistributed here.

## Food-4c

See `food4c/README.md`.
