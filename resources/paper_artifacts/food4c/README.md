# Food-4c

The Food-4c human annotations come from Liu et al., *Organizing Unstructured Image Collections using Natural Language* (arXiv:2410.05217), and are not redistributed here. The Food101 criterion labels of the semantic pipeline are in `resources/vlm_labels/food101/`.

`redundancy.py` reproduces the redundancy comparison of Appendix F: the pairwise AMI between the five Food101 criteria of the pipeline (mean 0.032, maximum 0.071, over the 75,750 training images) and between the four human Food-4c taxonomies (mean 0.360, maximum 0.625, over all 101,000 images).

```bash
python resources/paper_artifacts/food4c/redundancy.py /path/to/annotation_food4c.json
```

`agreement.py` reproduces the agreement between the pipeline's cuisine labels and the human cuisine labels: AMI 0.503, ARI 0.465 and direct agreement on 73.0% of 68,250 images, with the 15 human cuisine values coarsened to the pipeline's five. It also prints the AMI of the pipeline's other criteria with the same human labels (0.092 for preparation, 0.018 for texture). The coarsening is documented at the top of the script.

```bash
python resources/paper_artifacts/food4c/agreement.py /path/to/annotation_food4c.json
```
