"""Reproduce the criterion-redundancy comparison of Appendix F (Food-4c).

Usage:
    python resources/paper_artifacts/food4c/redundancy.py [/path/to/annotation_food4c.json]

Prints the mean and maximum pairwise AMI between the five Food101 criteria of
the semantic pipeline (from resources/vlm_labels/food101) and, when the Food-4c
annotation file is given, between its four human taxonomies.
"""

import itertools
import json
import sys
from pathlib import Path

from sklearn.metrics import adjusted_mutual_info_score

LABEL_ROOT = Path(__file__).resolve().parents[2] / "vlm_labels" / "food101"
CRITERIA = ["composition", "cuisine", "formality", "preparation", "texture"]
HUMAN_TAXONOMIES = ["food_type", "dietary_preference", "course", "food_cuisine"]


def mean_and_max_ami(columns):
    values = [
        adjusted_mutual_info_score(columns[a], columns[b])
        for a, b in itertools.combinations(columns, 2)
    ]
    return sum(values) / len(values), max(values)


def pipeline_redundancy():
    labels = {}
    for criterion in CRITERIA:
        path = LABEL_ROOT / f"output_{criterion}_train" / "step3_result.jsonl"
        with path.open(encoding="utf-8") as handle:
            labels[criterion] = {r["image"]: r["class"] for r in map(json.loads, handle)}
    images = sorted(set.intersection(*(set(d) for d in labels.values())))
    columns = {c: [labels[c][i] for i in images] for c in CRITERIA}
    return len(images), mean_and_max_ami(columns)


def human_redundancy(annotation_path):
    with open(annotation_path, encoding="utf-8") as handle:
        annotations = json.load(handle)
    images = sorted(annotations)
    columns = {t: [annotations[i][t] for i in images] for t in HUMAN_TAXONOMIES}
    return len(images), mean_and_max_ami(columns)


if __name__ == "__main__":
    n, (mean, maximum) = pipeline_redundancy()
    print(f"Pipeline criteria, {n} images: mean pairwise AMI {mean:.3f}, maximum {maximum:.3f}")
    if len(sys.argv) > 1:
        n, (mean, maximum) = human_redundancy(sys.argv[1])
        print(f"Food-4c human taxonomies, {n} images: mean pairwise AMI {mean:.3f}, maximum {maximum:.3f}")
