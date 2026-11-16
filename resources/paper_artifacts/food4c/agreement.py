"""Reproduce the Food-4c agreement of Appendix F (cuisine).

Usage:
    python resources/paper_artifacts/food4c/agreement.py /path/to/annotation_food4c.json

Compares the cuisine labels of the semantic pipeline (resources/vlm_labels/food101)
with the human cuisine labels of Food-4c on the Food101 training images.

Food-4c assigns one of 15 cuisine values to each Food101 class, while the
pipeline settled on five (American, Asian, French, Italian, Mexican). The human
values are therefore coarsened to those five, following the grouping that the
pipeline describes in its own label-set step (output_cuisine_train/step2b_result.txt):
Canadian joins American, Mediterranean cuisines join Italian, Spanish joins
Mexican, and the East, South-East and South Asian cuisines form Asian. Classes
that Food-4c labels "general" or "middle eastern" are left out, which leaves
91 classes and 68,250 images.

Prints the adjusted mutual information, the adjusted Rand index and the direct
agreement between the two labelings, and the adjusted mutual information of the
pipeline's other criteria with the same human labels.
"""

import json
import sys
from pathlib import Path

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

LABEL_ROOT = Path(__file__).resolve().parents[2] / "vlm_labels" / "food101"
OTHER_CRITERIA = ["composition", "formality", "preparation", "texture"]

COARSE_CUISINE = {
    "american": "American",
    "canadian": "American",
    "chinese": "Asian",
    "indian": "Asian",
    "japanese": "Asian",
    "korean": "Asian",
    "thai": "Asian",
    "vietnamese": "Asian",
    "french": "French",
    "greek": "Italian",
    "italian": "Italian",
    "mexican": "Mexican",
    "spanish": "Mexican",
}


def annotation_key(image):
    """Map 'food101/images/<class>/<id>.jpg' to the Food-4c key '<class>_<id>'."""
    parts = Path(image).parts
    return f"{parts[-2]}_{Path(parts[-1]).stem}"


def pipeline_labels(criterion):
    path = LABEL_ROOT / f"output_{criterion}_train" / "step3_result.jsonl"
    with path.open(encoding="utf-8") as handle:
        return {annotation_key(r["image"]): r["class"] for r in map(json.loads, handle)}


def main(annotation_path):
    with open(annotation_path, encoding="utf-8") as handle:
        annotations = json.load(handle)

    cuisine = pipeline_labels("cuisine")
    images = sorted(k for k in cuisine if annotations[k]["food_cuisine"] in COARSE_CUISINE)
    human = [COARSE_CUISINE[annotations[k]["food_cuisine"]] for k in images]
    pipeline = [cuisine[k] for k in images]

    agreement = sum(h == p for h, p in zip(human, pipeline)) / len(images)
    print(f"Images: {len(images)}")
    print(f"Pipeline cuisine against human cuisine: AMI {adjusted_mutual_info_score(human, pipeline):.3f}, "
          f"ARI {adjusted_rand_score(human, pipeline):.3f}, agreement {100 * agreement:.1f}%")
    for criterion in OTHER_CRITERIA:
        other = pipeline_labels(criterion)
        score = adjusted_mutual_info_score(human, [other[k] for k in images])
        print(f"Pipeline {criterion} against human cuisine: AMI {score:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
