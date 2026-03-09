Evaluation
==========

The `evaluation` package computes distributional and gradient-diversity metrics
for partition quality analysis.

Metric families:

- Label-distribution metrics
- Feature-distribution metrics
- Gradient-diversity metrics

Typical workflow:

1. Prepare split files and optional embeddings.
2. Compute metrics with functions in `evaluation.metrics`.
3. Summarize/visualize with scripts such as `scripts/generate_report.py`.

See API reference: :doc:`../api/evaluation`
