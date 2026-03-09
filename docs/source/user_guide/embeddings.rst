Embeddings
==========

Embedding-based partitioning and analysis are available through the `embedding`
package and the `EmbeddingSplitter`.

Available extractor families include:

- Inception
- DINOv2
- CLIP / SigLIP
- MAE
- Lightweight baselines (raw pixels, sparse projection, untrained autoencoder)

Typical pipeline:

1. Choose extractor/model variant.
2. Configure pooling and cache policy.
3. Generate embeddings.
4. Use embeddings for semantic clustering in partitioning/evaluation workflows.

See API reference: :doc:`../api/embedding`
