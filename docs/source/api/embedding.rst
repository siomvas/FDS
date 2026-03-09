Embedding Package
=================

Embedding utilities are implemented under:

- `embedding.embedding_extractors` (model registry + extractors)
- `embedding.cache_manager` (persistent embedding cache)

Primary Python entrypoints:

- `ModelRegistry.create(...)`
- `EmbeddingExtractor.extract_embeddings(...)`
- `CacheManager.get(...)`
- `CacheManager.put(...)`

For runnable examples, see user guides:

- `user_guide/embeddings`
