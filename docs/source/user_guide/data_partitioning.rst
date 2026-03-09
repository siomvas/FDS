Data Partitioning
=================

FDS provides several splitter strategies under the `partitioning` package:

- `RandomSplitter` for IID baselines
- `DirichletSplitter` for label-based non-IID partitions
- `EmbeddingSplitter` for semantic non-IID partitions
- `MultivariateDirichletSplitter` for multi-criterion heterogeneity

Core Workflow
-------------

1. Build/load a dataset with `data_loading.dataset.BaseDataset`.
2. Construct a splitter with `n_clients`, `heterogeneity`, and `valset_type`.
3. Call `get_client_splits()` and persist via `store_client_splits(...)`.
4. Feed split files to FL data modules via `split_file`.

Split Schema Notes
------------------

- Classification samples use `{"image", "label"}`
- Segmentation samples use `{"image", "mask_path"}` (legacy `label` alias supported)
- Detection samples support annotation-path based schemas
