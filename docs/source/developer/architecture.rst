Architecture
============

High-level subsystems:

- `data_loading`: dataset registry, dataset classes, FL data modules
- `partitioning`: splitter implementations and split serialization
- `embedding`: feature extraction and caching
- `evaluation`: partition diagnostics and metrics
- `fl_training`: model factory, engines, FL execution entrypoints

Primary execution path:

1. Load dataset(s) via `BaseDataset`.
2. Build client splits with a `BaseDatasetSplitter` implementation.
3. Initialize task-appropriate data module with split file.
4. Create model through `fl_training.models.factory`.
5. Run training/evaluation engine for selected task.
