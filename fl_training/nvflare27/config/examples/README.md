# FDS Experiment Configuration Examples

This directory contains example YAML experiment configurations for FDS federated learning.

## Usage

1. **Copy example config:**
   ```bash
   cp classification_example.yaml my_experiment.yaml
   ```

2. **Edit configuration:**
   - Update `experiment.name` to your experiment name
   - Set `experiment.dataset_splits` to your DatasetSplits JSON path
   - Adjust hyperparameters in `training` section

3. **Generate NVFlare job config:**
   ```python
   from fl_training.nvflare27.config import JobConfigGenerator

   generator = JobConfigGenerator()
   job_config_path = generator.generate(
       yaml_config_path="my_experiment.yaml",
       output_dir="./jobs",
       client_script_path="train.py"
   )
   print(f"Job config generated: {job_config_path}")
   ```

4. **Run FL experiment** (in Phase 8):
   ```bash
   nvflare simulator ./jobs/my_experiment_job_config.json
   ```

## Configuration Sections

### `experiment`
- `name`: Unique experiment identifier
- `task_type`: "classification", "segmentation", or "detection"
- `dataset`: Dataset name
- `dataset_splits`: Path to DatasetSplits JSON from partitioning

### `federated_learning`
- `num_rounds`: Total FL rounds
- `num_clients`: Number of clients
- `aggregator`: Aggregation strategy ("fedavg", "fedprox", "scaffold")
- `min_clients`: Minimum clients before starting round
- `wait_time`: Wait time after min_clients (seconds)

### `training`
- `model`: Model architecture (e.g., "resnet18", "deeplabv3_resnet50", "yolov5s")
- `local_epochs`: Epochs per round per client
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `amp_enabled`: Use automatic mixed precision
- `grad_tracking`: Enable gradient diversity tracking

### `resources`
- `client_gpu_mem`: GPU memory per client (informational)
- `server_cpu`: Server CPU count (informational)

## Task-Specific Notes

**Classification:**
- Typical batch_size: 32-64
- Typical gpu_mem: 4GB

**Segmentation:**
- Smaller batch_size: 8-16 (pixel-wise dense predictions)
- More gpu_mem: 8GB

**Detection:**
- Smallest batch_size: 4-8 (bounding box computations)
- Most gpu_mem: 10GB (YOLO/DETR models)

## Next Steps

After generating job configs, proceed to:
- **Phase 3**: Implement ClassificationLearner
- **Phase 4**: Implement SegmentationLearner
- **Phase 5**: Implement DetectionLearner
- **Phase 8**: Run end-to-end experiments
