"""End-to-end experiment orchestration for FDS federated learning.

This script provides a unified CLI for running FL experiments from YAML configuration
to results collection. It orchestrates the complete workflow:
    1. Load YAML experiment configuration
    2. Generate NVFlare job configuration
    3. Execute FL training (simulator or production mode)
    4. Collect and organize results (TensorBoard, models, summaries)

Usage:
    # Run in simulator mode (local testing with 10 clients)
    python run_experiment.py \\
        --config classification_experiment.yaml \\
        --mode simulator \\
        --workspace ./nvflare_workspace

    # Production mode (multi-node HPC deployment)
    python run_experiment.py \\
        --config classification_experiment.yaml \\
        --mode production \\
        --workspace /shared/nvflare_workspace

    # Override number of clients from YAML
    python run_experiment.py \\
        --config my_experiment.yaml \\
        --mode simulator \\
        --num_clients 5

Architecture:
    - Automates YAML → NVFlare JSON → FL execution → Results pipeline
    - Supports both simulator (single-machine) and production (multi-node) modes
    - Organizes results with timestamps for experiment tracking
    - Integrates with existing config system (FDSExperimentConfig, JobConfigGenerator)
    - Preserves TensorBoard logs, model checkpoints, and experiment metadata

Dependencies:
    - fl_training.nvflare27.config.experiment_config
    - fl_training.nvflare27.config.config_generator
    - nvflare (installed separately: pip install nvflare~=2.7.0)
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directories to path for FDS imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fl_training.nvflare27.config.experiment_config import FDSExperimentConfig
from fl_training.nvflare27.config.config_generator import JobConfigGenerator


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with experiment configuration
    """
    parser = argparse.ArgumentParser(
        description="FDS Federated Learning Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classification experiment in simulator mode
  python run_experiment.py \\
      --config config/examples/classification_example.yaml \\
      --mode simulator \\
      --workspace ./nvflare_workspace

  # Segmentation experiment with custom client count
  python run_experiment.py \\
      --config my_segmentation_exp.yaml \\
      --mode simulator \\
      --num_clients 5

  # Production deployment (multi-node HPC)
  python run_experiment.py \\
      --config production_exp.yaml \\
      --mode production \\
      --workspace /shared/nvflare_prod
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["simulator", "production"],
        help="Execution mode: simulator (local testing) or production (multi-node deployment)"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="./nvflare_workspace",
        help="NVFlare workspace directory (default: ./nvflare_workspace)"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Override number of clients from YAML config (optional)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Base directory for experiment results (default: ./results)"
    )

    return parser.parse_args()


def check_nvflare_installed() -> bool:
    """Check if NVFlare is installed and accessible.

    Returns:
        bool: True if nvflare command is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["nvflare", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def print_nvflare_install_instructions():
    """Print installation instructions for NVFlare."""
    print("\n" + "=" * 80)
    print("ERROR: NVFlare not found")
    print("=" * 80)
    print("\nNVFlare is required to run FL experiments.")
    print("\nInstallation:")
    print("  pip install nvflare~=2.7.0")
    print("\nVerify installation:")
    print("  nvflare --version")
    print("\nFor more information, see:")
    print("  https://nvidia.github.io/NVFlare/")
    print("=" * 80 + "\n")


def load_config(config_path: str) -> FDSExperimentConfig:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        FDSExperimentConfig: Loaded experiment configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
        ValueError: If required configuration fields are missing
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Example configs available at: fl_training/nvflare27/config/examples/"
        )

    try:
        config = FDSExperimentConfig.from_yaml(config_path)
        return config
    except Exception as e:
        print(f"\nERROR: Failed to load configuration from {config_path}")
        print(f"Reason: {e}")
        print("\nPlease check:")
        print("  1. YAML syntax is valid")
        print("  2. All required fields are present (experiment, federated_learning, training)")
        print("  3. Field values match expected types")
        print("\nSee examples at: fl_training/nvflare27/config/examples/")
        raise


def generate_job_config(
    config: FDSExperimentConfig,
    yaml_config_path: str,
    workspace: Path
) -> Path:
    """Generate NVFlare job configuration from YAML config.

    Args:
        config: Loaded experiment configuration
        yaml_config_path: Path to source YAML config
        workspace: NVFlare workspace directory

    Returns:
        Path: Generated job config JSON file path

    Raises:
        ValueError: If job template not found for task type
        FileNotFoundError: If client script (train.py) not found
    """
    # Determine client script path (relative to job config location)
    train_script = Path(__file__).parent / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"Client training script not found: {train_script}\n"
            f"Expected at: fl_training/nvflare27/train.py"
        )

    # Create job config output directory
    jobs_dir = workspace / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Generate job config
    generator = JobConfigGenerator()
    try:
        job_config_path = generator.generate(
            yaml_config_path=yaml_config_path,
            output_dir=str(jobs_dir),
            client_script_path=str(train_script)
        )
        return job_config_path
    except Exception as e:
        print(f"\nERROR: Failed to generate NVFlare job config")
        print(f"Reason: {e}")
        raise


def run_simulator(
    job_config_path: Path,
    workspace: Path,
    num_clients: int,
    num_threads: Optional[int] = None
):
    """Run FL experiment in NVFlare simulator mode.

    Args:
        job_config_path: Path to generated job config JSON
        workspace: NVFlare workspace directory
        num_clients: Number of client instances to simulate
        num_threads: Number of concurrent threads (defaults to num_clients)

    Raises:
        subprocess.CalledProcessError: If nvflare simulator command fails
    """
    if num_threads is None:
        num_threads = num_clients

    # Build nvflare simulator command
    cmd = [
        "nvflare",
        "simulator",
        str(job_config_path),
        "-w", str(workspace),
        "-n", str(num_clients),
        "-t", str(num_threads)
    ]

    print(f"\nRunning command: {' '.join(cmd)}\n")
    print("-" * 80)

    # Run simulator (output streams to console in real-time)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print("ERROR: NVFlare simulator failed")
        print("=" * 80)
        print(f"\nCommand: {' '.join(cmd)}")
        print(f"Exit code: {e.returncode}")
        print("\nCheck the logs above for error details.")
        raise


def print_production_instructions(
    job_config_path: Path,
    workspace: Path,
    config: FDSExperimentConfig
):
    """Print instructions for production deployment.

    Args:
        job_config_path: Path to generated job config
        workspace: NVFlare workspace directory
        config: Experiment configuration
    """
    print("\n" + "=" * 80)
    print("PRODUCTION MODE: Manual Deployment Required")
    print("=" * 80)
    print("\nJob configuration generated successfully:")
    print(f"  {job_config_path}")
    print("\nTo deploy this experiment on multi-node HPC cluster:")
    print("\n1. **Provision NVFlare infrastructure:**")
    print("   nvflare provision -p project.yml")
    print("\n2. **Deploy job to server:**")
    print(f"   nvflare job submit {job_config_path}")
    print("\n3. **Start server:**")
    print("   cd startup && ./start.sh")
    print("\n4. **Start clients on compute nodes:**")
    print(f"   # On each of {config.federated_learning.num_clients} nodes:")
    print("   cd startup/site-N && ./start.sh")
    print("\n5. **Monitor progress:**")
    print("   nvflare job list")
    print("   nvflare job show <job_id>")
    print("\nFor detailed production setup, see:")
    print("  https://nvidia.github.io/NVFlare/user_guide/production_deployment.html")
    print("=" * 80 + "\n")


def collect_results(
    workspace: Path,
    config: FDSExperimentConfig,
    results_base_dir: Path,
    mode: str
) -> Path:
    """Collect and organize experiment results.

    Creates timestamped results directory with:
        - TensorBoard logs
        - Final model checkpoint
        - Experiment summary JSON

    Args:
        workspace: NVFlare workspace directory
        config: Experiment configuration
        results_base_dir: Base directory for results
        mode: Execution mode (simulator or production)

    Returns:
        Path: Results directory for this experiment

    Raises:
        FileNotFoundError: If expected result files not found in workspace
    """
    # Create results directory: results/{exp_name}/{timestamp}/
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = results_base_dir / config.experiment.name / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCollecting results to: {results_dir}")

    # Locate workspace results (different paths for simulator vs production)
    if mode == "simulator":
        workspace_results = workspace / "simulate_job"
    else:
        # Production mode: job results are in workspace/server/{job_id}
        # For now, just document the path
        print("\n  Production results location: {workspace}/server/<job_id>/")
        return results_dir

    if not workspace_results.exists():
        print(f"\nWARNING: Workspace results not found at {workspace_results}")
        print("This may be expected if the job hasn't completed yet.")
        return results_dir

    # Copy TensorBoard logs
    tb_source = workspace_results / "tb_events"
    if tb_source.exists():
        tb_dest = results_dir / "tensorboard"
        shutil.copytree(tb_source, tb_dest, dirs_exist_ok=True)
        print(f"  ✓ TensorBoard logs: {tb_dest}")
    else:
        print(f"  ⚠ TensorBoard logs not found at {tb_source}")

    # Copy final model checkpoint
    model_source = workspace_results / "app_server"
    if model_source.exists():
        model_dest = results_dir / "model"
        model_dest.mkdir(exist_ok=True)
        # Find final model checkpoint (glob pattern for FL_global_model.pt or similar)
        model_files = list(model_source.glob("*.pt"))
        if model_files:
            for model_file in model_files:
                shutil.copy2(model_file, model_dest / model_file.name)
            print(f"  ✓ Model checkpoint: {model_dest}")
        else:
            print(f"  ⚠ Model checkpoint not found in {model_source}")
    else:
        print(f"  ⚠ Model directory not found at {model_source}")

    # Save experiment summary
    summary = {
        "experiment_name": config.experiment.name,
        "task_type": config.experiment.task_type,
        "dataset": config.experiment.dataset,
        "timestamp": timestamp,
        "mode": mode,
        "configuration": config.to_dict(),
        "results_location": str(results_dir)
    }

    summary_file = results_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Experiment summary: {summary_file}")

    return results_dir


def main():
    """Main execution flow for experiment runner.

    Workflow:
        1. Parse CLI arguments
        2. Check NVFlare installation
        3. Load YAML configuration
        4. Generate NVFlare job config
        5. Execute experiment (simulator or production)
        6. Collect results

    Exits with non-zero status on errors.
    """
    # Parse arguments
    args = parse_args()

    print("=" * 80)
    print("FDS Federated Learning Experiment Runner")
    print("=" * 80)

    # Step 1: Check NVFlare installation
    print("\n[1/5] Checking NVFlare installation...")
    if not check_nvflare_installed():
        print_nvflare_install_instructions()
        sys.exit(1)
    print("  ✓ NVFlare installed")

    # Step 2: Load experiment configuration
    print(f"\n[2/5] Loading experiment config: {args.config}")
    try:
        config = load_config(args.config)
    except Exception:
        sys.exit(1)

    print(f"  ✓ Experiment: {config.experiment.name}")
    print(f"  ✓ Task type: {config.experiment.task_type}")
    print(f"  ✓ Dataset: {config.experiment.dataset}")

    # Override num_clients if provided
    if args.num_clients is not None:
        config.federated_learning.num_clients = args.num_clients
        print(f"  ✓ Overriding num_clients: {args.num_clients}")

    # Step 3: Generate NVFlare job configuration
    print(f"\n[3/5] Generating NVFlare job config...")
    workspace = Path(args.workspace)
    try:
        job_config_path = generate_job_config(config, args.config, workspace)
    except Exception:
        sys.exit(1)

    print(f"  ✓ Job config written to: {job_config_path}")

    # Step 4: Execute FL experiment
    if args.mode == "simulator":
        print(f"\n[4/5] Running FL simulation ({config.federated_learning.num_clients} clients, {config.federated_learning.num_rounds} rounds)...")
        try:
            run_simulator(
                job_config_path,
                workspace,
                config.federated_learning.num_clients
            )
        except subprocess.CalledProcessError:
            sys.exit(1)
        print("-" * 80)
        print("  ✓ Simulation complete")

        # Step 5: Collect results
        print("\n[5/5] Collecting results...")
        results_dir = collect_results(
            workspace,
            config,
            Path(args.results_dir),
            args.mode
        )

        print("\n" + "=" * 80)
        print("✓ Experiment complete!")
        print("=" * 80)
        print(f"\nResults saved to: {results_dir}")
        print("\nView TensorBoard logs:")
        print(f"  tensorboard --logdir {results_dir}/tensorboard")
        print("\nExperiment summary:")
        print(f"  cat {results_dir}/experiment_summary.json")
        print("=" * 80 + "\n")

    else:  # production mode
        print(f"\n[4/5] Production deployment mode")
        print_production_instructions(job_config_path, workspace, config)
        print("\n[5/5] Skipping automatic result collection (manual deployment)")
        print("\nResults will be available at:")
        print(f"  {workspace}/server/<job_id>/")


if __name__ == "__main__":
    main()
