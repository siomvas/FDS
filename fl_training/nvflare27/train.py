"""Client training script for NVFlare 2.7 federated learning.

This script is the entry point for NVFlare ScriptExecutor during FL training.
It parses command-line arguments, instantiates the appropriate task-specific learner
(classification, segmentation, or detection), and executes FL training using the
NVFlare Client API.

Usage:
    Called automatically by NVFlare ScriptExecutor with arguments from job config:

    python train.py \\
        --data_path outputs/splits.json \\
        --model_name resnet18 \\
        --task_type classification \\
        --num_classes 200 \\
        --lr 0.001 \\
        --local_epochs 1 \\
        --batch_size 32 \\
        --amp_enabled \\
        --grad_tracking \\
        --seed 42

Architecture:
    - Task-agnostic entry point supporting all 3 task types
    - Conditional learner instantiation based on task_type argument
    - NVFlare Client API integration handled by learner.execute()
    - HPC proxy configuration for pretrained model downloads
    - Reproducible seeding for research experiments

Dependencies:
    - fl_training.nvflare27.learners.classification_learner
    - fl_training.nvflare27.learners.segmentation_learner
    - fl_training.nvflare27.learners.detection_learner
    - nvflare.client (NVFlare Client API)
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path for FDS imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fl_training.nvflare27.learners.classification_learner import ClassificationLearner
from fl_training.nvflare27.learners.segmentation_learner import SegmentationLearner
from fl_training.nvflare27.learners.detection_learner import DetectionLearner


def parse_args():
    """Parse command-line arguments from NVFlare ScriptExecutor.

    Returns:
        argparse.Namespace: Parsed arguments with all training configuration

    Raises:
        SystemExit: If required arguments missing or invalid values provided
    """
    parser = argparse.ArgumentParser(
        description="FDS NVFlare 2.7 Client Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classification training (Tiny ImageNet)
  python train.py --data_path outputs/splits.json --model_name resnet18 \\
      --task_type classification --num_classes 200 --lr 0.001 --local_epochs 1 \\
      --batch_size 32 --amp_enabled --grad_tracking --seed 42

  # Segmentation training (ADE20K)
  python train.py --data_path outputs/splits.json --model_name deeplabv3_resnet50 \\
      --task_type segmentation --num_classes 151 --lr 0.001 --local_epochs 1 \\
      --batch_size 8 --amp_enabled --seed 42

  # Detection training (COCO)
  python train.py --data_path outputs/splits.json --model_name fasterrcnn_resnet50 \\
      --task_type detection --num_classes 80 --lr 0.001 --local_epochs 1 \\
      --batch_size 4 --seed 42
        """
    )

    # Data and model configuration
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to DatasetSplits JSON file (from partitioning splitters)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model architecture identifier (resnet18, vgg11, deeplabv3_resnet50, fasterrcnn_resnet50, etc.)"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["classification", "segmentation", "detection"],
        help="Task type determines which learner to instantiate"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of output classes (classification: 200 for Tiny ImageNet, segmentation: 151 for ADE20K, detection: 80 for COCO)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local training epochs per FL round (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32, detection typically uses 2-8)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4)"
    )

    # Training optimizations
    parser.add_argument(
        "--amp_enabled",
        action="store_true",
        help="Enable automatic mixed precision (AMP) for GPU memory optimization"
    )
    parser.add_argument(
        "--grad_tracking",
        action="store_true",
        help="Enable gradient diversity tracking for FL analysis"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training device (default: cuda, falls back to cpu if CUDA unavailable)"
    )

    return parser.parse_args()


def create_learner_config(args):
    """Create learner configuration dict from parsed arguments.

    Args:
        args: argparse.Namespace from parse_args()

    Returns:
        dict: Configuration dictionary for learner initialization with keys:
            - splits_path: Path to DatasetSplits JSON
            - task_type: Task type identifier
            - num_classes: Number of output classes
            - lr: Learning rate
            - local_epochs: Epochs per round
            - batch_size: Training batch size
            - num_workers: DataLoader workers
            - device: torch device
            - amp_enabled: Enable AMP flag
            - grad_tracking_enabled: Enable gradient tracking flag
            - seed: Random seed
    """
    return {
        "splits_path": args.data_path,
        "task_type": args.task_type,
        "num_classes": args.num_classes,
        "lr": args.lr,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "amp_enabled": args.amp_enabled,
        "grad_tracking_enabled": args.grad_tracking,
        "seed": args.seed
    }


def main():
    """Main entry point for client training script.

    Workflow:
        1. Parse command-line arguments
        2. Create learner configuration
        3. Instantiate task-specific learner based on task_type
        4. Execute FL training via learner.execute()

    The learner.execute() method handles:
        - NVFlare Client API lifecycle (receive global model, train, send updates)
        - Data loading via FDSDataFactory
        - Model creation with HPC proxy configuration
        - TensorBoard metric logging
        - Gradient tracking (if enabled)

    Raises:
        ValueError: If task_type is invalid (should not happen due to argparse choices)
        SystemExit: If required arguments missing or learner initialization fails
    """
    # Parse arguments
    args = parse_args()

    # Create learner configuration
    config = create_learner_config(args)

    # Log startup information
    print("=" * 80)
    print("FDS NVFlare 2.7 Client Training Script")
    print("=" * 80)
    print(f"Task Type: {args.task_type}")
    print(f"Model: {args.model_name}")
    print(f"Data Path: {args.data_path}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Learning Rate: {args.lr}")
    print(f"Local Epochs: {args.local_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"AMP Enabled: {args.amp_enabled}")
    print(f"Gradient Tracking: {args.grad_tracking}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Instantiate task-specific learner
    if args.task_type == "classification":
        learner = ClassificationLearner(
            data_path=args.data_path,
            model_name=args.model_name,
            config=config
        )
        print("✓ ClassificationLearner instantiated")

    elif args.task_type == "segmentation":
        learner = SegmentationLearner(
            data_path=args.data_path,
            model_name=args.model_name,
            config=config
        )
        print("✓ SegmentationLearner instantiated")

    elif args.task_type == "detection":
        learner = DetectionLearner(
            data_path=args.data_path,
            model_name=args.model_name,
            config=config
        )
        print("✓ DetectionLearner instantiated")

    else:
        # Should never reach here due to argparse choices validation
        raise ValueError(
            f"Invalid task_type: {args.task_type}. "
            f"Expected one of: ['classification', 'segmentation', 'detection']"
        )

    # Execute FL training
    print("\nStarting NVFlare Client API training loop...")
    print("-" * 80)
    learner.execute()
    print("-" * 80)
    print("✓ Training complete")


if __name__ == "__main__":
    main()
