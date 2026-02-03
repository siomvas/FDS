import argparse
import sys
from pathlib import Path

from fl_training.nvflare27.learners.classification_learner import ClassificationLearner
from fl_training.nvflare27.learners.segmentation_learner import SegmentationLearner
from fl_training.nvflare27.learners.detection_learner import DetectionLearner


def parse_args():
    parser = argparse.ArgumentParser(
        description="FDS NVFlare 2.7 Client Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,)

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
