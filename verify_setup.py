"""
Setup Verification Script

Run this script to verify that all paths and dependencies are correctly configured
before running the multivariate partitioning evaluation on HPC.
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def check_file_exists(path, description):
    """Check if a file exists and print status."""
    path = Path(path)
    if path.exists():
        logger.info(f"✓ {description}: {path}")
        return True
    else:
        logger.info(f"✗ {description}: NOT FOUND - {path}")
        return False

def check_directory_exists(path, description):
    """Check if a directory exists and print status."""
    path = Path(path)
    if path.exists() and path.is_dir():
        logger.info(f"✓ {description}: {path}")
        return True
    else:
        logger.info(f"✗ {description}: NOT FOUND - {path}")
        return False

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        logger.info(f"✓ {module_name}")
        return True
    except ImportError:
        logger.info(f"✗ {module_name} - NOT INSTALLED")
        return False

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("=" * 80)
    logger.info("MULTIVARIATE PARTITIONING SETUP VERIFICATION")
    logger.info("=" * 80)

    all_checks_passed = True

    # Check implementation files
    logger.info("\n[1/5] Checking Implementation Files")
    logger.info("-" * 80)
    checks = [
        ("data_partitioning/multivariate_dirichlet_splitter.py", "Multivariate Splitter"),
        ("data_partitioning/base_splitter.py", "Base Splitter"),
        ("data_partitioning/dirichlet_splitter.py", "Dirichlet Splitter"),
        ("data_partitioning/random_splitter.py", "Random Splitter"),
        ("data_partitioning/dataset.py", "Dataset Loader"),
    ]
    for path, desc in checks:
        if not check_file_exists(path, desc):
            all_checks_passed = False

    # Check evaluation scripts
    logger.info("\n[2/5] Checking Evaluation Scripts")
    logger.info("-" * 80)
    checks = [
        ("evaluation/test_multivariate_splitter.py", "Test Script"),
        ("evaluation/compare_partitioning_methods.py", "Evaluation Script"),
        ("evaluation/metrics.py", "Metrics Module"),
        ("examples/multivariate_partitioning_example.py", "Example Script"),
    ]
    for path, desc in checks:
        if not check_file_exists(path, desc):
            all_checks_passed = False

    # Check cluster label data
    logger.info("\n[3/5] Checking Cluster Label Data")
    logger.info("-" * 80)
    base_dir = "data_partitioning/legacy_vlm/data/tiny_imagenet"
    criteria_dirs = [
        "output_color_test",
        "output_main_object_test",
        "output_size_test",
        "output_location_test",
        "output_time_test",
    ]
    for criterion_dir in criteria_dirs:
        jsonl_path = f"{base_dir}/{criterion_dir}/step3_result.jsonl"
        if not check_file_exists(jsonl_path, criterion_dir):
            all_checks_passed = False

    # Check Python dependencies
    logger.info("\n[4/5] Checking Python Dependencies")
    logger.info("-" * 80)
    required_packages = [
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "PIL",
        "pandas",
        "alibi_detect",
    ]
    for package in required_packages:
        if not check_import(package):
            all_checks_passed = False

    # Check documentation
    logger.info("\n[5/5] Checking Documentation")
    logger.info("-" * 80)
    checks = [
        ("data_partitioning/MULTIVARIATE_USAGE.md", "Usage Guide"),
        ("MULTIVARIATE_IMPLEMENTATION_SUMMARY.md", "Implementation Summary"),
    ]
    for path, desc in checks:
        if not check_file_exists(path, desc):
            all_checks_passed = False

    # Important paths to configure
    logger.info("\n" + "=" * 80)
    logger.info("IMPORTANT: Update These Paths for HPC Deployment")
    logger.info("=" * 80)
    logger.info("\nIn evaluation/compare_partitioning_methods.py:")
    logger.info("  - BASE_PATH: Path to TinyImageNet data on HPC")
    logger.info("  - CLUSTER_LABELS_DIR: Path to VLM cluster labels")
    logger.info("\nIn evaluation/test_multivariate_splitter.py:")
    logger.info("  - base_path: Path to TinyImageNet data on HPC")
    logger.info("  - cluster_labels_dir: Path to VLM cluster labels")
    logger.info("\nIn examples/multivariate_partitioning_example.py:")
    logger.info("  - base_path: Path to TinyImageNet data on HPC")
    logger.info("  - cluster_labels_dir: Path to VLM cluster labels")

    # Final summary
    logger.info("\n" + "=" * 80)
    if all_checks_passed:
        logger.info("✓ ALL CHECKS PASSED!")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("1. Update paths in evaluation scripts for HPC")
        logger.info("2. Transfer code to HPC")
        logger.info("3. Run: python evaluation/test_multivariate_splitter.py")
        logger.info("4. Run: python evaluation/compare_partitioning_methods.py")
        return 0
    else:
        logger.info("✗ SOME CHECKS FAILED")
        logger.info("=" * 80)
        logger.info("\nPlease resolve the issues above before deploying to HPC.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
