"""
Setup Verification Script

Run this script to verify that all paths and dependencies are correctly configured
before running the multivariate partitioning evaluation on HPC.
"""

import sys
from pathlib import Path

def check_file_exists(path, description):
    """Check if a file exists and print status."""
    path = Path(path)
    if path.exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: NOT FOUND - {path}")
        return False

def check_directory_exists(path, description):
    """Check if a directory exists and print status."""
    path = Path(path)
    if path.exists() and path.is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: NOT FOUND - {path}")
        return False

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError:
        print(f"✗ {module_name} - NOT INSTALLED")
        return False

def main():
    print("=" * 80)
    print("MULTIVARIATE PARTITIONING SETUP VERIFICATION")
    print("=" * 80)

    all_checks_passed = True

    # Check implementation files
    print("\n[1/5] Checking Implementation Files")
    print("-" * 80)
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
    print("\n[2/5] Checking Evaluation Scripts")
    print("-" * 80)
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
    print("\n[3/5] Checking Cluster Label Data")
    print("-" * 80)
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
    print("\n[4/5] Checking Python Dependencies")
    print("-" * 80)
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
    print("\n[5/5] Checking Documentation")
    print("-" * 80)
    checks = [
        ("data_partitioning/MULTIVARIATE_USAGE.md", "Usage Guide"),
        ("MULTIVARIATE_IMPLEMENTATION_SUMMARY.md", "Implementation Summary"),
    ]
    for path, desc in checks:
        if not check_file_exists(path, desc):
            all_checks_passed = False

    # Important paths to configure
    print("\n" + "=" * 80)
    print("IMPORTANT: Update These Paths for HPC Deployment")
    print("=" * 80)
    print("\nIn evaluation/compare_partitioning_methods.py:")
    print("  - BASE_PATH: Path to TinyImageNet data on HPC")
    print("  - CLUSTER_LABELS_DIR: Path to VLM cluster labels")
    print("\nIn evaluation/test_multivariate_splitter.py:")
    print("  - base_path: Path to TinyImageNet data on HPC")
    print("  - cluster_labels_dir: Path to VLM cluster labels")
    print("\nIn examples/multivariate_partitioning_example.py:")
    print("  - base_path: Path to TinyImageNet data on HPC")
    print("  - cluster_labels_dir: Path to VLM cluster labels")

    # Final summary
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Update paths in evaluation scripts for HPC")
        print("2. Transfer code to HPC")
        print("3. Run: python evaluation/test_multivariate_splitter.py")
        print("4. Run: python evaluation/compare_partitioning_methods.py")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        print("\nPlease resolve the issues above before deploying to HPC.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
