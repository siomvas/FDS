from setuptools import setup, find_packages

setup(
    name="fds",
    version="0.1.0",
    packages=find_packages(),  # This will find all packages with __init__.py
    package_dir={'': '.'},  # Packages are in the root directory
    install_requires=[
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.2",
    ],
    extras_require={
        "fl": [
            "nvflare>=2.7.0,<3.0.0",
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "timm>=0.9.0",
            "transformers>=4.30.0",
            "segmentation-models-pytorch>=0.3.3",
            "pillow>=10.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
        ],
    },
    python_requires=">=3.10",
    description="FDS - Federated Dataset Simulation",
)
