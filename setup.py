from setuptools import setup, find_packages

setup(
    name="fds",
    version="0.1.0",
    packages=find_packages(),  # This will find all packages with __init__.py
    package_dir={'': '.'},  # Packages are in the root directory
    install_requires=[
        "pyyaml>=6.0",  # Required for YAML experiment configuration
    ],
    python_requires=">=3.10",
    description="FDS - Federated Dataset Simulation",
)
