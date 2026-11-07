from setuptools import find_packages, setup


def _dedupe(values):
    return list(dict.fromkeys(values))


CORE_REQUIRES = [
    "pyyaml>=6.0",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.2",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
]

FL_REQUIRES = [
    "nvflare>=2.7.2,<3.0.0",
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "tensorboard>=2.14.0",
    "optuna>=3.6.0",
    "timm>=0.9.0",
    "transformers>=4.30.0",
    "segmentation-models-pytorch>=0.3.3",
    "scipy>=1.10.0",
]

PARTITIONING_REQUIRES = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "umap-learn>=0.5.5",
    "tqdm>=4.66.0",
    "optuna>=3.6.0",
]

EMBEDDING_REQUIRES = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "timm>=0.9.0",
    "transformers>=4.30.0",
    "kymatio>=0.3.0",
    "scikit-learn>=1.3.0",
    "tqdm>=4.66.0",
]

EVALUATION_REQUIRES = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.18.0",
    "vendi-score>=0.0.2",
]

DEV_REQUIRES = [
    "pytest>=8.0.0",
    "hypothesis>=6.0.0",
]

EXTRAS = {
    "fl": _dedupe(FL_REQUIRES),
    "partitioning": _dedupe(PARTITIONING_REQUIRES),
    "embedding": _dedupe(EMBEDDING_REQUIRES),
    "evaluation": _dedupe(EVALUATION_REQUIRES),
    "dev": _dedupe(DEV_REQUIRES),
}

EXTRAS["all"] = _dedupe(
    FL_REQUIRES
    + PARTITIONING_REQUIRES
    + EMBEDDING_REQUIRES
    + EVALUATION_REQUIRES
    + DEV_REQUIRES
)

PACKAGE_INCLUDE_PATTERNS = [
    "data_loading",
    "data_loading.*",
    "embedding",
    "embedding.*",
    "evaluation",
    "evaluation.*",
    "fl_training",
    "fl_training.*",
    "partitioning",
    "partitioning.*",
    "research",
    "research.*",
    "utils",
    "utils.*",
]


setup(
    name="fds",
    version="0.1.0",
    packages=find_packages(include=PACKAGE_INCLUDE_PATTERNS),
    package_dir={"": "."},
    py_modules=[],
    install_requires=CORE_REQUIRES,
    extras_require=EXTRAS,
    python_requires=">=3.10",
    description="FDS - Federated Dataset Simulation",
    license="Apache-2.0",
)
