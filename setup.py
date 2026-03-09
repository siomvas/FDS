from setuptools import setup, find_packages


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
    "nvflare>=2.7.0,<3.0.0",
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "tensorboard>=2.14.0",
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

DOCS_REQUIRES = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
    "sphinx-autodoc-typehints>=2.0.0",
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
    "docs": _dedupe(DOCS_REQUIRES),
    "dev": _dedupe(DEV_REQUIRES),
}

EXTRAS["all"] = _dedupe(
    FL_REQUIRES
    + PARTITIONING_REQUIRES
    + EMBEDDING_REQUIRES
    + EVALUATION_REQUIRES
    + DOCS_REQUIRES
    + DEV_REQUIRES
)


setup(
    name="fds",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=CORE_REQUIRES,
    extras_require=EXTRAS,
    python_requires=">=3.10",
    description="FDS - Federated Dataset Simulation",
)
