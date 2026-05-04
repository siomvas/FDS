"""
eval_config.py - Shared Configuration & Data Structures
Defines the contract for evaluation metrics emitted by evaluate_partitions.py.
"""
import json
import re
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any
from pathlib import Path

log = logging.getLogger(__name__)


def _infer_dataset_name(path: Path) -> str:
    grandparent = path.parent.parent.name.lower()
    return grandparent if grandparent not in {"embedding_partitions", "partitions", "."} else "unknown"


def _read_client_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "clients" not in data or not isinstance(data["clients"], dict):
        raise KeyError(f"Partition file {path} is missing a valid 'clients' mapping.")
    return len(data["clients"])

# ============================================================================
# 1. Configuration Data Classes
# ============================================================================

@dataclass
class PartitionConfig:
    """Metadata describing a specific partition configuration."""
    # Core Identity
    partition_path: str
    filename: str
    method: str                  # "Random", "Dirichlet", "VLM", "Embedding"
    dataset_name: str            # "tiny_imagenet", "ade20k"
    
    # Embedding Specifics (None for Baselines)
    extractor: Optional[str] = None         # "dinov2-base", "clip-base"
    token_strategy: Optional[str] = None    # "cls", "k_last_cls"
    pooling: Optional[str] = None           # "mean", "gem", "spectral"
    preprocessing: Optional[str] = None     # "l2", "umap", "none"
    clustering: Optional[str] = None        # "hdbscan", "kmeans"
    
    # Hyperparameters
    n_clients: int = 0
    heterogeneity: float = 0.0              # Alpha for Dirichlet/VLM
    seed: int = 0
    
    # Catch-all for granular params
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Generate a clean label for plots."""
        if self.method == "Embedding":
            # Create a descriptive label for embedding methods
            # e.g. "DINOv2 (HDBScan)" or "CLIP (UMAP)"
            details = []
            if self.extractor: details.append(self.extractor.split('-')[0]) # DINOv2
            if self.preprocessing and self.preprocessing != "none": 
                details.append(self.preprocessing.upper()) # UMAP
            if self.clustering: details.append(self.clustering.capitalize())
            return f"{'-'.join(details)} (n={self.n_clients})"
        
        elif self.method == "VLM":
            return f"VLM (n={self.n_clients}, α={self.heterogeneity})"
            
        else:
            return f"{self.method} (α={self.heterogeneity})"

    @classmethod
    def from_path(cls, path_str: str) -> "PartitionConfig":
        """
        Factory: Robustly parses metadata from file path and filename.
        Handles Random, Dirichlet, VLM, and Embedding formats. If n_clients 
        is missing from filename (e.g. HDBSCAN), it reads the JSON to count them.
        """
        path = Path(path_str)
        name = path.name.lower()
        parent_dir = path.parent.name.lower()
        dataset = _infer_dataset_name(path)
        
        config = cls(
            partition_path=str(path),
            filename=path.name,
            method="Unknown",
            dataset_name=dataset
        )

        # --- Case A: Baseline Methods (Random / Dirichlet / VLM) ---
        # These typically follow the format: method_n10_alpha0.5_seed42.json
        if any(x in name for x in ["random", "dirichlet", "vlm", "multivariate"]):
            
            if "random" in name: config.method = "Random"
            elif "vlm" in name or "multivariate" in name: config.method = "VLM"
            else: config.method = "Dirichlet"
            
            # Parse n_clients (e.g., "_n10_")
            n_search = re.search(r'(_n|_clients_?)(\d+)', name)
            if n_search: config.n_clients = int(n_search.group(2))
            
            # Parse alpha (e.g., "_alpha0.5_")
            a_search = re.search(r'(_alpha|_a)([\d\.]+)', name)
            if a_search: config.heterogeneity = float(a_search.group(2))
            
            # Parse seed
            s_search = re.search(r'(_seed|_s)(\d+)', name)
            if s_search: config.seed = int(s_search.group(2))
            
            return config

        # --- Case B: Embedding Methods ---
        # Format: {tokens}_{agg}_k{k}_{pool}_norm-{bool}_dr-{method}_{cluster}_...
        
        config.method = "Embedding"
        config.extractor = parent_dir # Grab from directory structure
        
        # 1. Parsing Flags
        if "norm-true" in name:
            config.extra_params["normalize"] = True
        
        # DR Method
        dr_search = re.search(r'dr-([a-zA-Z0-9]+)', name)
        config.preprocessing = "umap" if (dr_search and "umap" in dr_search.group(1)) else "none"

        # Clustering Method
        if "hdbscan" in name: config.clustering = "hdbscan"
        elif "kmeans" in name: config.clustering = "kmeans"
        elif "spectral" in name: config.clustering = "spectral"
        elif "knn" in name: config.clustering = "knn"

        # Token Strategy
        known_tokens = ["cls", "k_last_cls", "k_last_all", "all_last"]
        for t in known_tokens:
            if name.startswith(t):
                config.token_strategy = t
                break

        # Pooling 
        known_pooling = ["mean", "max", "gem", "spectral", "lastk"]
        for p in known_pooling:
            if f"_{p}_" in name:
                config.pooling = p
                if p != "mean": break 

        # Seed
        s_search = re.search(r'seed(\d+)', name)
        if s_search: config.seed = int(s_search.group(1))
        
        # K-Last
        k_search = re.search(r'_k(\d+)_', name)
        if k_search: config.extra_params["k_last"] = int(k_search.group(1))


        # --- Determine n_clients ---
        # 1. Try filename (KMeans/Spectral usually have it)
        n_search = re.search(r'(_n|_clients_?)(\d+)', name)
        if n_search:
            config.n_clients = int(n_search.group(2))
        else:
            # 2. Fallback: Open file to count keys (HDBSCAN)
            config.n_clients = _read_client_count(path)

        return config

@dataclass
class MetricResults:
    """Standardized container for all metric values."""
    
    # --- Label Metrics ---
    # JS Divergence of label distributions (Higher = More Heterogeneous)
    # JSD_global refers to divergence vs overall distribution
    JSD_mean: Optional[float] = None
    JSD_std: Optional[float] = None
    JSD_global_mean: Optional[float] = None
    JSD_global_std: Optional[float] = None
    
    # TV Distance of label distributions
    TV_mean: Optional[float] = None
    TV_std: Optional[float] = None
    
    # --- Balance Metrics ---
    samples_per_client_mean: float = 0.0
    samples_per_client_std: float = 0.0
    samples_per_client_min: int = 0
    samples_per_client_max: int = 0
    balance_cv: float = 0.0
    min_size: int = 0
    max_size: int = 0
    
    # --- Feature Metrics (Visual) ---
    # FID: Dist shift vs global (Lower = Better/Closer to global)
    fid_mean: Optional[float] = None
    fid_std: Optional[float] = None
    mmd_features_mean: Optional[float] = None
    mmd_features_std: Optional[float] = None
    wasserstein_mean: Optional[float] = None
    wasserstein_std: Optional[float] = None
    energy_dist_mean: Optional[float] = None
    energy_dist_std: Optional[float] = None
    # ks_multivariate_reject_rate: Optional[float] = None
    # ks_multivariate_min_pvalue: Optional[float] = None
    vendi_score_mean: Optional[float] = None
    vendi_score_std: Optional[float] = None
    
    # KNN Separability: How well client clusters separate (Higher = More Heterogeneous)
    knn_sep_mean: Optional[float] = None
    knn_sep_std: Optional[float] = None
    
    # Feature Variance: Spread of features (Higher = More Spread)
    feature_variance_mean: Optional[float] = None
    feature_variance_std: Optional[float] = None
    
    # Intrinsic Dimensionality
    id_mean: Optional[float] = None
    id_std: Optional[float] = None
    
    # Centroid Distance: Distance between client centers
    centroid_dist_mean: Optional[float] = None
    centroid_dist_std: Optional[float] = None

@dataclass
class EvaluationResult:
    """The final object saved to disk for each experiment."""
    config: PartitionConfig
    metrics: MetricResults
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Flattens the object for CSV/JSON export."""
        base = asdict(self.config)
        # Flatten metrics into the same level
        base.update(asdict(self.metrics))
        base["timestamp"] = self.timestamp
        return base

# ============================================================================
# 2. Shared Constants (The "Single Source of Truth")
# ============================================================================
# Used by the visualizer to auto-generate plots.
METRIC_META = {
    # --- Label Heterogeneity ---
    "JSD_mean": {
        "title": "Label Heterogeneity (JS Divergence)",
        "ylabel": "Jensen-Shannon Div",
        "higher_is_better": True,
        "std_field": "JSD_std",
        "category": "label"
    },
    "JSD_global_mean": {
        "title": "Global Deviation (JSD vs Global)",
        "ylabel": "JSD to Global",
        "higher_is_better": True,
        "std_field": "JSD_global_std",
        "category": "label"
    },
    "TV_mean": {
        "title": "Label Heterogeneity (Total Variation)",
        "ylabel": "TV Distance",
        "higher_is_better": True,
        "std_field": "TV_std",
        "category": "label"
    },

    # --- Feature Distance (Distribution Shift) ---
    "fid_mean": {
        "title": "Distribution Shift (FID)",
        "ylabel": "FID Score",
        "higher_is_better": False,
        "std_field": "fid_std",
        "category": "feature_distance"
    },
    "mmd_features_mean": {
        "title": "MMD (Features)",
        "ylabel": "MMD Score",
        "higher_is_better": False,
        "std_field": "mmd_features_std",
        "category": "feature_distance"
    },
    "wasserstein_mean": {
        "title": "Sliced Wasserstein Distance",
        "ylabel": "Wasserstein Dist",
        "higher_is_better": False,
        "std_field": "wasserstein_std",
        "category": "feature_distance"
    },
    "energy_dist_mean": {
        "title": "Energy Distance",
        "ylabel": "Energy Distance",
        "higher_is_better": False,
        "std_field": "energy_dist_std",
        "category": "feature_distance"
    },
    # "ks_multivariate_reject_rate": {
    #     "title": "KS Test Rejection Rate",
    #     "ylabel": "Rejection Rate",
    #     "higher_is_better": False, # Lower = distributions are more similar
    #     "std_field": None, # Rate is a single scalar summary
    #     "category": "feature_distance"
    # },

    # --- Feature Heterogeneity (Separability) ---
    "knn_sep_mean": {
        "title": "Visual Heterogeneity (k-NN Separability)",
        "ylabel": "Separability Score",
        "higher_is_better": True,
        "std_field": "knn_sep_std",
        "category": "feature_heterogeneity"
    },
    "centroid_dist_mean": {
        "title": "Semantic Distance (Centroids)",
        "ylabel": "Euclidean Distance",
        "higher_is_better": True,
        "std_field": "centroid_dist_std",
        "category": "feature_heterogeneity"
    },

    # --- Feature Diversity (Intra-Client) ---
    "vendi_score_mean": {
        "title": "Intra-Client Diversity (Vendi Score)",
        "ylabel": "Effective Rank",
        "higher_is_better": True,
        "std_field": "vendi_score_std",
        "category": "feature_diversity"
    },
    "id_mean": {
        "title": "Intrinsic Dimensionality",
        "ylabel": "Estimated Dimension",
        "higher_is_better": True,
        "std_field": "id_std",
        "category": "feature_diversity"
    },
    "feature_variance_mean": {
        "title": "Feature Variance",
        "ylabel": "Variance",
        "higher_is_better": True,
        "std_field": "feature_variance_std",
        "category": "feature_diversity"
    },

    # --- Balance ---
    "balance_cv": {
        "title": "Data Imbalance (CV)",
        "ylabel": "Coefficient of Variation",
        "higher_is_better": False, # 0 is perfect balance
        "std_field": None, 
        "category": "balance"
    }
}
