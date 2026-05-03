"""Multivariate Dirichlet data splitting using Vision-Language Model annotations.

This module implements advanced data partitioning for federated learning by creating
heterogeneity across multiple semantic criteria simultaneously. Unlike traditional
label-based partitioning that only considers class labels, this approach uses
Vision-Language Model (VLM) annotations to create realistic multi-dimensional
heterogeneity.

Key Concepts
------------

**Multi-Criterion Heterogeneity**:
    Instead of only distributing samples by class labels, this splitter considers
    multiple visual criteria:
    - **Color**: Color palettes and schemes (e.g., warm, cool, monochrome)
    - **Main Object**: Primary subject types (e.g., animals, people, buildings)
    - **Size**: Object scale and composition (e.g., close-up, wide-angle)
    - **Location**: Environmental context (e.g., indoor, outdoor, urban, natural)
    - **Time**: Temporal context (e.g., day, night, seasons)

**Client Preferences**:
    Each client has preference vectors for each criterion, sampled from Dirichlet
    distributions. These preferences determine which types of samples the client
    is more likely to receive, creating natural heterogeneity patterns.

**Affinity-Based Assignment**:
    Samples are assigned to clients using softmax probabilities based on affinity
    scores. The affinity between a sample and client is the sum of the client's
    preferences for the sample's attributes across all criteria.

VLM Annotation Format
---------------------

This splitter expects pre-computed VLM annotations in JSONL format:

    {"image": "/path/to/image.jpg", "class": "warm_colors"}
    {"image": "/path/to/image2.jpg", "class": "cool_colors"}

The annotations should be organized by criterion in separate directories:
    - output_color_train/step3_result.jsonl
    - output_main_object_train/step3_result.jsonl
    - output_size_train/step3_result.jsonl
    - output_location_train/step3_result.jsonl
    - output_time_train/step3_result.jsonl

Alpha Parameter Effects
-----------------------

The heterogeneity (alpha) parameter controls preference diversity:

- **Low alpha (0.1-0.5)**: Highly heterogeneous preferences
    - Clients have strong preferences for specific attributes
    - Example: Client A prefers warm colors AND outdoor scenes AND daytime
    - Creates extreme specialization across multiple dimensions

- **Medium alpha (1.0-5.0)**: Moderately heterogeneous preferences
    - Clients have diverse but not extreme preferences
    - Balanced across multiple criteria
    - Realistic for many federated scenarios

- **High alpha (10.0+)**: Nearly uniform preferences
    - All clients have similar preferences across criteria
    - Approaches random (IID) assignment

Examples
--------

Basic usage with VLM annotations:

    >>> from partitioning import MultivariateDirichletSplitter
    >>> from data_loading.dataset import BaseDataset
    >>>
    >>> train_data = BaseDataset("tiny_imagenet", "/data/", "train")
    >>> val_data = BaseDataset("tiny_imagenet", "/data/", "val")
    >>>
    >>> splitter = MultivariateDirichletSplitter(
    ...     train_dataset=train_data,
    ...     val_dataset=val_data,
    ...     n_clients=10,
    ...     heterogeneity=0.5,  # High heterogeneity
    ...     cluster_labels_dir="/data/vlm_annotations/",
    ...     valset_type="global",
    ...     seed=42,
    ...     min_samples_per_client=100
    ... )
    >>> splits = splitter.get_client_splits()
    >>> preferences = splitter.get_preference_summary()

Analyzing client preferences:

    >>> summary = splitter.get_preference_summary()
    >>> # See each client's top preferences for each criterion
    >>> for client_id in range(3):
    ...     for criterion in splitter.CRITERIA:
    ...         top_pref = summary[f"{criterion}_top_preferences"][client_id]
    ...         f"Client {client_id} prefers {criterion}={top_pref['label']} ({top_pref['preference']:.2f})"

See Also
--------

- DirichletSplitter: Single-criterion (label-based) heterogeneity
- EmbeddingSplitter: Semantic clustering without explicit criteria
- For VLM annotation generation, see: docs/MULTIVARIATE_USAGE.md
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base_splitter import BaseDatasetSplitter, DatasetSplits, ClientSplit
from data_loading.dataset import BaseDataset

logger = logging.getLogger(__name__)


class MultivariateDirichletSplitter(BaseDatasetSplitter):
    """Split dataset using multi-criterion VLM annotations with Dirichlet preferences.

    This splitter creates federated data heterogeneity across multiple semantic
    dimensions simultaneously using Vision-Language Model (VLM) annotations. Unlike
    traditional single-criterion splitters, this approach models realistic scenarios
    where clients have preferences across multiple independent criteria.

    The splitting process:
    1. Load VLM annotations for each criterion (color, object, size, location, time)
    2. Sample client preference vectors from Dirichlet distributions for each criterion
    3. Compute affinity scores between samples and clients based on preferences
    4. Assign samples using softmax probabilities over affinity scores
    5. Retry with resampled preferences if minimum sample constraints aren't met

    Attributes:
        CRITERIA: List of supported criteria (color, main_object, size, location, time)
        CRITERION_TO_DIR: Mapping from criteria to VLM annotation directory names
        cluster_labels_dir: Path to directory containing VLM annotation files
        min_samples_per_client: Minimum samples each client must receive
        max_retries: Maximum retry attempts for constraint satisfaction
        temperature: Softmax temperature for assignment probabilities
        cluster_labels: Loaded VLM annotations (criterion -> image -> label)
        unique_labels: Unique label values for each criterion
        client_preferences: Sampled Dirichlet preferences (client -> criterion -> vector)

    Examples:
        Create heterogeneous splits with strong preferences:

        >>> splitter = MultivariateDirichletSplitter(
        ...     train_dataset=train_data,
        ...     val_dataset=val_data,
        ...     n_clients=10,
        ...     heterogeneity=0.3,  # Low alpha = strong preferences
        ...     cluster_labels_dir="/data/vlm_annotations/",
        ...     valset_type="global",
        ...     seed=42
        ... )
        >>> splits = splitter.get_client_splits()

        Analyze client preferences after splitting:

        >>> summary = splitter.get_preference_summary()
        >>> summary['color_top_preferences'][0]
        {'client': 0, 'label': 'warm_colors', 'preference': 0.85}

    See Also:
        DirichletSplitter: Single-criterion (label-based) heterogeneity
        EmbeddingSplitter: Semantic clustering-based partitioning
    """

    CRITERIA_tiny = ["color", "main_object", "size", "location", "time"]
    CRITERION_TO_DIR_tiny = {
        "color": "output_color_train",
        "main_object": "output_main_object_train",
        "size": "output_size_train",
        "location": "output_location_train",
        "time": "output_time_train",
    }
    CRITERIA_food = ["color", "main_object", "size", "angle", "lighting"]
    CRITERION_TO_DIR_food = {
        "color": "output_color_train",
        "main_object": "output_main_object_train",
        "size": "output_size_train",
        "angle": "output_angle_train",
        "lighting": "output_lighting_train",
    }
    CRITERIA_ade20k = ["color", "main_object", "size", "scene", "light"]
    CRITERION_TO_DIR_ade20k = {
        "color": "output_color_train",
        "main_object": "output_main_object_train",
        "size": "output_size_train",
        "scene": "output_scene_train",
        "light": "output_light_train",
    }
    CRITERIA_VOC = ["color", "light", "size", "location", "main_object"]
    CRITERION_TO_DIR_VOC = {
        "color": "output_color_train",
        "light": "output_light_train",
        "size": "output_size_train",
        "location": "output_location_train",
        "main_object": "output_main_object_train",
    }
    _PATH_ANCHORS = ("VOC2012", "tiny-imagenet-200", "food101", "food-101", "ADE20K", "bdd100k")
    _PATH_ANCHOR_ALIASES = {"food-101": "food101"}
    _WILDCARD_TOKEN = "*"

    def __init__(
        self,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset],
        n_clients: int,
        heterogeneity: float,
        cluster_labels_dir: str,
        valset_type: str = "global",
        seed: Optional[int] = None,
        min_samples_per_client: int = 10,
        max_retries: int = 1000,
        temperature: float = 1.0,
        joint_min_cell_count: int = 10,
        joint_drop_order: Optional[List[str]] = None,
        joint_overflow_bucket: str = "__RARE__",
    ):
        """Initialize the multivariate Dirichlet splitter.

        Args:
            train_dataset: Training dataset to split across clients.
            val_dataset: Optional validation dataset. Required if valset_type
                is "local" or "global".
            n_clients: Number of clients in the federation. Must be >= 1.
            heterogeneity: Dirichlet concentration parameter (alpha) controlling
                preference diversity. Applies to all criteria uniformly.
                - Low (0.1-0.5): Strong, skewed preferences (high heterogeneity)
                - Medium (1.0-5.0): Moderate preferences (balanced heterogeneity)
                - High (10.0+): Weak, uniform preferences (low heterogeneity, ~IID)
            cluster_labels_dir: Path to directory containing VLM annotation JSONL
                files. Expected structure:
                    cluster_labels_dir/
                    ├── output_color_train/step3_result.jsonl
                    ├── output_main_object_train/step3_result.jsonl
                    ├── output_size_train/step3_result.jsonl
                    ├── output_location_train/step3_result.jsonl
                    └── output_time_train/step3_result.jsonl
            valset_type: Validation set configuration - "local", "global", or "none".
                See BaseDatasetSplitter for details.
            seed: Random seed for reproducibility. Controls preference sampling
                and sample assignment randomness.
            min_samples_per_client: Minimum number of samples each client must
                receive. If not satisfied, preferences are resampled (up to
                max_retries attempts).
            max_retries: Maximum number of retry attempts if min_samples_per_client
                constraint is not satisfied. After max_retries, raises ValueError.
            temperature: Softmax temperature for assignment probabilities.
                - Low (<1.0): Sharpens probabilities (more deterministic)
                - 1.0: Standard softmax (default)
                - High (>1.0): Smooths probabilities (more random)
            joint_min_cell_count: Minimum support required for a full joint
                criterion cell before it is merged via wildcard backoff.
                Set to 1 to disable rare-cell merging.
            joint_drop_order: Criteria order used only for rare-cell merging.
                If omitted, defaults to CRITERIA order for the dataset.
            joint_overflow_bucket: Bucket name used when a rare raw cell cannot
                reach joint_min_cell_count even after full wildcard backoff.

        Raises:
            ValueError: If valset_type is invalid or incompatible with val_dataset.

        Examples:
            Basic initialization with moderate heterogeneity:

            >>> splitter = MultivariateDirichletSplitter(
            ...     train_dataset=train_data,
            ...     val_dataset=val_data,
            ...     n_clients=10,
            ...     heterogeneity=1.0,
            ...     cluster_labels_dir="/data/vlm_annotations/",
            ...     seed=42
            ... )

            High heterogeneity with strict sample constraints:

            >>> splitter = MultivariateDirichletSplitter(
            ...     train_dataset=train_data,
            ...     val_dataset=None,
            ...     n_clients=20,
            ...     heterogeneity=0.1,  # Strong preferences
            ...     cluster_labels_dir="/data/vlm_annotations/",
            ...     valset_type="none",
            ...     min_samples_per_client=50,
            ...     max_retries=5000,
            ...     seed=42
            ... )
        """
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=n_clients,
            heterogeneity=heterogeneity,
            valset_type=valset_type,
            seed=seed,
        )

        # Choose criteria based on dataset type. Support both BaseDataset and
        # lightweight test/mocked datasets that don't expose dataset_name.
        dataset_name = getattr(train_dataset, "dataset_name", None)
        if dataset_name is None:
            dataset_name = getattr(train_dataset, "requested_dataset_name", None)
        if dataset_name is None:
            logger.warning(
                "train_dataset has no dataset_name attribute; defaulting multivariate criteria to tiny_imagenet."
            )
            dataset_name = "tiny_imagenet"
        dataset_name = str(dataset_name).lower()

        if dataset_name in {"tiny_imagenet", "tinyimagenet", "tiny-imagenet"}:
            self.CRITERIA = self.CRITERIA_tiny
            self.CRITERION_TO_DIR = self.CRITERION_TO_DIR_tiny
        elif dataset_name in {"food101"}:
            self.CRITERIA = self.CRITERIA_food
            self.CRITERION_TO_DIR = self.CRITERION_TO_DIR_food
        elif dataset_name in {"ade20k"}:
            self.CRITERIA = self.CRITERIA_ade20k
            self.CRITERION_TO_DIR = self.CRITERION_TO_DIR_ade20k
        elif dataset_name in {"voc", "pascal_voc", "pascal-voc", "pascalvoc"}:
            self.CRITERIA = self.CRITERIA_VOC
            self.CRITERION_TO_DIR = self.CRITERION_TO_DIR_VOC
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported for "
                f"MultivariateDirichletSplitter. Supported datasets: "
                f"tiny_imagenet, food101, ade20k, pascal_voc."
            )

        self.cluster_labels_dir = Path(cluster_labels_dir)
        self.min_samples_per_client = min_samples_per_client
        self.max_retries = max_retries
        self.temperature = temperature
        self.joint_min_cell_count = int(joint_min_cell_count)
        if self.joint_min_cell_count < 1:
            raise ValueError("joint_min_cell_count must be >= 1.")

        drop_order = list(joint_drop_order) if joint_drop_order is not None else list(self.CRITERIA)
        invalid = [criterion for criterion in drop_order if criterion not in self.CRITERIA]
        if invalid:
            raise ValueError(
                f"joint_drop_order contains invalid criteria {invalid}. "
                f"Valid criteria for this dataset: {self.CRITERIA}"
            )
        drop_order = list(dict.fromkeys(drop_order))
        for criterion in self.CRITERIA:
            if criterion not in drop_order:
                drop_order.append(criterion)
        self.joint_drop_order = drop_order

        if not str(joint_overflow_bucket).strip():
            raise ValueError("joint_overflow_bucket must be a non-empty string.")
        self.joint_overflow_bucket = str(joint_overflow_bucket).strip()

        # Will be populated during splitting
        self.cluster_labels: Dict[str, Dict[str, str]] = {}
        self.unique_labels: Dict[str, List[str]] = {}
        self.client_preferences: Dict[int, Dict[str, np.ndarray]] = {}
        self.joint_cell_preferences: Dict[Tuple[str, ...], np.ndarray] = {}
        self._last_raw_joint_cell_counts: Dict[Tuple[str, ...], int] = {}
        self._last_merged_joint_cell_counts: Dict[Tuple[str, ...], int] = {}

    def _load_cluster_labels(self) -> None:
        """Load VLM cluster labels from JSONL files for all criteria.

        Reads annotation files from the cluster_labels_dir for each criterion
        (color, main_object, size, location, time) and populates self.cluster_labels
        and self.unique_labels. Supports both JSON array and JSONL formats.

        The method normalizes image paths to be user-independent using dataset-anchored
        suffixes when possible (e.g., "VOC2012/images/train/xxx.jpg"), with a fallback
        to the last 5 components for legacy compatibility.

        Raises:
            FileNotFoundError: If any expected annotation file (step3_result.jsonl)
                is missing from the criterion directories.

        Side Effects:
            - Populates self.cluster_labels: {criterion: {image_path: label}}
            - Populates self.unique_labels: {criterion: [sorted unique labels]}
            - Prints loading progress for each criterion
        """
        logger.info("Loading cluster labels from JSONL files...")

        for criterion in self.CRITERIA:
            criterion_dir = self.cluster_labels_dir / self.CRITERION_TO_DIR[criterion]
            jsonl_path = criterion_dir / "step3_result.jsonl"

            if not jsonl_path.exists():
                raise FileNotFoundError(
                    f"Cluster labels not found at {jsonl_path}. "
                    f"Please ensure VLM clustering has been run."
                )

            # Load labels for this criterion
            criterion_labels = {}
            with open(jsonl_path, 'r') as f:
                content = f.read().strip()

                # Check if it's a JSON array (starts with '[') or JSONL (starts with '{')
                if content.startswith('['):
                    # JSON array format
                    entries = json.loads(content)
                    for entry in entries:
                        full_image_path = entry["image"]
                        image_path = self._normalize_image_key(full_image_path)

                        cluster_label = entry["class"]
                        criterion_labels[image_path] = cluster_label
                else:
                    # JSONL format (one JSON object per line)
                    f.seek(0)  # Reset file pointer
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        full_image_path = entry["image"]
                        image_path = self._normalize_image_key(full_image_path)
                        cluster_label = entry["class"]
                        criterion_labels[image_path] = cluster_label

            self.cluster_labels[criterion] = criterion_labels

            # Get unique labels for this criterion
            unique_labels = sorted(set(criterion_labels.values()))
            if not unique_labels:
                raise ValueError(
                    f"No cluster labels loaded for criterion '{criterion}' from {jsonl_path}."
                )
            self.unique_labels[criterion] = unique_labels

            logger.info(
                "%s: %d unique labels, %d images",
                criterion,
                len(unique_labels),
                len(criterion_labels),
            )

    def _sample_client_preferences(self) -> None:
        """Sample client preference vectors from Dirichlet distributions.

        For each client and each criterion, samples a preference vector from
        Dirichlet(alpha) where alpha = [heterogeneity, ..., heterogeneity].
        The dimensionality equals the number of unique labels for that criterion.

        Lower heterogeneity (alpha) values produce more skewed preferences,
        while higher values produce more uniform preferences.

        Side Effects:
            - Populates self.client_preferences: {client_id: {criterion: vector}}
            - Prints preference statistics for client 0 and aggregate stats
            - Sets random seed if self.seed is not None
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        logger.info("Sampling client preferences with alpha=%.4f...", self.heterogeneity)

        # Track statistics across all clients and criteria
        all_max_prefs = []
        all_min_prefs = []
        all_stds = []

        for client_id in range(self.n_clients):
            self.client_preferences[client_id] = {}

            for criterion in self.CRITERIA:
                n_labels = len(self.unique_labels[criterion])
                alpha = np.full(n_labels, self.heterogeneity)
                preferences = np.random.dirichlet(alpha)

                self.client_preferences[client_id][criterion] = preferences

                # Collect statistics
                all_max_prefs.append(np.max(preferences))
                all_min_prefs.append(np.min(preferences))
                all_stds.append(np.std(preferences))

                # Debug: show top preferences for first client
                if client_id == 0:
                    label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels[criterion])}
                    top_idx = np.argmax(preferences)
                    top_label = self.unique_labels[criterion][top_idx]
                    logger.debug(
                        "Client 0, %s: top=%s (%.3f), min=%.3f, std=%.3f",
                        criterion,
                        top_label,
                        preferences[top_idx],
                        preferences.min(),
                        preferences.std(),
                    )

        # Print aggregate statistics
        logger.info("Preference statistics across all clients:")
        logger.info(
            "Max preference - Mean: %.4f, Std: %.4f",
            float(np.mean(all_max_prefs)),
            float(np.std(all_max_prefs)),
        )
        logger.info(
            "Min preference - Mean: %.4f, Std: %.4f",
            float(np.mean(all_min_prefs)),
            float(np.std(all_min_prefs)),
        )
        logger.info(
            "Std deviation - Mean: %.4f, Std: %.4f",
            float(np.mean(all_stds)),
            float(np.std(all_stds)),
        )

    def _get_cluster_label_for_sample(self, image_path: str, criterion: str) -> str:
        """Get the cluster label for a sample under a specific criterion.

        Args:
            image_path: Full path to the image.
            criterion: The criterion to query (e.g., "color", "main_object").

        Returns:
            The cluster label string for the sample under the criterion.

        Raises:
            ValueError: If cluster label not found (data mismatch between dataset and VLM annotations).
        """
        normalized_path = self._normalize_image_key(image_path)
        try:
            return self.cluster_labels[criterion][normalized_path]
        except KeyError:
            raise ValueError(
                f"Cluster label for image {normalized_path} under criterion '{criterion}' not found. "
                f"This indicates a mismatch between your dataset and VLM annotations."
            )

    @classmethod
    def _normalize_image_key(cls, image_path: str) -> str:
        """Normalize image path across environments using anchored dataset suffixes."""
        parts = [part for part in str(image_path).replace("\\", "/").split("/") if part]
        if not parts:
            return ""

        for anchor in cls._PATH_ANCHORS:
            if anchor in parts:
                anchor_idx = parts.index(anchor)
                parts[anchor_idx] = cls._PATH_ANCHOR_ALIASES.get(anchor, anchor)
                return "/".join(parts[anchor_idx:])

        if len(parts) >= 5:
            return "/".join(parts[-5:])
        return "/".join(parts)

    def _compute_affinity_score(self, image_path: str, client_id: int) -> float:
        """Compute affinity score between a sample and a client.

        The affinity score measures how well a sample matches a client's preferences
        across all criteria. It is computed as the sum of the client's preference
        weights for the sample's labels across all criteria:

            affinity = Σ[criterion] preference[criterion][sample_label[criterion]]

        Higher affinity scores indicate stronger matches between sample attributes
        and client preferences.

        Args:
            image_path: Path to the sample image.
            client_id: ID of the client (0 to n_clients-1).

        Returns:
            Affinity score as a float. Typically in range [0, n_criteria], where
            n_criteria=5 (color, main_object, size, location, time).

        Raises:
            ValueError: If the sample's cluster label is missing for any criterion.
        """
        affinity = 0.0

        for criterion in self.CRITERIA:
            cluster_label = self._get_cluster_label_for_sample(image_path, criterion)

            # Get the preference weight for this label
            label_idx = self.unique_labels[criterion].index(cluster_label)
            preference = self.client_preferences[client_id][criterion][label_idx]
            affinity += preference

        return affinity

    def _assign_samples_with_softmax(
        self,
        sample_indices: List[int],
        sample_paths: List[str]
    ) -> Dict[int, List[int]]:
        """Assign samples to clients using softmax probabilities over affinity scores.

        For each sample, computes affinity scores with all clients and converts
        them to assignment probabilities using softmax:

            P(client_i | sample) = exp(affinity_i / T) / Σ exp(affinity_j / T)

        where T is the temperature parameter. Samples are then randomly assigned
        to clients according to these probabilities.

        Args:
            sample_indices: List of dataset sample indices to assign.
            sample_paths: List of corresponding image paths (same length as indices).

        Returns:
            Dictionary mapping client_id (int) to list of assigned sample indices.
            All samples are assigned exactly once.

        Side Effects:
            - Sets random seed to self.seed + 1000 if seed is not None
            - Prints assignment statistics for first 10 samples (affinities, probs, entropy)
        """
        if self.seed is not None:
            np.random.seed(self.seed + 1000)  # Different seed for assignment

        client_assignments = {client_id: [] for client_id in range(self.n_clients)}

        # DEBUG: Track statistics for first 10 samples
        debug_affinities = []
        debug_probabilities = []
        debug_max_probs = []

        for idx, (sample_idx, image_path) in enumerate(zip(sample_indices, sample_paths)):
            # Compute affinity scores for all clients
            affinities = np.array([
                self._compute_affinity_score(image_path, client_id)
                for client_id in range(self.n_clients)
            ])

            # Apply softmax with temperature
            affinities_scaled = affinities / self.temperature
            exp_affinities = np.exp(affinities_scaled - np.max(affinities_scaled))  # Numerical stability
            probabilities = exp_affinities / exp_affinities.sum()

            # DEBUG: Collect statistics for first 10 samples
            if idx < 10:
                debug_affinities.append(affinities)
                debug_probabilities.append(probabilities)
                debug_max_probs.append(np.max(probabilities))

            # Sample client based on probabilities
            assigned_client = np.random.choice(self.n_clients, p=probabilities)
            client_assignments[assigned_client].append(sample_idx)

        # DEBUG: Print statistics
        logger.info("Assignment statistics (alpha=%.4f):", self.heterogeneity)
        if debug_affinities:
            all_affinities = np.array(debug_affinities)
            logger.info("Affinity scores (first 10 samples):")
            logger.info(
                "Mean: %.4f, Std: %.4f",
                float(all_affinities.mean()),
                float(all_affinities.std()),
            )
            logger.info(
                "Range: [%.4f, %.4f]",
                float(all_affinities.min()),
                float(all_affinities.max()),
            )

            all_probs = np.array(debug_probabilities)
            logger.info("Softmax probabilities (first 10 samples):")
            logger.info(
                "Max prob mean: %.4f, Std: %.4f",
                float(np.mean(debug_max_probs)),
                float(np.std(debug_max_probs)),
            )
            logger.info(
                "Entropy (avg): %.4f",
                float(-np.mean([np.sum(p * np.log(p + 1e-10)) for p in all_probs])),
            )

        return client_assignments

    def _build_joint_cells(
        self,
        sample_indices: List[int],
        sample_paths: List[str],
    ) -> Dict[Tuple[str, ...], List[int]]:
        """Build mapping from joint criterion cell to dataset indices."""
        cell_to_indices: Dict[Tuple[str, ...], List[int]] = {}
        for sample_idx, image_path in zip(sample_indices, sample_paths):
            joint_cell = tuple(
                self._get_cluster_label_for_sample(image_path, criterion)
                for criterion in self.CRITERIA
            )
            cell_to_indices.setdefault(joint_cell, []).append(sample_idx)
        return cell_to_indices

    @classmethod
    def _pattern_matches_cell(cls, pattern: Tuple[str, ...], cell: Tuple[str, ...]) -> bool:
        """Return True when a wildcard pattern matches a raw joint cell."""
        return all(p == cls._WILDCARD_TOKEN or p == c for p, c in zip(pattern, cell))

    def _resolve_merged_joint_cell_key(
        self,
        raw_cell: Tuple[str, ...],
        raw_cell_counts: Dict[Tuple[str, ...], int],
    ) -> Tuple[str, ...]:
        """Resolve rare raw joint cells to merged wildcard keys."""
        if raw_cell_counts[raw_cell] >= self.joint_min_cell_count:
            return raw_cell

        drop_indices = [self.CRITERIA.index(criterion) for criterion in self.joint_drop_order]
        pattern = list(raw_cell)
        for drop_idx in drop_indices:
            pattern[drop_idx] = self._WILDCARD_TOKEN
            candidate = tuple(pattern)
            pooled_count = sum(
                count
                for cell, count in raw_cell_counts.items()
                if self._pattern_matches_cell(candidate, cell)
            )
            if pooled_count >= self.joint_min_cell_count:
                return candidate

        return tuple([self.joint_overflow_bucket] * len(self.CRITERIA))

    def _merge_rare_joint_cells(
        self,
        raw_cell_to_indices: Dict[Tuple[str, ...], List[int]],
    ) -> Dict[Tuple[str, ...], List[int]]:
        """Merge rare raw joint cells using wildcard backoff patterns."""
        raw_cell_counts = {cell: len(indices) for cell, indices in raw_cell_to_indices.items()}
        merged: Dict[Tuple[str, ...], List[int]] = {}
        for raw_cell, indices in raw_cell_to_indices.items():
            merged_cell = self._resolve_merged_joint_cell_key(raw_cell, raw_cell_counts)
            merged.setdefault(merged_cell, []).extend(indices)
        return merged

    def _assign_joint_cells_with_dirichlet(
        self,
        merged_cell_to_indices: Dict[Tuple[str, ...], List[int]],
        rng: np.random.Generator,
    ) -> Dict[int, List[int]]:
        """Assign merged joint-cell samples to clients via Dirichlet over clients."""
        client_assignments = {client_id: [] for client_id in range(self.n_clients)}
        self.joint_cell_preferences = {}

        for joint_cell in sorted(merged_cell_to_indices.keys()):
            indices = merged_cell_to_indices[joint_cell]
            probabilities = rng.dirichlet(
                np.full(self.n_clients, self.heterogeneity, dtype=float)
            )
            self.joint_cell_preferences[joint_cell] = probabilities.copy()
            clients = rng.choice(self.n_clients, size=len(indices), p=probabilities)
            for sample_idx, client_id in zip(indices, clients):
                client_assignments[int(client_id)].append(sample_idx)

        for client_id in range(self.n_clients):
            rng.shuffle(client_assignments[client_id])
        return client_assignments

    def _validate_assignments(self, client_assignments: Dict[int, List[int]]) -> bool:
        """Validate that all clients satisfy the minimum sample constraint.

        Args:
            client_assignments: Dictionary mapping client_id to list of assigned
                sample indices.

        Returns:
            True if all clients have at least min_samples_per_client samples,
            False otherwise.
        """
        for client_id in range(self.n_clients):
            if len(client_assignments[client_id]) < self.min_samples_per_client:
                return False
        return True

    def _partition_with_retry(
        self,
        sample_indices: List[int],
        sample_paths: List[str]
    ) -> Dict[int, List[int]]:
        """Partition samples with retry logic to satisfy minimum sample constraints.

        Attempts to assign samples to clients while ensuring all clients receive
        at least min_samples_per_client samples. If the constraint is not satisfied,
        it resamples merged-cell Dirichlet draws with a different seed and retries.

        The retry mechanism helps overcome unlucky preference samples that result
        in highly imbalanced assignments. Each retry uses a different random seed
        (seed + attempt).

        Args:
            sample_indices: List of dataset sample indices to partition.
            sample_paths: List of corresponding image paths.

        Returns:
            Dictionary mapping client_id to assigned sample indices. Guaranteed
            to satisfy the minimum sample constraint.

        Raises:
            ValueError: If unable to satisfy min_samples_per_client constraint
                after max_retries attempts. Includes the actual sample distribution
                from the final attempt in the error message.

        Side Effects:
            - May resample merged-cell Dirichlet probabilities multiple times
            - Prints success message if constraints satisfied after retries
        """
        if len(sample_indices) < (self.min_samples_per_client * self.n_clients):
            raise ValueError(
                f"Unable to satisfy min_samples_per_client={self.min_samples_per_client}: "
                f"need at least {self.min_samples_per_client * self.n_clients} samples, "
                f"but received {len(sample_indices)}."
            )

        raw_cell_to_indices = self._build_joint_cells(sample_indices, sample_paths)
        merged_cell_to_indices = self._merge_rare_joint_cells(raw_cell_to_indices)
        self._last_raw_joint_cell_counts = {
            cell: len(indices) for cell, indices in raw_cell_to_indices.items()
        }
        self._last_merged_joint_cell_counts = {
            cell: len(indices) for cell, indices in merged_cell_to_indices.items()
        }

        logger.info(
            "Joint-cell partitioning: %d raw cells -> %d merged cells "
            "(min_cell_count=%d, drop_order=%s)",
            len(self._last_raw_joint_cell_counts),
            len(self._last_merged_joint_cell_counts),
            self.joint_min_cell_count,
            self.joint_drop_order,
        )

        for attempt in range(self.max_retries):
            attempt_seed = None if self.seed is None else (self.seed + attempt)
            rng = np.random.default_rng(attempt_seed)

            # Assign samples from each merged joint cell
            client_assignments = self._assign_joint_cells_with_dirichlet(
                merged_cell_to_indices=merged_cell_to_indices,
                rng=rng,
            )

            # Validate minimum samples constraint
            if self._validate_assignments(client_assignments):
                if attempt > 0:
                    logger.info("Satisfied constraints after %d attempts.", attempt + 1)
                return client_assignments

        # Failed to satisfy constraints
        samples_per_client = [len(client_assignments[i]) for i in range(self.n_clients)]
        raise ValueError(
            f"Unable to satisfy min_samples_per_client={self.min_samples_per_client} "
            f"after {self.max_retries} attempts. Got: {samples_per_client}"
        )

    def _compute_client_splits(self) -> DatasetSplits:
        """Compute client splits using multivariate Dirichlet approach.

        Implements the complete splitting pipeline:
        1. Load VLM cluster labels from annotation files
        2. Build raw joint criterion cells per sample
        3. Merge rare cells via wildcard backoff (optional)
        4. Assign each merged cell to clients via Dirichlet(client) sampling
        5. Handle validation set according to valset_type configuration
        6. Build ClientSplit structures for each client

        Returns:
            DatasetSplits dictionary containing:
                - "clients": Dict[str, ClientSplit] mapping client IDs to their
                  train/val data
                - "global_val": Optional list of validation samples (if valset_type="global")

        Side Effects:
            - Populates self.cluster_labels and self.unique_labels
            - Populates joint-cell assignment diagnostics/preferences
            - Prints progress messages and statistics

        Examples:
            After calling get_client_splits():

            >>> splits = splitter.get_client_splits()
            >>> client_0_train = splits["clients"]["client_0"]["train"]
            >>> f"Client 0 has {len(client_0_train)} training samples"
        """

        self._load_cluster_labels()

        train_indices = list(range(len(self.train_dataset)))
        train_paths = [self._coerce_dataset_item_to_sample(self.train_dataset[i])["image"] for i in train_indices]

        logger.info(
            "Partitioning %d training samples among %d clients...",
            len(train_indices),
            self.n_clients,
        )

        train_assignments = self._partition_with_retry(train_indices, train_paths)

        samples_per_client = [len(train_assignments[i]) for i in range(self.n_clients)]
        logger.info("Training samples per client: %s", samples_per_client)
        logger.info(
            "Training distribution mean/std: %.1f / %.1f",
            float(np.mean(samples_per_client)),
            float(np.std(samples_per_client)),
        )
        # Handle validation set
        val_assignments = None
        global_val = None

        if self.valset_type == "global":
            if self.val_dataset is not None:
                global_val = [self._coerce_dataset_item_to_sample(self.val_dataset[i]) for i in range(len(self.val_dataset))]
        elif self.valset_type == "local":
            if self.val_dataset is not None:
                val_indices = list(range(len(self.val_dataset)))
                val_paths = [self._coerce_dataset_item_to_sample(self.val_dataset[i])["image"] for i in val_indices]
                val_assignments = self._partition_with_retry(val_indices, val_paths)

        # Build ClientSplit structures
        clients: Dict[str, ClientSplit] = {}

        for client_id in range(self.n_clients):
            # Training data
            train_data = [self._coerce_dataset_item_to_sample(self.train_dataset[idx]) for idx in train_assignments[client_id]]

            # Validation data
            val_data = []
            if self.valset_type == "local" and val_assignments is not None:
                val_data = [self._coerce_dataset_item_to_sample(self.val_dataset[idx]) for idx in val_assignments[client_id]]

            clients[f"client_{client_id}"] = {
                "train": train_data,
                "val": val_data,
            }

        return {
            "clients": clients,
            "global_val": global_val,
        }

    def get_preference_summary(self) -> Dict[str, Any]:
        """Return partition preferences summary for the current splitter state.

        Two summary modes are supported:
        1. Joint-cell mode (default partitioning path): summaries of merged
           joint cells and each cell's top client preference.
        2. Legacy per-criterion mode: available when client_preferences were
           sampled explicitly through _sample_client_preferences().

        Returns:
            Dictionary with either:
                - joint-cell fields (assignment_mode="joint_dirichlet"), or
                - legacy per-criterion top-preference fields.
            Returns an error dict if neither preference source exists.
        """
        if self.joint_cell_preferences:
            sorted_cells = sorted(
                self._last_merged_joint_cell_counts.items(),
                key=lambda entry: entry[1],
                reverse=True,
            )
            top_cells = []
            for cell, count in sorted_cells[: min(10, len(sorted_cells))]:
                preferences = self.joint_cell_preferences.get(cell)
                if preferences is None:
                    continue
                top_client = int(np.argmax(preferences))
                top_cells.append(
                    {
                        "cell": {
                            criterion: cell[idx] for idx, criterion in enumerate(self.CRITERIA)
                        },
                        "count": int(count),
                        "top_client": top_client,
                        "top_preference": float(preferences[top_client]),
                    }
                )

            return {
                "assignment_mode": "joint_dirichlet",
                "n_clients": self.n_clients,
                "heterogeneity": self.heterogeneity,
                "criteria": self.CRITERIA,
                "n_raw_joint_cells": len(self._last_raw_joint_cell_counts),
                "n_merged_joint_cells": len(self._last_merged_joint_cell_counts),
                "joint_min_cell_count": self.joint_min_cell_count,
                "joint_drop_order": self.joint_drop_order,
                "top_merged_cells": top_cells,
            }

        if not self.client_preferences:
            return {"error": "Preferences not yet sampled. Run get_client_splits() first."}

        summary = {
            "n_clients": self.n_clients,
            "heterogeneity": self.heterogeneity,
            "criteria": self.CRITERIA,
        }

        # For each criterion, get the top preference for each client
        for criterion in self.CRITERIA:
            top_prefs = []
            for client_id in range(self.n_clients):
                prefs = self.client_preferences[client_id][criterion]
                top_idx = np.argmax(prefs)
                top_label = self.unique_labels[criterion][top_idx]
                top_value = prefs[top_idx]
                top_prefs.append({
                    "client": client_id,
                    "label": top_label,
                    "preference": float(top_value),
                })
            summary[f"{criterion}_top_preferences"] = top_prefs

        return summary
