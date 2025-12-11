import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict

from data_partitioning.base_splitter import BaseDatasetSplitter, DatasetSplits, ClientSplit
from data_partitioning.dataset import BaseDataset


class MultivariateDirichletSplitter(BaseDatasetSplitter):
    """
    Splits dataset using multiple VLM-generated clustering criteria with
    Dirichlet-based weighted random assignment.

    Each client has preference vectors for each criterion (color, object, size,
    location, time), sampled from Dirichlet distribution. Samples are assigned
    based on softmax probabilities derived from affinity scores.
    """

    CRITERIA = ["color", "main_object", "size", "location", "time"]
    CRITERION_TO_DIR = {
        "color": "output_color_train",
        "main_object": "output_main_object_train",
        "size": "output_size_train",
        "location": "output_location_train",
        "time": "output_time_train",
    }

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
    ):
        """
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            n_clients: Number of clients to split data among
            heterogeneity: Alpha parameter for Dirichlet distribution
                          Lower α (e.g., 0.1) → high heterogeneity (skewed preferences)
                          Higher α (e.g., 10.0) → low heterogeneity (uniform preferences)
            cluster_labels_dir: Path to directory containing cluster label JSONL files
            valset_type: "global", "local", or "none"
            seed: Random seed for reproducibility
            min_samples_per_client: Minimum samples each client must have
            max_retries: Maximum attempts to satisfy min_samples constraint
            temperature: Temperature parameter for softmax (default 1.0)
        """
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_clients=n_clients,
            heterogeneity=heterogeneity,
            valset_type=valset_type,
            seed=seed,
        )

        self.cluster_labels_dir = Path(cluster_labels_dir)
        self.min_samples_per_client = min_samples_per_client
        self.max_retries = max_retries
        self.temperature = temperature

        # Will be populated during splitting
        self.cluster_labels: Dict[str, Dict[str, str]] = {}
        self.unique_labels: Dict[str, List[str]] = {}
        self.client_preferences: Dict[int, Dict[str, np.ndarray]] = {}
    def _load_cluster_labels(self) -> None:

        print("Loading cluster labels from JSONL files...")

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
                        image_path = entry["image"]
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
                        image_path = entry["image"]
                        cluster_label = entry["class"]
                        criterion_labels[image_path] = cluster_label

            self.cluster_labels[criterion] = criterion_labels

            # Get unique labels for this criterion
            unique_labels = sorted(set(criterion_labels.values()))
            self.unique_labels[criterion] = unique_labels

            print(f"  {criterion}: {len(unique_labels)} unique labels, "
                  f"{len(criterion_labels)} images")

    def _sample_client_preferences(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)

        print(f"\nSampling client preferences with α={self.heterogeneity}...")

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
                    print(f"  Client 0, {criterion}: top={top_label} ({preferences[top_idx]:.3f}), "
                          f"min={preferences.min():.3f}, std={preferences.std():.3f}")

        # Print aggregate statistics
        print(f"\n  Preference Statistics Across All Clients:")
        print(f"    Max preference - Mean: {np.mean(all_max_prefs):.4f}, Std: {np.std(all_max_prefs):.4f}")
        print(f"    Min preference - Mean: {np.mean(all_min_prefs):.4f}, Std: {np.std(all_min_prefs):.4f}")
        print(f"    Std deviation - Mean: {np.mean(all_stds):.4f}, Std: {np.std(all_stds):.4f}")

    def _get_cluster_label_for_sample(self, image_path: str, criterion: str) -> Optional[str]:
        return self.cluster_labels[criterion].get(image_path)

    def _compute_affinity_score(self, image_path: str, client_id: int) -> float:
        """
        Compute affinity score between a sample and a client.

        Affinity = Σ(preference[criterion][sample's label]) across all criteria

        Args:
            image_path: Path to the sample image
            client_id: ID of the client

        Returns:
            Affinity score (sum of preferences across all criteria)
        """
        affinity = 0.0

        for criterion in self.CRITERIA:
            cluster_label = self._get_cluster_label_for_sample(image_path, criterion)

            if cluster_label is None:
                # If label missing, use uniform preference (1/n_labels)
                affinity += 1.0 / len(self.unique_labels[criterion])
                continue

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
        """
        Assign samples to clients using softmax probabilities based on affinity scores.

        Args:
            sample_indices: List of sample indices to assign
            sample_paths: List of corresponding image paths

        Returns:
            Dict mapping client_id to list of assigned sample indices
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
        print(f"\n  Assignment Statistics (α={self.heterogeneity}):")
        if debug_affinities:
            all_affinities = np.array(debug_affinities)
            print(f"    Affinity scores (first 10 samples):")
            print(f"      Mean: {all_affinities.mean():.4f}, Std: {all_affinities.std():.4f}")
            print(f"      Range: [{all_affinities.min():.4f}, {all_affinities.max():.4f}]")

            all_probs = np.array(debug_probabilities)
            print(f"    Softmax probabilities (first 10 samples):")
            print(f"      Max prob mean: {np.mean(debug_max_probs):.4f}, Std: {np.std(debug_max_probs):.4f}")
            print(f"      Entropy (avg): {-np.mean([np.sum(p * np.log(p + 1e-10)) for p in all_probs]):.4f}")

        return client_assignments

    def _validate_assignments(self, client_assignments: Dict[int, List[int]]) -> bool:
        """
        Check if all clients have at least min_samples_per_client samples.

        Args:
            client_assignments: Dict mapping client_id to sample indices

        Returns:
            True if valid, False otherwise
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
        """
        Partition samples with retry logic to satisfy min_samples constraint.

        Args:
            sample_indices: List of sample indices
            sample_paths: List of corresponding image paths

        Returns:
            Dict mapping client_id to assigned sample indices

        Raises:
            ValueError: If unable to satisfy constraints after max_retries
        """
        for attempt in range(self.max_retries):
            # Resample preferences with different seed each attempt
            if attempt > 0:
                if self.seed is not None:
                    np.random.seed(self.seed + attempt)
                self._sample_client_preferences()

            # Assign samples
            client_assignments = self._assign_samples_with_softmax(sample_indices, sample_paths)

            # Validate
            if self._validate_assignments(client_assignments):
                if attempt > 0:
                    print(f"  Satisfied constraints after {attempt + 1} attempts")
                return client_assignments

        # Failed to satisfy constraints
        samples_per_client = [len(client_assignments[i]) for i in range(self.n_clients)]
        raise ValueError(
            f"Unable to satisfy min_samples_per_client={self.min_samples_per_client} "
            f"after {self.max_retries} attempts. Got: {samples_per_client}"
        )

    def _compute_client_splits(self) -> DatasetSplits:
        """
        Compute client splits using multivariate Dirichlet approach.

        Returns:
            DatasetSplits containing client assignments
        """

        self._load_cluster_labels()
        self._sample_client_preferences()

        train_indices = list(range(len(self.train_dataset)))
        train_paths = [self.train_dataset[i][0] for i in train_indices]  # (path, label)

        print(f"\nPartitioning {len(train_indices)} training samples among {self.n_clients} clients...")

        train_assignments = self._partition_with_retry(train_indices, train_paths)

        samples_per_client = [len(train_assignments[i]) for i in range(self.n_clients)]
        print(f"Training samples per client: {samples_per_client}")
        print(f"  Mean: {np.mean(samples_per_client):.1f}, Std: {np.std(samples_per_client):.1f}")

        # Handle validation set
        val_assignments = None
        global_val = None

        if self.valset_type == "global":
            if self.val_dataset is not None:
                global_val = [
                    {"image": self.val_dataset[i][0], "label": self.val_dataset[i][1]}
                    for i in range(len(self.val_dataset))
                ]
        elif self.valset_type == "local":
            if self.val_dataset is not None:
                val_indices = list(range(len(self.val_dataset)))
                val_paths = [self.val_dataset[i][0] for i in val_indices]
                val_assignments = self._partition_with_retry(val_indices, val_paths)

        # Build ClientSplit structures
        clients: Dict[str, ClientSplit] = {}

        for client_id in range(self.n_clients):
            # Training data
            train_data = [
                {"image": self.train_dataset[idx][0], "label": self.train_dataset[idx][1]}
                for idx in train_assignments[client_id]
            ]

            # Validation data
            val_data = []
            if self.valset_type == "local" and val_assignments is not None:
                val_data = [
                    {"image": self.val_dataset[idx][0], "label": self.val_dataset[idx][1]}
                    for idx in val_assignments[client_id]
                ]

            clients[f"client_{client_id}"] = {
                "train": train_data,
                "val": val_data,
            }

        return {
            "clients": clients,
            "global_val": global_val,
        }

    def get_preference_summary(self) -> Dict[str, Any]:
        """
        Get a summary of client preferences for analysis.

        Returns:
            Dictionary with preference statistics
        """
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
