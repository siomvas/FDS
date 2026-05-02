"""
Simple example demonstrating multivariate Dirichlet partitioning.

This script shows basic usage of the MultivariateDirichletSplitter
and how to analyze the resulting partitions.
"""

from collections import Counter
import logging

from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter
from data_loading.dataset import BaseDataset

logger = logging.getLogger(__name__)


def analyze_client_distribution(splits, client_id: int, cluster_labels: dict):
    """Analyze the distribution of clusters for a specific client."""
    client_name = f"client_{client_id}"
    client_data = splits["clients"][client_name]["train"]

    logger.info("%s", f"{'=' * 60}")
    logger.info("Client %d Analysis", client_id)
    logger.info("%s", f"{'=' * 60}")
    logger.info("Total samples: %d", len(client_data))

    # Analyze each criterion
    for criterion in ["color", "main_object", "size", "location", "time"]:
        cluster_counter = Counter()

        for item in client_data:
            image_path = item["image"]
            cluster_label = cluster_labels[criterion].get(image_path)
            if cluster_label:
                cluster_counter[cluster_label] += 1

        logger.info("%s distribution:", criterion.upper())
        total = sum(cluster_counter.values())
        for label, count in cluster_counter.most_common(5):
            pct = 100 * count / total
            logger.info("%s: %d (%.1f%%)", label, count, pct)


def main():
    # Configuration
    dataset_name = "tiny_imagenet"
    base_path = "/vol/research/tubui/tiny-imagenet-200/"
    cluster_labels_dir = "/path/to/Documents/GitHub/FDS/data_partitioning/legacy_vlm/data/tiny_imagenet"

    logger.info("%s", "=" * 80)
    logger.info("MULTIVARIATE DIRICHLET PARTITIONING EXAMPLE")
    logger.info("%s", "=" * 80)

    # Load datasets
    logger.info("[Step 1] Loading datasets...")
    train_dataset = BaseDataset(dataset_name, base_path, "train")
    val_dataset = BaseDataset(dataset_name, base_path, "val")
    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Validation samples: %d", len(val_dataset))

    # Create splitter with high heterogeneity
    logger.info("[Step 2] Creating splitter (n_clients=6, α=0.1)...")
    splitter = MultivariateDirichletSplitter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_clients=6,
        heterogeneity=0.1,  # High heterogeneity
        cluster_labels_dir=cluster_labels_dir,
        valset_type="global",
        seed=42,
        min_samples_per_client=100,
    )

    # Get splits
    logger.info("[Step 3] Computing client partitions...")
    splits = splitter.get_client_splits()

    # Show basic statistics
    logger.info("[Step 4] Basic Statistics")
    logger.info("%s", "-" * 80)
    for client_id in range(splitter.n_clients):
        client_name = f"client_{client_id}"
        n_train = len(splits["clients"][client_name]["train"])
        n_val = len(splits["clients"][client_name]["val"])
        logger.info("%s: %d training samples, %d validation samples", client_name, n_train, n_val)

    # Show preference summary
    logger.info("[Step 5] Client Preferences")
    logger.info("%s", "-" * 80)
    summary = splitter.get_preference_summary()
    if summary.get("assignment_mode") == "joint_dirichlet":
        logger.info(
            "Joint cells: raw=%s merged=%s",
            summary.get("n_raw_joint_cells"),
            summary.get("n_merged_joint_cells"),
        )
        for cell_data in summary.get("top_merged_cells", []):
            logger.info(
                "Cell=%s count=%d top_client=%d pref=%.3f",
                cell_data["cell"],
                cell_data["count"],
                cell_data["top_client"],
                cell_data["top_preference"],
            )
    else:
        for criterion in splitter.CRITERIA:
            logger.info("%s - Top preferences:", criterion.upper())
            for client_data in summary[f"{criterion}_top_preferences"]:
                logger.info(
                    "Client %s: %s (preference=%.3f)",
                    client_data["client"],
                    client_data["label"],
                    client_data["preference"],
                )

    # Detailed analysis for first 2 clients
    logger.info("[Step 6] Detailed Client Analysis")
    logger.info("%s", "-" * 80)

    for client_id in range(min(2, splitter.n_clients)):
        analyze_client_distribution(splits, client_id, splitter.cluster_labels)

    # Show global validation set
    if splits["global_val"]:
        logger.info("[Step 7] Global Validation Set")
        logger.info("%s", "-" * 80)
        logger.info("Total samples: %d", len(splits["global_val"]))

    logger.info("%s", "=" * 80)
    logger.info("Example completed successfully.")
    logger.info("%s", "=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
