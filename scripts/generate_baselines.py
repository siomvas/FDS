"""
generate_baselines.py
Generates baseline partitions (Random, Dirichlet, VLM) and saves them as JSON
to match the format of embedding-based partitions.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loading.dataset import BaseDataset
from partitioning.random_splitter import RandomSplitter
from partitioning.dirichlet_splitter import DirichletSplitter
from partitioning.multivariate_dirichlet_splitter import MultivariateDirichletSplitter

def save_split(splitter, output_path, method_name, alpha, n_clients):
    splits = splitter.get_client_splits()
    
    # Convert to serializable format (paths only)
    serializable_splits = {"clients": {}}
    for client_id, data in splits["clients"].items():
        # Extract just the image paths/IDs
        serializable_splits["clients"][client_id] = {
            "train": [item["image"] for item in data["train"]],
            "val": [item["image"] for item in data["val"]]
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_splits, f, indent=2)
    print(f"✓ Saved {method_name} (n={n_clients}, a={alpha}) to {output_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_clients", type=int, nargs="+", default=[10])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--cluster_labels_dir", type=str, default=None, 
                        help="Path to VLM annotations directory (required for VLM splits)")
    args = parser.parse_args()

    # Load Dataset
    train_dataset = BaseDataset(args.dataset, args.base_path, "train")
    val_dataset = BaseDataset(args.dataset, args.base_path, "val")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for n in args.n_clients:
            # 1. Random (IID)
            splitter = RandomSplitter(train_dataset, val_dataset, n, heterogeneity=0.0, seed=seed)
            save_split(splitter, output_dir / f"random_n{n}_seed{seed}.json", "Random", 0.0, n)

            # 2. Dirichlet (Non-IID)
            for alpha in args.alphas:
                splitter = DirichletSplitter(train_dataset, val_dataset, n, heterogeneity=alpha, seed=seed)
                save_split(splitter, output_dir / f"dirichlet_n{n}_alpha{alpha}_seed{seed}.json", "Dirichlet", alpha, n)
            
            # 3. VLM
            if args.cluster_labels_dir:
                vlm_path = Path(args.cluster_labels_dir)
                if vlm_path.exists():
                    for alpha in args.alphas:
                        try:
                            print(f"Generating VLM split (alpha={alpha})...")
                            splitter = MultivariateDirichletSplitter(
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                n_clients=n,
                                heterogeneity=alpha,
                                cluster_labels_dir=str(vlm_path),
                                valset_type="global",
                                seed=seed,
                                min_samples_per_client=10 # Prevent empty clients
                            )
                            # Filename format matches eval_config regex: vlm_n{N}_alpha{A}_seed{S}.json
                            save_split(splitter, output_dir / f"vlm_n{n}_alpha{alpha}_seed{seed}.json", "VLM", alpha, n)
                        except Exception as e:
                            print(f"✗ Failed VLM (a={alpha}): {e}")
                            traceback.print_exc()
                else:
                    print(f"Warning: cluster_labels_dir {vlm_path} does not exist. Skipping VLM.")
            else:
                print("Skipping VLM (no cluster_labels_dir provided).")

if __name__ == "__main__":
    main()