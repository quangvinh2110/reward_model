import os
import random
from datasets import load_from_disk
import argparse


def compute_n_verification(example):
    return {
        "n_verification": (
            example["label"] + 1 if example["label"] >= 0 else len(example["steps"])
        )
    }


def sample_uniform(dataset, n_samples=250):
    # Add n_verification column
    dataset = dataset.map(compute_n_verification)
    # Group indices by n_verification value
    from collections import defaultdict

    groups = defaultdict(list)
    for idx, example in enumerate(dataset):
        groups[example["n_verification"]].append(idx)
    unique_n = list(groups.keys())
    n_groups = len(unique_n)
    samples_per_group = n_samples // n_groups
    remainder = n_samples % n_groups
    selected_indices = []
    for i, n in enumerate(unique_n):
        group = groups[n]
        k = samples_per_group + (1 if i < remainder else 0)
        if len(group) < k:
            k = len(group)
        selected_indices.extend(random.sample(group, k))
    # If we have less than n_samples, prioritize larger n_verification
    if len(selected_indices) < n_samples:
        remaining = list(set(range(len(dataset))) - set(selected_indices))
        remaining_sorted = sorted(
            remaining, key=lambda idx: dataset[idx]["n_verification"], reverse=True
        )
        selected_indices.extend(remaining_sorted[: n_samples - len(selected_indices)])
    return dataset.select(selected_indices)


def sample_largest(dataset, n_samples=250):
    dataset = dataset.map(compute_n_verification)
    # Sort all indices by n_verification descending
    sorted_indices = sorted(
        range(len(dataset)),
        key=lambda idx: dataset[idx]["n_verification"],
        reverse=True,
    )
    selected_indices = sorted_indices[:n_samples]
    return dataset.select(selected_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample ProcessBench splits with different strategies."
    )
    parser.add_argument(
        "--mode",
        choices=["uniform", "largest"],
        default="uniform",
        help="Sampling strategy: uniform or largest",
    )
    parser.add_argument(
        "--num_samples", type=int, default=250, help="Number of samples per split"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # The splits to process
    splits = ["gsm8k", "math", "olympiadbench", "omnimath"]
    input_base = "/your/path/to/datasets/ProcessBench"
    output_base = f"/your/path/to/datasets/ProcessBench-{args.num_samples}"

    os.makedirs(output_base, exist_ok=True)

    for split in splits:
        print(f"Processing {split}...")
        ds_path = os.path.join(input_base, split)
        dataset = load_from_disk(ds_path)
        sampled = (
            sample_uniform(dataset, n_samples=args.num_samples)
            if args.mode == "uniform"
            else sample_largest(dataset, n_samples=args.num_samples)
        )
        out_path = os.path.join(output_base, split)
        sampled.save_to_disk(out_path)
        print(f"Saved {split} to {out_path}")

    print(f"All splits processed and saved to {output_base}.")
