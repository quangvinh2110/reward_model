import os
import random
from datasets import load_from_disk

# The splits to process
splits = ["gsm8k", "math", "olympiadbench", "omnimath"]
input_base = "/raid/vinh/resources/datasets/ProcessBench"
output_base = "/raid/vinh/resources/datasets/ProcessBench-250"

os.makedirs(output_base, exist_ok=True)


def compute_n_verification(example):
    return {
        "n_verification": (
            example["label"] if example["label"] > 0 else len(example["steps"])
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


for split in splits:
    print(f"Processing {split}...")
    ds_path = os.path.join(input_base, split)
    dataset = load_from_disk(ds_path)
    sampled = sample_uniform(dataset)
    out_path = os.path.join(output_base, split)
    sampled.save_to_disk(out_path)
    print(f"Saved {split} to {out_path}")

print("All splits processed and saved to ProcessBench-250.")
