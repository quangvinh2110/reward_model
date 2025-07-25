import sys
import os
import yaml
import json
import networkx as nx
from networkx.readwrite import node_link_data
from datasets import load_from_disk
from src.modules.client import OpenaiClient
from src.modules.constructor import AutoConstructor
from datasets import Dataset
from tqdm import tqdm
from multiprocessing import Pool, Manager

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def load_config(config_path):
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")
    # Set defaults if not present in config
    if not (config["api_endpoint"] and config["model"]):
        raise ValueError("api_endpoint and model must be provided in config")
    config["splits"] = config.get("splits", None) or [
        "gsm8k",
        "math",
        "olympiadbench",
        "omnimath",
    ]
    config["constructor_type"] = config.get("constructor_type", None) or "targeted"
    config["output_dir"] = (
        config.get("output_dir", None) or "/raid/vinh/resources/datasets"
    )
    config["dataset_path"] = config.get("dataset_path", None) or "Qwen/ProcessBench"
    config["sample_size"] = config.get("sample_size", None) or 100
    config["construction_kwargs"] = config.get("construction_kwargs", None) or {}
    config["generation_kwargs"] = config.get("generation_kwargs", None) or {}
    config["num_workers"] = config.get("num_workers", None) or 16
    return config


def build_graph_for_sample(constructor, sample, construction_kwargs, generation_kwargs):
    solution_graph = nx.DiGraph()
    for step_idx, step in enumerate(sample["steps"]):
        solution_graph.add_node(step_idx, content=step, resolved=False)
    solution_graph.nodes[0]["resolved"] = True
    constructor(
        problem=sample["problem"],
        solution_graph=solution_graph,
        construction_kwargs=construction_kwargs,
        generation_kwargs=generation_kwargs,
    )
    return json.dumps(
        node_link_data(solution_graph, edges="edges"),
        ensure_ascii=False,
    )


def build_graph_for_sample_helper(args):
    constructor, sample, construction_kwargs, generation_kwargs, draft_path, lock = args
    graph_json = build_graph_for_sample(
        constructor, sample, construction_kwargs, generation_kwargs
    )
    d = sample.copy()
    d["graph"] = graph_json
    # Write to JSONL file
    with lock:
        with open(draft_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return d


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_graph.py <config.yaml|config.json>")
        sys.exit(1)
    config_path = sys.argv[1]
    config = load_config(config_path)

    client = OpenaiClient(
        endpoint=config["api_endpoint"],
        model=config["model"],
        api_key=os.getenv("API_KEY"),
    )
    constructor = AutoConstructor.from_type(
        config.get("constructor_type", "targeted"), client=client
    )
    for split in config["splits"]:
        output_dir = os.path.join(
            config["output_dir"],
            config.get("constructor_type", "targeted"),
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f)
        dataset = load_from_disk(os.path.join(config["dataset_path"], split)).shuffle(
            seed=42
        )
        if config["sample_size"] is not None and config["sample_size"] > 0:
            num_samples = min(config["sample_size"], len(dataset))
            input_data = dataset.select(range(num_samples))
        else:
            input_data = dataset
        manager = Manager()
        lock = manager.Lock()
        draft_path = os.path.join(output_dir, f"{split}.draft.jsonl")
        # Check for existing draft and determine how many samples are already processed
        start_idx = 0
        if os.path.exists(draft_path):
            with open(draft_path, "r", encoding="utf-8") as f:
                for start_idx, _ in enumerate(f, 1):
                    pass
        # Only process samples that have not been processed yet
        if start_idx >= len(input_data):
            print(
                f"Draft for {split} already complete with {start_idx} samples. Skipping."
            )
            continue
        tasks = [
            (
                constructor,
                input_data[i],
                config["construction_kwargs"],
                config["generation_kwargs"],
                draft_path,
                lock,
            )
            for i in range(start_idx, len(input_data))
        ]
        if not tasks:
            print(f"No new samples to process for {split}.")
            continue
        with Pool(processes=config["num_workers"]) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(build_graph_for_sample_helper, tasks),
                    total=len(tasks),
                )
            )
        # Convert results to HuggingFace Dataset and save
        hf_dataset = Dataset.from_list(results)
        hf_dataset.save_to_disk(os.path.join(output_dir, split))
        print(
            f"Saved HuggingFace dataset for {split} to {os.path.join(output_dir, f'{split}') }"
        )


if __name__ == "__main__":
    main()
