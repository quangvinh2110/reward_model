import sys
import os
import yaml
import json
from datetime import datetime
from datasets import load_from_disk
from utils import write_jsonl
from src.modules.client import OpenaiClient
from src.modules.constructor import AutoConstructor

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
    config["splits"] = config.get("splits") or [
        "gsm8k",
        "math",
        "olympiadbench",
        "omnimath",
    ]
    config["constructor_type"] = config.get("constructor_type") or "targeted"
    config["output_dir"] = config.get("output_dir") or "/raid/vinh/resources/results"
    config["dataset_path"] = config.get("dataset_path") or "Qwen/ProcessBench"
    config["sample_size"] = config.get("sample_size") or 100
    config["construction_kwargs"] = config.get("construction_kwargs") or {}
    config["generation_kwargs"] = config.get("generation_kwargs") or {}
    config["num_workers"] = config.get("num_workers") or 16
    return config


def build_graph_for_sample(constructor, sample, construction_kwargs, generation_kwargs):
    import networkx as nx
    from networkx.readwrite import node_link_data

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
    return json.dumps(node_link_data(solution_graph, edges="edges"), ensure_ascii=False)


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
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for split in config["splits"]:
        output_dir = os.path.join(
            config["output_dir"],
            config["model"],
            config.get("constructor_type", "targeted"),
            current_time,
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
        results = []
        for i in range(len(input_data)):
            sample = input_data[i]
            graph_json = build_graph_for_sample(
                constructor,
                sample,
                config["construction_kwargs"],
                config["generation_kwargs"],
            )
            d = sample.copy()
            d["graph"] = graph_json
            results.append(d)
        write_jsonl(results, os.path.join(output_dir, f"{split}_graphs.jsonl"))
        print(
            f"Saved graphs for {split} to {os.path.join(output_dir, f'{split}_graphs.jsonl')}"
        )


if __name__ == "__main__":
    main()
