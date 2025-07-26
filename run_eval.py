# Usage example:
# python run_eval.py path/to/config.yaml
# or
# python run_eval.py path/to/config.json

import sys
import os
import json
import yaml
import numpy as np
from collections import Counter
from datasets import load_from_disk
from datetime import datetime
from src.utils.data import parse_from_boxed, reformat_model_name
from src.modules.verifier import AutoVerifier
from src.modules.client import OpenaiClient
from multiprocessing import Pool, Manager
from tqdm import tqdm

# os.environ["http_proxy"] = ""
# os.environ["https_proxy"] = ""


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
    if not (config.get("api_endpoint") and config.get("model")):
        raise ValueError("api_endpoint and model must be provided in config")
    config["splits"] = config.get("splits") or [
        "gsm8k",
        "math",
        "olympiadbench",
        "omnimath",
    ]
    config["verifier_type"] = config.get("verifier_type") or "sequential"
    config["output_dir"] = config.get("output_dir") or "/raid/vinh/resources/results"
    config["cache_dir"] = (
        config.get("cache_dir") or "/raid/vinh/resources/results/cache"
    )
    config["use_voting"] = config.get("use_voting") or False
    config["voting_n"] = config.get("voting_n") or 8
    config["dataset_path"] = config.get("dataset_path") or "Qwen/ProcessBench"
    config["sample_size"] = config.get("sample_size") or 100
    config["verifier_kwargs"] = config.get("verifier_kwargs") or {}
    config["construction_kwargs"] = config.get("construction_kwargs") or {}
    config["generation_kwargs"] = config.get("generation_kwargs") or {}
    config["num_workers"] = config.get("num_workers") or 16
    return config


def verify_sample(verifier, sample, config):
    # Do the verification and post-processing for a single sample
    result = verifier._verify_one(
        sample["uid"],
        sample,
        config["construction_kwargs"],
        config["generation_kwargs"],
    )
    d = sample.copy()
    # Post-processing for prediction and match
    if not config["use_voting"]:
        pred = parse_from_boxed(result["generated_critique"][0])
        try:
            pred = int(pred)
        except:
            pred = None
    else:
        preds = [parse_from_boxed(e) for e in result["generated_critique"]]
        preds = [e for e in preds if e is not None]
        if len(preds) == 0:
            pred = None
        else:
            pred = Counter(preds).most_common(1)[0][0]
            try:
                pred = int(pred)
            except:
                pred = None
    d["generated_critique"] = result["generated_critique"]
    d["graph"] = result["graph"]
    d["time"] = result["time"]
    d["prediction"] = pred
    d["match"] = pred == d["label"]
    return d


def verify_one_helper(args):
    verifier, sample, config, draft_path, lock = args
    d = verify_sample(verifier, sample, config)
    with lock:
        with open(draft_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return d


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_eval.py <config.yaml|config.json>")
        sys.exit(1)
    config_path = sys.argv[1]
    config = load_config(config_path)

    # Initialize OpenAI client
    client = OpenaiClient(
        endpoint=config["api_endpoint"],
        model=config["model"],
        api_key=os.getenv("API_KEY"),
    )

    # Initialize appropriate verifier
    verifier = AutoVerifier.from_type(
        verifier_type=config["verifier_type"],
        client=client,
        **config["verifier_kwargs"],
    )

    if not config["use_voting"]:
        config["generation_kwargs"]["n"] = 1
    else:
        config["generation_kwargs"]["n"] = config["voting_n"]

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for split in config["splits"]:
        output_dir = os.path.join(
            config["output_dir"],
            reformat_model_name(config["model"]),
            config["verifier_type"],
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

        input_data = [dict(sample, uid=i) for i, sample in enumerate(input_data)]
        # --- Cache logic ---
        cache_dir = config.get("cache_dir")
        if not cache_dir:
            cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        draft_path = os.path.join(cache_dir, f"{split}.draft.jsonl")
        processed_uids = set()
        res_data = []
        if os.path.exists(draft_path):
            with open(draft_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        processed_uids.add(sample["uid"])
                    except Exception:
                        continue
        # Only process samples whose uid is not in processed_uids
        manager = Manager()
        lock = manager.Lock()
        tasks = [
            (verifier, sample, config, draft_path, lock)
            for sample in input_data
            if sample["uid"] not in processed_uids
        ]
        if not tasks:
            print(f"No new samples to process for {split}.")
        if tasks:
            with Pool(processes=config["num_workers"]) as pool:
                res_data.extend(
                    list(
                        tqdm(
                            pool.imap_unordered(verify_one_helper, tasks),
                            total=len(tasks),
                        )
                    )
                )
        res_data.sort(key=lambda d: d["uid"])
        error_data = [e for e in res_data if e["label"] != -1]
        correct_data = [e for e in res_data if e["label"] == -1]
        with open(os.path.join(output_dir, f"{split}_error.jsonl"), "w") as f:
            for e in error_data:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(output_dir, f"{split}_correct.jsonl"), "w") as f:
            for e in correct_data:
                f.write(json.dumps(e) + "\n")
        acc1 = np.mean([e["match"] for e in error_data]) * 100
        acc2 = np.mean([e["match"] for e in correct_data]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        result_str = (
            f"{split} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}"
        )
        print(result_str)
        with open(os.path.join(output_dir, "results.txt"), "a") as f:
            f.write(result_str + "\n")


if __name__ == "__main__":
    main()
