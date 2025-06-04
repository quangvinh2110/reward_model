import argparse
import numpy as np
import os
import json
from collections import Counter
from datasets import load_dataset
import re

from src.pipelines.generative_rm import GenerativeRewardModel


def extract_answer(solution_text: str):
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        choices=["gsm8k", "math", "olympiadbench", "omnimath"],
        help="List of sub-datasets to evaluate on",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model or model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--use_voting",
        action="store_true",
        help="Whether to use voting-based evaluation",
    )
    parser.add_argument(
        "--voting_n",
        type=int,
        default=8,
        help="Number of voting samples",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Qwen/ProcessBench",
        help="Path to the dataset. Can be a local path or a HuggingFace dataset name.",
    )
    args = parser.parse_args()

    args.model_name = os.path.basename(args.model_name_or_path)

    # Initialize GenerativeRewardModel
    model = GenerativeRewardModel(
        backend="vllm", model_name_or_path=args.model_name_or_path
    )

    if not args.use_voting:
        generation_kwargs = {
            "temperature": 0.0,
            "max_tokens": 32768 if "QwQ" in args.model_name_or_path else 8192,
        }
    else:
        if "Qwen2.5-Math" in args.model_name_or_path:
            generation_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "n": args.voting_n,
                "max_tokens": 32768 if "QwQ" in args.model_name_or_path else 8192,
            }
        else:
            generation_kwargs = {
                "temperature": 1,
                "top_p": 0.9,
                "n": args.voting_n,
                "max_tokens": 32768 if "QwQ" in args.model_name_or_path else 8192,
            }

    if args.configs is None:
        args.configs = ["gsm8k", "math", "olympiadbench", "omnimath"]

    for config in args.configs:
        if not args.use_voting:
            output_dir = os.path.join(args.output_dir, args.model_name)
        else:
            output_dir = os.path.join(args.output_dir, f"{args.model_name}_voting")
        os.makedirs(output_dir, exist_ok=True)

        input_data = load_dataset(args.dataset_path, split=config)

        # Prepare problem-solution pairs
        problem_solution_pairs = [(e["problem"], e["steps"]) for e in input_data]

        # Generate critiques using the model
        generated_critiques = model(problem_solution_pairs, **generation_kwargs)

        res_data = []
        for i in range(len(input_data)):
            d = input_data[i].copy()

            if not args.use_voting:
                pred = extract_answer(generated_critiques[i])
                try:
                    pred = int(pred)
                except:
                    pred = None
            else:
                # For voting, we need to handle multiple outputs
                preds = [
                    extract_answer(e)
                    for e in generated_critiques[i : i + args.voting_n]
                ]
                preds = [e for e in preds if e is not None]
                if len(preds) == 0:
                    pred = None
                else:
                    pred = Counter(preds).most_common(1)[0][0]
                    try:
                        pred = int(pred)
                    except:
                        pred = None

            d["generated_critique"] = (
                generated_critiques[i]
                if not args.use_voting
                else generated_critiques[i : i + args.voting_n]
            )
            d["prediction"] = pred
            d["match"] = pred == d["label"]

            res_data.append(d)

        error_data = [e for e in res_data if e["label"] != -1]
        correct_data = [e for e in res_data if e["label"] == -1]

        with open(os.path.join(output_dir, f"{config}_error.jsonl"), "w") as f:
            for e in error_data:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(output_dir, f"{config}_correct.jsonl"), "w") as f:
            for e in correct_data:
                f.write(json.dumps(e) + "\n")

        acc1 = np.mean([e["match"] for e in error_data]) * 100
        acc2 = np.mean([e["match"] for e in correct_data]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        print(f"{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}")


if __name__ == "__main__":
    main()
