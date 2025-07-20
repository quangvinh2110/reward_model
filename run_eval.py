import argparse
import numpy as np
import os
import json
from collections import Counter
from datasets import load_from_disk
from datetime import datetime
from src.utils.data import parse_from_boxed
from src.modules.verifier import AutoVerifier
from src.modules.client import OpenaiClient


os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def parse_args():
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
        "--verifier_type",
        type=str,
        default="sequential",
        choices=["sequential", "stepwise", "parc", "logicflow"],
        help="Type of verifier to use (sequential: full solution at once, stepwise: step by step)",
    )
    parser.add_argument(
        "--api_endpoint",
        type=str,
        default=None,
        help="API endpoint URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Name of the model served at the API endpoint (required for openai clients)",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling parameter (default: 20)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate (default: 8192)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples to evaluate from each dataset (default: 100). Use -1 to evaluate the entire dataset.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Whether to enable thinking mode in tokenizer (default: False)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize OpenAI client
    client = OpenaiClient(
        endpoint=args.api_endpoint,
        model=args.model,
        api_key=os.getenv("API_KEY"),
    )

    # Initialize appropriate verifier
    verifier = AutoVerifier.from_type(
        verifier_type=args.verifier_type,
        client=client,
        show_progress=True,
    )

    if not args.use_voting:
        generation_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        }
    else:
        generation_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "n": args.voting_n,
            "max_tokens": args.max_tokens,
        }

    if args.configs is None:
        args.configs = ["gsm8k", "math", "olympiadbench", "omnimath"]

    for config in args.configs:
        # Get current datetime for output directory
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if not args.use_voting:
            output_dir = os.path.join(
                args.output_dir, args.model, args.verifier_type, current_time
            )
        else:
            output_dir = os.path.join(
                args.output_dir,
                f"{args.model}_voting_{args.voting_n}",
                args.verifier_type,
                current_time,
            )
        os.makedirs(output_dir, exist_ok=True)

        dataset = load_from_disk(os.path.join(args.dataset_path, config)).shuffle(
            seed=42
        )
        if args.sample_size is not None and args.sample_size > 0:
            dataset = dataset.select(range(args.sample_size))

        # Select a subset of the dataset if --sample is set to a positive integer
        if args.sample_size is not None and args.sample_size > 0:
            num_samples = min(args.sample_size, len(dataset))
            input_data = dataset.select(range(num_samples))
        else:
            # Use the full dataset when --sample is -1 or 0
            input_data = dataset

        # Generate critiques using the verifier
        generated_results = verifier(
            input_data.to_list(), num_workers=16, **generation_kwargs
        )

        res_data = []
        for i in range(len(input_data)):
            d = input_data[i].copy()
            result = generated_results[i]
            if not args.use_voting:
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
