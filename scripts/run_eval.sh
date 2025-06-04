#!/bin/bash

# Run evaluation script with default settings
python run_eval.py \
    --model_name_or_path "path/to/your/model" \
    --output_dir "./outputs" \
    --configs gsm8k math olympiadbench omnimath \
    --dataset_path "Qwen/ProcessBench"

# Uncomment below for voting-based evaluation
# python run_eval.py \
#     --model_name_or_path "path/to/your/model" \
#     --output_dir "./outputs" \
#     --configs gsm8k math olympiadbench omnimath \
#     --dataset_path "Qwen/ProcessBench" \
#     --use_voting \
#     --voting_n 8
