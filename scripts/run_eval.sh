#!/bin/bash

# Run evaluation script with default settings
python run_eval.py \
    --model_path "path/to/your/model" \
    --output_dir "./outputs" \
    --configs gsm8k math olympiadbench omnimath

# Uncomment below for voting-based evaluation
# python run_eval.py \
#     --model_path "path/to/your/model" \
#     --output_dir "./outputs" \
#     --configs gsm8k math olympiadbench omnimath \
#     --use_voting \
#     --voting_n 8
