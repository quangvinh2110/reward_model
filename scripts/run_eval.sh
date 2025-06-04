#!/bin/bash

# Run evaluation with vllm backend (default)
# python /raid/vinh/reward_model/run_eval.py \
#     --model_name_or_path "path/to/your/model" \
#     --output_dir "./outputs" \
#     --configs gsm8k math olympiadbench omnimath \
#     --dataset_path "Qwen/ProcessBench"

# Run evaluation with transformers backend
# python /raid/vinh/reward_model/run_eval.py \
#     --model_name_or_path "path/to/your/model" \
#     --model_backend "transformers" \
#     --output_dir "./outputs" \
#     --configs gsm8k math olympiadbench omnimath \
#     --dataset_path "Qwen/ProcessBench"

# Run evaluation with vllm_api backend
python /raid/vinh/reward_model/run_eval.py \
    --model_name_or_path "/workspace/home/NLP_CORE/HUB_LLM/Meta-Llama-3.3-70B-Instruct" \
    --model_backend "vllm_api" \
    --api_endpoint "http://10.254.138.192:9002/v1" \
    --served_model_name "Meta-Llama-3-70B-Instruct" \
    --output_dir "/raid/vinh/resources/results" \
    --configs gsm8k math olympiadbench omnimath \
    --dataset_path "/raid/vinh/resources/datasets/ProcessBench"

# Run evaluation with tgi_api backend
# python /raid/vinh/reward_model/run_eval.py \
#     --model_name_or_path "path/to/your/model" \
#     --model_backend "tgi_api" \
#     --api_endpoint "http://localhost:8080" \
#     --served_model_name "your_model_name" \
#     --output_dir "./outputs" \
#     --configs gsm8k math olympiadbench omnimath \
#     --dataset_path "Qwen/ProcessBench"

# Uncomment below for voting-based evaluation
# python /raid/vinh/reward_model/run_eval.py \
#     --model_name_or_path "path/to/your/model" \
#     --output_dir "./outputs" \
#     --configs gsm8k math olympiadbench omnimath \
#     --dataset_path "Qwen/ProcessBench" \
#     --use_voting \
#     --voting_n 8
