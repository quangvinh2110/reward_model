#!/bin/bash
export API_KEY=
MODEL_NAME_OR_PATH="/workspace/home/NLP_CORE/HUB_LLM/Meta-Llama-3.3-70B-Instruct"
MODEL_BACKEND="openai"
API_ENDPOINT="http://10.254.138.192:9002"
SERVED_MODEL_NAME="Meta-Llama-3-70B-Instruct"
OUTPUT_DIR="/raid/vinh/resources/results"
CONFIGS="gsm8k math olympiadbench omnimath"
DATASET_PATH="/raid/vinh/resources/datasets/ProcessBench"
VERIFIER_TYPE="aggregative"
# VERIFIER_TYPE="iterative"

# Run evaluation with the new verifier API
python /raid/vinh/reward_model/run_eval.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_backend $MODEL_BACKEND \
    --api_endpoint $API_ENDPOINT \
    --served_model_name $SERVED_MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --configs $CONFIGS \
    --dataset_path $DATASET_PATH \
    --verifier_type $VERIFIER_TYPE