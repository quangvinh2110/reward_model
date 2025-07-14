#!/bin/bash
export API_KEY=
MODEL_NAME_OR_PATH="/workspace/home/NLP_CORE/HUB_LLM/Meta-Llama-3.3-70B-Instruct"
CLIENT_TYPE="openai"
API_ENDPOINT="http://10.254.138.192:9002"
MODEL="Meta-Llama-3-70B-Instruct"
OUTPUT_DIR="/raid/vinh/resources/results"
CONFIGS="gsm8k math olympiadbench omnimath"
DATASET_PATH="/raid/vinh/resources/datasets/ProcessBench"
VERIFIER_TYPE="sequential"
# VERIFIER_TYPE="stepwise"
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS=8192

# Run evaluation with the new verifier API
python /raid/vinh/reward_model/run_eval.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --client_type $MODEL_BACKEND \
    --api_endpoint $API_ENDPOINT \
    --model $MODEL \
    --output_dir $OUTPUT_DIR \
    --configs $CONFIGS \
    --dataset_path $DATASET_PATH \
    --verifier_type $VERIFIER_TYPE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --max_tokens $MAX_TOKENS \
    # --enable_thinking