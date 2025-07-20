#!/bin/bash
export API_KEY=
API_ENDPOINT="http://10.254.138.189:8107"
MODEL="Qwen2.5-32B-Instruct"
OUTPUT_DIR="/raid/vinh/resources/results"
CONFIGS="gsm8k math olympiadbench omnimath"
DATASET_PATH="/raid/vinh/resources/datasets/ProcessBench"
# VERIFIER_TYPE="sequential"
# VERIFIER_TYPE="stepwise"
VERIFIER_TYPE="parc"
# VERIFIER_TYPE="logicflow"
TEMPERATURE=0
TOP_P=0.8
TOP_K=1
MAX_TOKENS=4096

# Run evaluation with the new verifier API
echo "Running evaluation with $VERIFIER_TYPE verifier and $MODEL model"
python /raid/vinh/reward_model/run_eval.py \
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
    --sample_size 250 \
    # --enable_thinking