#!/bin/bash
export API_KEY=

CONFIG_FILE="/raid/vinh/reward_model/configs/qwen25_14b.yaml"

# Run evaluation with the new verifier API
echo "Running evaluation with $CONFIG_FILE"
python /raid/vinh/reward_model/run_eval.py $CONFIG_FILE