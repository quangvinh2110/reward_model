#!/bin/bash
export API_KEY=

CONFIG_FILE="/raid/vinh/reward_model/configs/qwen25_32b.yaml"

echo "Running evaluation with $CONFIG_FILE"
python /raid/vinh/reward_model/run_eval.py $CONFIG_FILE
