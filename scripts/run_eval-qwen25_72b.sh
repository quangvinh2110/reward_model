#!/bin/bash
export API_KEY=

CONFIG_FILE="/home/admin/NLP/reward_model/reward_model/configs/qwen25_72b.yaml"

echo "Running evaluation with $CONFIG_FILE"
python /home/admin/NLP/reward_model/reward_model/run_eval.py $CONFIG_FILE
