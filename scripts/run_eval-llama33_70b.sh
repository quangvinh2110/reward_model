#!/bin/bash
export API_KEY=

CONFIG_FILE="/raid/vinh/reward_model/configs/llama33_70b.yaml"

# Run evaluation with config file
python /raid/vinh/reward_model/run_eval.py $CONFIG_FILE