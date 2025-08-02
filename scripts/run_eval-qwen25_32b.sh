#!/bin/bash
export API_KEY=

CONFIG_FILE="/your/path/to/configs/qwen25_32b.yaml"

echo "Running evaluation with $CONFIG_FILE"
python /your/path/to/run_eval.py $CONFIG_FILE
