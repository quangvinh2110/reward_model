#!/bin/bash

# Default values
MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
OUTPUT_DIR="/your/path/to/folder/to/save/model"
CACHE_DIR="/your/path/to/folder/to/cache/model"

# Check if model name is provided
if [ -z "$MODEL_NAME" ]; then
    echo "Error: Model name is required"
    show_help
fi

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Run the Python script
python src/utils/download_hf_models.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" 