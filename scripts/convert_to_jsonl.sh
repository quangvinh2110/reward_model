#! /bin/bash

python src/data_utils/convert_to_jsonl.py \
    --input_path "/mnt/e/AAAI-26/resources/data/Bespoke-Stratos-17k/train-00000-of-00001.parquet" \
    --file_type "parquet" \
    --output_path "/mnt/e/AAAI-26/resources/data/Bespoke-Stratos-17k/train-00000-of-00001.jsonl"
