import pandas as pd
import argparse

from utils import write_jsonl


def convert_to_jsonl(file_path: str, file_type: str, output_path: str) -> None:
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'json':
        df = pd.read_json(file_path)
    elif file_type == 'parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    write_jsonl(df.to_dict(orient='records'), output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--file_type', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_to_jsonl(args.input_path, args.file_type, args.output_path)

