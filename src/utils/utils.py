from typing import List, Dict
import json


def write_jsonl(data: List[Dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(data: Dict, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def batch_iter(data: List, batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def read_txt(file_path: str) -> str:
    """Read a text file and return its contents as a string.

    Args:
        file_path (str): Path to the text file

    Returns:
        str: Contents of the text file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
