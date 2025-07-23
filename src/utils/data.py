import re
import json
from typing import Any


JSON_PATTERN = re.compile(r"```json\n([\s\S]*?)\n```")
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def parse_from_boxed(text: str) -> str:
    """Extract the last boxed answer from text.

    Args:
        text (str): Text containing a boxed answer

    Returns:
        Optional[str]: The extracted answer if found, None otherwise
    """
    matches = BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return ""


def parse_from_json(text: str) -> dict:
    """Extract the last json answer from text.

    Args:
        text (str): Text containing a json answer

    Returns:
        Optional[str]: The extracted answer if found, None otherwise
    """
    matches = JSON_PATTERN.findall(text)
    if matches:
        for match in matches[::-1]:
            try:
                return json.loads(match.strip())
            except:
                continue
    return {}


def to_int(text: Any) -> int:
    try:
        return int(text)
    except:
        return -100000
