import re
import json


def parse_from_boxed(text: str) -> str:
    """Extract the last boxed answer from text.

    Args:
        text (str): Text containing a boxed answer

    Returns:
        Optional[str]: The extracted answer if found, None otherwise
    """
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(boxed_pattern, text)
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
    json_pattern = r"```json\n(.*?)\n```"
    matches = re.findall(json_pattern, text)
    try:
        return json.loads(matches[-1].strip())
    except:
        return {}
