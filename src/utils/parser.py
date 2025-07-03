import re
from typing import Optional


def parse_from_boxed(text: str) -> Optional[str]:
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
    return None
