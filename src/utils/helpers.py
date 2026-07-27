"""
Common helper functions.
"""

import os
from typing import Any, Dict
import json


def ensure_dir(path: str) -> str:
    """Ensures directory exists."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Saves dictionary to JSON file."""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str) -> Dict[str, Any]:
    """Loads JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
