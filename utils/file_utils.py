import json
import os
import shutil
from typing import Dict, List, Union


def write_file(file_path: str, content: str | Dict):
    """Writes string or JSON content to a file."""
    if isinstance(content, dict):
        content = json.dumps(content)
    with open(file_path, 'w') as file:
        file.write(content + '\n')


def create_dirs(file_dir: str):
    """Creates directories if they do not exist."""
    try:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    except Exception:
        pass


def delete_dirs(file_dir: str):
    """Deletes a directory and its contents."""
    try:
        shutil.rmtree(file_dir)
    except Exception:
        pass


def exists_file(file_path: str) -> bool:
    """Checks if a file exists."""
    return os.path.exists(file_path)


def read_file(file_path: str) -> str:
    """Reads and returns the content of a file."""
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def create_or_clear_file(file_path: str):
    """Creates or clears the contents of a file."""
    with open(file_path, 'w'):
        pass


def load_config(config_path):
    """Loads and returns JSON configuration from a file."""
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def write_json(file_path: str, content: Union[List, Dict]):
    """Writes JSON content to a file with indentation (4 spaces)."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(content, indent=4))


def read_json(file_path: str) -> Union[List, Dict]:
    """Reads and returns JSON content from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    return content
