import json
from typing import List, Dict, Union
import os


def _resolve_safe_jsonl_path(file_path: str) -> Union[str, None]:
    """Resolve and validate a JSONL path under a safe base directory."""
    base_dir = os.path.realpath(os.environ.get("RESULTS_BASE_DIR", os.getcwd()))
    candidate_path = os.path.realpath(file_path)

    try:
        if os.path.commonpath([base_dir, candidate_path]) != base_dir:
            return None
    except ValueError:
        return None

    if not candidate_path.endswith(".jsonl"):
        return None

    return candidate_path


def read_jsonl(file_path: str) -> List[Dict]:
    """Reads a JSONL file and returns a list of dictionaries."""
    safe_path = _resolve_safe_jsonl_path(file_path)
    if not safe_path or not os.path.exists(safe_path):
        return []
    data = []
    with open(safe_path, 'r') as file:
        line = file.readline()
        while line != '':
            line = json.loads(line)
            data.append(line)
            line = file.readline()
    return data


def write_jsonl(file_path: str, data: Union[List[Dict], Dict]):
    """Writes a list of dictionaries or a single dictionary to a JSONL file."""
    if type(data) == dict:
        data = [data]
    with open(file_path, 'w') as file:
        for line in data:
            line = json.dumps(line)
            file.write(line + '\n')


def append_jsonl(file_path: str, data: Union[List[Dict], Dict]):
    """Appends a list of dictionaries or a single dictionary to a JSONL file."""
    if type(data) == dict:
        data = [data]
    with open(file_path, 'a') as file:
        for line in data:
            line = json.dumps(line)
            file.write(line + '\n')


def dir_jsonl_files(dir_path: str) -> List[str]:
    """Returns a list of JSONL files from a dictionary that are formatted as `result_{i}.jsonl`."""
    i = 0
    res = []
    while True:
        file_path = os.path.join(dir_path, f'result_{i}.jsonl')
        if not os.path.exists(file_path):
            break
        res.append(file_path)
        i += 1
    return res
