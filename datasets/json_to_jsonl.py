#!/usr/bin/env python3
import json
import sys

def json_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Input JSON must be an array of objects.", file=sys.stderr)
        sys.exit(1)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: json_to_jsonl.py input.json output.jsonl", file=sys.stderr)
        sys.exit(1)

    json_to_jsonl(sys.argv[1], sys.argv[2])

