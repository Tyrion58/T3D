#!/usr/bin/env python3
"""
Script to rename the key "full_output" to "answer" in a JSON file.
Usage: python rename_key.py --model_name <name> --dataset <dataset> [--input <path>] [--output <path>]
"""

import argparse
import json
from pathlib import Path


def rename_key_in_json(input_file, output_file):
    """
    Read JSON file, rename "full_output" to "answer" in each object, and write to output file.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Unwrap {"_config": ..., "data": [...]} format if present
    items = data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        items = data["data"]
    elif not isinstance(data, list):
        print("Warning: Unexpected JSON structure (expected list or {\"data\": [...]})")
        return

    count = 0
    for item in items:
        if isinstance(item, dict) and "full_output" in item:
            item["answer"] = item.pop("full_output")
            count += 1
    if count == 0:
        print("Warning: No 'full_output' keys found — nothing renamed")
    else:
        print(f"Renamed 'full_output' to 'answer' in {count} items")

    # Write the modified data to output file
    print("Writing output file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rename key "full_output" to "answer" in JSON file. '
                    'Paths are built from --dataset and --model_name unless --input/--output are set.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['SDAR-1.7B-Chat', 'SDAR-4B-Chat'],
        help='Model name: SDAR-1.7B-Chat or SDAR-4B-Chat.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name or subfolder, used in default input/output paths.'
    )
    args = parser.parse_args()
    input_file = Path(__file__).parent / f"{args.model_name}-{args.dataset}.json"
    output_file = Path(__file__).parent / f"{args.model_name}-{args.dataset}_renamed.json"

    return input_file, output_file


if __name__ == "__main__":
    input_file, output_file = parse_args()
    rename_key_in_json(input_file, output_file)

