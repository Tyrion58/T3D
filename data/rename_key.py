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

    # Process each item in the array
    if isinstance(data, list):
        count = 0
        for item in data:
            if isinstance(item, dict) and "full_output" in item:
                item["answer"] = item.pop("full_output")
                count += 1
        print(f"Renamed 'full_output' to 'answer' in {count} items")
    elif isinstance(data, dict):
        # Handle case where JSON is a single object
        if "full_output" in data:
            data["answer"] = data.pop("full_output")
            print("Renamed 'full_output' to 'answer' in the object")
    else:
        print("Warning: Unexpected JSON structure")
        return

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

