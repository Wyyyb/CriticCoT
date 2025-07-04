#!/usr/bin/env python3
"""
Convert parquet data to JSON format for training
"""

import pandas as pd
import json
import argparse
import os
from typing import List, Union

def convert_parquet_to_json(parquet_file: str, output_file: str, jsonl_format: bool = False):
    """
    Convert parquet file to JSON format
    
    Args:
        parquet_file: Input parquet file path
        output_file: Output JSON file path
        jsonl_format: If True, output as JSONL (one JSON object per line)
    """
    print(f"Reading parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    print(f"Converting {len(df)} records...")
    
    if jsonl_format:
        # JSONL format: one JSON object per line
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Convert numpy arrays to lists for JSON serialization
                row_dict = row.to_dict()
                for key, value in row_dict.items():
                    if hasattr(value, 'tolist'):  # numpy array
                        row_dict[key] = value.tolist()
                f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
    else:
        # Single JSON file with array of objects
        data = []
        for _, row in df.iterrows():
            # Convert numpy arrays to lists for JSON serialization
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if hasattr(value, 'tolist'):  # numpy array
                    row_dict[key] = value.tolist()
            data.append(row_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion completed! Output saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert parquet data to JSON format')
    parser.add_argument('--input', '-i', required=True, help='Input parquet file path')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path')
    parser.add_argument('--jsonl', action='store_true', help='Output as JSONL format (one JSON object per line)')
    parser.add_argument('--batch', action='store_true', help='Process multiple files (input should be a directory)')
    
    args = parser.parse_args()
    
    if args.batch:
        # Process all parquet files in a directory
        if os.path.isdir(args.input):
            parquet_files = [f for f in os.listdir(args.input) if f.endswith('.parquet')]
            for parquet_file in parquet_files:
                input_path = os.path.join(args.input, parquet_file)
                output_path = os.path.join(args.output, parquet_file.replace('.parquet', '.json'))
                os.makedirs(args.output, exist_ok=True)
                convert_parquet_to_json(input_path, output_path, args.jsonl)
        else:
            print(f"Error: {args.input} is not a directory")
            return
    else:
        # Process single file
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist")
            return
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        convert_parquet_to_json(args.input, args.output, args.jsonl)

if __name__ == '__main__':
    main() 