#!/usr/bin/env python3
"""
Merge all chunk solution files into single files
"""

import json
import os
import argparse
import glob
from typing import List, Dict

def merge_solution_chunks(input_dir: str, output_file: str, model_name: str):
    """Merge all chunk files for a specific model"""
    print(f"Merging {model_name} solution chunks from {input_dir}")
    
    # Find all chunk files for this model
    pattern = os.path.join(input_dir, f"_data_minimax-dialogue_feishan_models_Qwen2.5-Math-7B_chunk_*.json")
    chunk_files = sorted(glob.glob(pattern))
    
    if not chunk_files:
        print(f"No chunk files found for pattern: {pattern}")
        return
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Merge all chunks
    merged_data = []
    for chunk_file in chunk_files:
        print(f"Loading {chunk_file}")
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        merged_data.extend(chunk_data)
    
    # Sort by original index to maintain order
    merged_data.sort(key=lambda x: x.get('extra_info', {}).get('index', 0))
    
    # Save merged data
    print(f"Saving {len(merged_data)} records to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully merged {len(merged_data)} records")

def main():
    parser = argparse.ArgumentParser(description='Merge solution chunk files')
    parser.add_argument('--qwen25_dir', type=str, default='../cft_data/solutions_qwen25',
                       help='Directory containing qwen-2.5-math-7b chunk files')
    parser.add_argument('--qwen3_dir', type=str, default='../cft_data/solutions_qwen3',
                       help='Directory containing qwen3-4b-base chunk files')
    parser.add_argument('--output_dir', type=str, default='../cft_data',
                       help='Output directory for merged files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Merge qwen-2.5-math-7b chunks
    if os.path.exists(args.qwen25_dir):
        qwen25_output = os.path.join(args.output_dir, 'deepscaler_qwen25_solutions.json')
        merge_solution_chunks(args.qwen25_dir, qwen25_output, 'qwen-2.5-math-7b')
    else:
        print(f"Warning: Directory {args.qwen25_dir} does not exist")
    
    # Merge qwen3-4b-base chunks
    if os.path.exists(args.qwen3_dir):
        qwen3_output = os.path.join(args.output_dir, 'deepscaler_qwen3_solutions.json')
        merge_solution_chunks(args.qwen3_dir, qwen3_output, 'qwen3-4b-base')
    else:
        print(f"Warning: Directory {args.qwen3_dir} does not exist")
    
    print("Merge completed!")

if __name__ == "__main__":
    main() 