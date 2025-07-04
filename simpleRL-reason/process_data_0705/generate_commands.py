#!/usr/bin/env python3
"""
Generate shell commands for running solution generation on different chunks
"""

import json
import os
import argparse

def load_deepscaler_data(file_path: str) -> list:
    """Load deepscaler training data to get total count"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_commands(input_file: str, chunk_size: int = 1000, num_samples: int = 8):
    """Generate shell commands for all chunks"""
    
    # Load data to get total count
    data = load_deepscaler_data(input_file)
    total_questions = len(data)
    num_chunks = (total_questions + chunk_size - 1) // chunk_size
    
    print(f"Total questions: {total_questions}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {num_chunks}")
    print()
    
    # Generate commands for qwen-2.5-math-7b
    print("# Commands for qwen-2.5-math-7b:")
    print()
    for chunk_id in range(num_chunks):
        print(f"# GPU {chunk_id % 8} - Chunk {chunk_id}")
        print(f"CUDA_VISIBLE_DEVICES={chunk_id % 8} python generate_solutions.py \\")
        print(f"  --input_file {input_file} \\")
        print(f"  --model qwen-2.5-math-7b \\")
        print(f"  --chunk_id {chunk_id} \\")
        print(f"  --chunk_size {chunk_size} \\")
        print(f"  --num_samples {num_samples} \\")
        print(f"  --output_dir ../cft_data/solutions_qwen25 &")
        print()
    
    print("# Commands for qwen3-4b-base:")
    print()
    for chunk_id in range(num_chunks):
        print(f"# GPU {chunk_id % 8} - Chunk {chunk_id}")
        print(f"CUDA_VISIBLE_DEVICES={chunk_id % 8} python generate_solutions.py \\")
        print(f"  --input_file {input_file} \\")
        print(f"  --model qwen3-4b-base \\")
        print(f"  --chunk_id {chunk_id} \\")
        print(f"  --chunk_size {chunk_size} \\")
        print(f"  --num_samples {num_samples} \\")
        print(f"  --output_dir ../cft_data/solutions_qwen3 &")
        print()
    
    # Generate shell script
    print("# Shell script for qwen-2.5-math-7b:")
    print("#!/bin/bash")
    print("# run_qwen25.sh")
    print()
    for chunk_id in range(num_chunks):
        print(f"CUDA_VISIBLE_DEVICES={chunk_id % 8} python generate_solutions.py \\")
        print(f"  --input_file {input_file} \\")
        print(f"  --model qwen-2.5-math-7b \\")
        print(f"  --chunk_id {chunk_id} \\")
        print(f"  --chunk_size {chunk_size} \\")
        print(f"  --num_samples {num_samples} \\")
        print(f"  --output_dir ../cft_data/solutions_qwen25 &")
        print()
    print("wait")
    print()
    
    print("# Shell script for qwen3-4b-base:")
    print("#!/bin/bash")
    print("# run_qwen3.sh")
    print()
    for chunk_id in range(num_chunks):
        print(f"CUDA_VISIBLE_DEVICES={chunk_id % 8} python generate_solutions.py \\")
        print(f"  --input_file {input_file} \\")
        print(f"  --model qwen3-4b-base \\")
        print(f"  --chunk_id {chunk_id} \\")
        print(f"  --chunk_size {chunk_size} \\")
        print(f"  --num_samples {num_samples} \\")
        print(f"  --output_dir ../cft_data/solutions_qwen3 &")
        print()
    print("wait")

def main():
    parser = argparse.ArgumentParser(description='Generate shell commands for chunked processing')
    parser.add_argument('--input_file', type=str, default='../cft_data/deepscaler_train.json',
                       help='Input deepscaler training data file')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Number of questions per chunk')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of solutions to generate per question')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    generate_commands(args.input_file, args.chunk_size, args.num_samples)

if __name__ == "__main__":
    main() 