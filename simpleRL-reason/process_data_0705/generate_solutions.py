#!/usr/bin/env python3
"""
Generate candidate solutions using vLLM with chunked processing
"""

import json
import os
import sys
import argparse
import time
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

def load_deepscaler_data(file_path: str) -> List[Dict]:
    """Load deepscaler training data"""
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data

def chunk_data(data: List[Dict], chunk_size: int = 1000) -> List[List[Dict]]:
    """Split data into chunks"""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunks.append(chunk)
    print(f"Split data into {len(chunks)} chunks of size {chunk_size}")
    return chunks

def generate_solutions_with_vllm(model: LLM, prompts: List[str], num_samples: int = 8) -> List[List[str]]:
    """Generate solutions using vLLM model"""
    print(f"Generating {num_samples} solutions for {len(prompts)} prompts...")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        n=num_samples
    )
    
    results = []
    batch_size = 1000  # Process in batches to avoid memory issues
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
        
        outputs = model.generate(batch_prompts, sampling_params)
        
        for output in outputs:
            # Extract generated texts from all samples
            generated_texts = [sample.text for sample in output.outputs]
            results.append(generated_texts)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.01)
    
    return results

def process_chunk(chunk_data: List[Dict], model_name: str, output_dir: str, chunk_id: int, num_samples: int = 8):
    """Process a single chunk of data"""
    print(f"Processing chunk {chunk_id} with model {model_name}")
    
    # Extract prompts
    prompts = [item['prompt'][0]['content'] for item in chunk_data]
    
    # Initialize model
    try:
        model = LLM(model=model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}")
        return
    
    # Generate solutions
    solutions = generate_solutions_with_vllm(model, prompts, num_samples)
    
    # Create output data
    output_data = []
    for i, (item, solution_list) in enumerate(zip(chunk_data, solutions)):
        output_item = item.copy()
        output_item['model_solutions'] = solution_list
        output_item['model_name'] = model_name
        output_item['chunk_id'] = chunk_id
        output_item['chunk_index'] = i
        output_data.append(output_item)
    
    # Save results
    output_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_chunk_{chunk_id:03d}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved chunk {chunk_id} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate candidate solutions with vLLM')
    parser.add_argument('--input_file', type=str, default='../cft_data/deepscaler_train.json',
                       help='Input deepscaler training data file')
    parser.add_argument('--output_dir', type=str, default='../cft_data/solutions',
                       help='Output directory for solution files')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to use for generation')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Number of questions per chunk')
    parser.add_argument('--chunk_id', type=int, default=None,
                       help='Specific chunk ID to process (if None, process all chunks)')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of solutions to generate per question')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_deepscaler_data(args.input_file)
    
    # Split into chunks
    chunks = chunk_data(data, args.chunk_size)
    
    if args.chunk_id is not None:
        # Process specific chunk
        if args.chunk_id >= len(chunks):
            print(f"Error: Chunk ID {args.chunk_id} is out of range (0-{len(chunks)-1})")
            return
        
        process_chunk(chunks[args.chunk_id], args.model, args.output_dir, args.chunk_id, args.num_samples)
    else:
        # Process all chunks
        for chunk_id, chunk in enumerate(chunks):
            process_chunk(chunk, args.model, args.output_dir, chunk_id, args.num_samples)
    
    print("Processing completed!")

if __name__ == "__main__":
    main() 