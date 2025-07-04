#!/usr/bin/env python3
"""
Format critique training data to match the original train data format
"""

import json
import os
import argparse
from typing import List, Dict, Any

def format_critique_data(critique_file: str, output_file: str):
    """Format critique data to match train data format"""
    print(f"Loading critique data from {critique_file}")
    with open(critique_file, 'r', encoding='utf-8') as f:
        critique_data = json.load(f)
    
    print(f"Formatting {len(critique_data)} critique items...")
    
    formatted_data = []
    
    for i, item in enumerate(critique_data):
        # Extract fields from critique data
        prompt_text = item.get('prompt', '')
        target = item.get('target', '')
        question = item.get('question', '')
        solution = item.get('solution', '')
        gt_answer = item.get('gt_answer', '')
        subject = item.get('subject', '')
        level = item.get('level', '')
        extra_info = item.get('extra_info', {})
        
        # Create formatted item matching train data structure
        formatted_item = {
            "answer": target,  # Use target (right/wrong) as answer
            "gt_answer": target,  # Ground truth is the same as target for critique
            "subject": subject,
            "level": level,
            "question": question,
            "target": target,
            "data_source": "deepscaler_critique",
            "prompt": [
                {
                    "content": prompt_text,
                    "role": "user"
                }
            ],
            "ability": "critique",  # Changed from "math" to "critique"
            "reward_model": {
                "ground_truth": target,
                "style": "rule"
            },
            "extra_info": {
                "answer": target,
                "index": i,
                "level": level,
                "split": "train",
                "original_index": extra_info.get('original_index', i),
                "solution_index": extra_info.get('solution_index', 0),
                "model_source": extra_info.get('model_source', 'unknown'),
                "solution": solution,  # Keep the original solution for reference
                "question_gt_answer": gt_answer  # Keep the original question's ground truth
            }
        }
        
        formatted_data.append(formatted_item)
        
        if (i + 1) % 1000 == 0:
            print(f"Formatted {i + 1}/{len(critique_data)} items...")
    
    # Save formatted data
    print(f"Saving {len(formatted_data)} formatted items to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Formatting completed!")
    print(f"Original critique items: {len(critique_data)}")
    print(f"Formatted items: {len(formatted_data)}")

def analyze_formatted_data(formatted_file: str):
    """Analyze the formatted data structure"""
    print(f"Analyzing formatted data from {formatted_file}")
    with open(formatted_file, 'r', encoding='utf-8') as f:
        formatted_data = json.load(f)
    
    if not formatted_data:
        print("No data found!")
        return
    
    # Show sample item structure
    print("Sample formatted item structure:")
    sample = formatted_data[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    # Analyze distribution
    right_count = 0
    wrong_count = 0
    
    for item in formatted_data:
        target = item.get('target', '')
        if target == 'right':
            right_count += 1
        elif target == 'wrong':
            wrong_count += 1
    
    total = len(formatted_data)
    print(f"\nDistribution analysis:")
    print(f"Total items: {total}")
    print(f"Right critiques: {right_count} ({right_count/total*100:.2f}%)")
    print(f"Wrong critiques: {wrong_count} ({wrong_count/total*100:.2f}%)")
    
    # Check field consistency
    print(f"\nField consistency check:")
    required_fields = ['answer', 'gt_answer', 'subject', 'level', 'question', 'target', 
                      'data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    
    for field in required_fields:
        present_count = sum(1 for item in formatted_data if field in item)
        print(f"  {field}: {present_count}/{total} items ({present_count/total*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Format critique data to match train data format')
    parser.add_argument('--critique_file', type=str, default='../cft_data/deepscaler_critique.json',
                       help='Input critique data file')
    parser.add_argument('--output_file', type=str, default='../cft_data/deepscaler_critique_formatted.json',
                       help='Output formatted critique data file')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze formatted data after generation')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.critique_file):
        print(f"Error: Input file {args.critique_file} does not exist")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Format critique data
    format_critique_data(args.critique_file, args.output_file)
    analyze_formatted_data(args.output_file)
    # Analyze formatted data if requested
    # if args.analyze:
    #     analyze_formatted_data(args.output_file)

if __name__ == "__main__":
    main() 