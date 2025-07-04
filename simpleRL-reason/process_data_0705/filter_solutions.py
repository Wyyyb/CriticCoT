#!/usr/bin/env python3
"""
Filter solutions by removing questions where both models answered all correctly or all incorrectly
"""

import json
import os
import argparse
import re
from typing import List, Dict, Any, Tuple

def extract_answer_from_boxed(text: str) -> str:
    """Extract answer from \boxed{} format"""
    # Look for \boxed{...} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(boxed_pattern, text)
    if match:
        return match.group(1).strip()
    
    # If no \boxed{}, try to find the final answer
    # Look for patterns like "The answer is X" or "Therefore, X"
    answer_patterns = [
        r'[Tt]he answer is\s*([^\n.,;]+)',
        r'[Tt]herefore[,\s]+([^\n.,;]+)',
        r'[Ss]o[,\s]+([^\n.,;]+)',
        r'[Hh]ence[,\s]+([^\n.,;]+)',
        r'[Tt]hus[,\s]+([^\n.,;]+)',
        r'[Aa]nswer[:\s]+([^\n.,;]+)',
        r'[Rr]esult[:\s]+([^\n.,;]+)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    # If still no match, return the last line that looks like an answer
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('\\') and len(line) < 50:
            return line
    
    return ""

def is_answer_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth"""
    if not predicted or not ground_truth:
        return False
    
    # Clean and normalize answers
    pred_clean = re.sub(r'[^\w\d\-+./()]', '', predicted.lower())
    gt_clean = re.sub(r'[^\w\d\-+./()]', '', ground_truth.lower())
    
    # Direct comparison
    if pred_clean == gt_clean:
        return True
    
    # Try numerical comparison
    try:
        pred_num = float(pred_clean)
        gt_num = float(gt_clean)
        return abs(pred_num - gt_num) < 1e-6
    except ValueError:
        pass
    
    return False

def check_model_consistency(solutions: List[str], ground_truth: str) -> Tuple[bool, bool]:
    """Check if all solutions are correct or all are incorrect"""
    if not solutions:
        return False, False
    
    correct_count = 0
    for solution in solutions:
        extracted_answer = extract_answer_from_boxed(solution)
        if is_answer_correct(extracted_answer, ground_truth):
            correct_count += 1
    
    all_correct = correct_count == len(solutions)
    all_incorrect = correct_count == 0
    
    return all_correct, all_incorrect

def filter_solutions(qwen25_file: str, qwen3_file: str, original_file: str, output_file: str):
    """Filter solutions and save filtered data"""
    print(f"Loading qwen-2.5-math-7b solutions from {qwen25_file}")
    with open(qwen25_file, 'r', encoding='utf-8') as f:
        qwen25_data = json.load(f)
    
    print(f"Loading qwen3-4b-base solutions from {qwen3_file}")
    with open(qwen3_file, 'r', encoding='utf-8') as f:
        qwen3_data = json.load(f)
    
    print(f"Loading original training data from {original_file}")
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Create index mapping for qwen25 data
    qwen25_index_map = {}
    for item in qwen25_data:
        index = item.get('extra_info', {}).get('index', item.get('chunk_index', 0))
        qwen25_index_map[index] = item
    
    # Create index mapping for qwen3 data
    qwen3_index_map = {}
    for item in qwen3_data:
        index = item.get('extra_info', {}).get('index', item.get('chunk_index', 0))
        qwen3_index_map[index] = item
    
    print(f"Processing {len(original_data)} questions...")
    
    filtered_data = []
    kept_count = 0
    removed_count = 0
    
    for i, original_item in enumerate(original_data):
        # Get solutions from both models
        qwen25_item = qwen25_index_map.get(i)
        qwen3_item = qwen3_index_map.get(i)
        
        if not qwen25_item or not qwen3_item:
            print(f"Warning: Missing solutions for question {i}")
            continue
        
        qwen25_solutions = qwen25_item.get('model_solutions', [])
        qwen3_solutions = qwen3_item.get('model_solutions', [])
        ground_truth = original_item.get('gt_answer', '')
        
        # Check consistency for both models
        qwen25_all_correct, qwen25_all_incorrect = check_model_consistency(qwen25_solutions, ground_truth)
        qwen3_all_correct, qwen3_all_incorrect = check_model_consistency(qwen3_solutions, ground_truth)
        
        # Keep if at least one model has mixed results
        should_keep = not (qwen25_all_correct and qwen3_all_correct) and not (qwen25_all_incorrect and qwen3_all_incorrect)
        
        if should_keep:
            # Add solution information to the original item
            filtered_item = original_item.copy()
            filtered_item['qwen25_solutions'] = qwen25_solutions
            filtered_item['qwen3_solutions'] = qwen3_solutions
            filtered_item['qwen25_all_correct'] = qwen25_all_correct
            filtered_item['qwen25_all_incorrect'] = qwen25_all_incorrect
            filtered_item['qwen3_all_correct'] = qwen3_all_correct
            filtered_item['qwen3_all_incorrect'] = qwen3_all_incorrect
            
            filtered_data.append(filtered_item)
            kept_count += 1
        else:
            removed_count += 1
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(original_data)} questions...")
    
    # Save filtered data
    print(f"Saving {len(filtered_data)} filtered questions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"Filtering completed!")
    print(f"Original questions: {len(original_data)}")
    print(f"Kept questions: {kept_count}")
    print(f"Removed questions: {removed_count}")
    print(f"Removal rate: {removed_count/len(original_data)*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Filter solutions by model consistency')
    parser.add_argument('--qwen25_file', type=str, default='../cft_data/deepscaler_qwen25_solutions.json',
                       help='qwen-2.5-math-7b solutions file')
    parser.add_argument('--qwen3_file', type=str, default='../cft_data/deepscaler_qwen3_solutions.json',
                       help='qwen3-4b-base solutions file')
    parser.add_argument('--original_file', type=str, default='../cft_data/deepscaler_train.json',
                       help='Original training data file')
    parser.add_argument('--output_file', type=str, default='../cft_data/deepscaler_train_filter.json',
                       help='Output filtered training data file')
    
    args = parser.parse_args()
    
    # Check if input files exist
    for file_path in [args.qwen25_file, args.qwen3_file, args.original_file]:
        if not os.path.exists(file_path):
            print(f"Error: Input file {file_path} does not exist")
            return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    filter_solutions(args.qwen25_file, args.qwen3_file, args.original_file, args.output_file)

if __name__ == "__main__":
    main() 