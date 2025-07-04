#!/usr/bin/env python3
"""
Generate critique training data from filtered solutions
Convert candidate solutions to right/wrong judgments
"""

import json
import os
import argparse
import re
from typing import List, Dict, Any
from filter_solutions import extract_answer_from_boxed, is_answer_correct

def create_critique_prompt(question: str, solution: str) -> str:
    """Create critique prompt for a solution"""
    prompt = f"""You are a mathematics expert. A student is trying to solve a question. Please explain briefly whether his answer is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]'

Question: {question}

Student's solution:
{solution}

Critique:"""
    return prompt

def create_critique_target(solution: str, ground_truth: str) -> str:
    """Create critique target (right/wrong) for a solution"""
    extracted_answer = extract_answer_from_boxed(solution)
    is_correct = is_answer_correct(extracted_answer, ground_truth)
    return "right" if is_correct else "wrong"

def generate_critique_data(filtered_file: str, output_file: str, max_samples_per_question: int = 4):
    """Generate critique training data from filtered solutions"""
    print(f"Loading filtered data from {filtered_file}")
    with open(filtered_file, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)
    
    print(f"Generating critique data for {len(filtered_data)} questions...")
    
    critique_data = []
    total_critiques = 0
    
    for i, item in enumerate(filtered_data):
        question = item.get('question', '')
        ground_truth = item.get('gt_answer', '')
        
        # Get solutions from both models
        qwen25_solutions = item.get('qwen25_solutions', [])
        qwen3_solutions = item.get('qwen3_solutions', [])
        
        # Combine solutions and limit samples per question
        all_solutions = qwen25_solutions + qwen3_solutions
        
        # Randomly sample solutions if we have too many
        if len(all_solutions) > max_samples_per_question:
            import random
            random.seed(42)  # For reproducibility
            all_solutions = random.sample(all_solutions, max_samples_per_question)
        
        # Generate critique data for each solution
        for j, solution in enumerate(all_solutions):
            # Create critique prompt
            prompt = create_critique_prompt(question, solution)
            
            # Create critique target
            target = create_critique_target(solution, ground_truth)
            
            # Create critique item
            critique_item = {
                'prompt': prompt,
                'target': target,
                'question': question,
                'solution': solution,
                'gt_answer': ground_truth,
                'subject': item.get('subject', ''),
                'level': item.get('level', ''),
                'data_source': 'deepscaler_critique',
                'extra_info': {
                    'original_index': i,
                    'solution_index': j,
                    'model_source': 'qwen25' if j < len(qwen25_solutions) else 'qwen3'
                }
            }
            
            critique_data.append(critique_item)
            total_critiques += 1
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(filtered_data)} questions, generated {total_critiques} critiques...")
    
    # Save critique data
    print(f"Saving {len(critique_data)} critique items to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(critique_data, f, ensure_ascii=False, indent=2)
    
    print(f"Critique data generation completed!")
    print(f"Original questions: {len(filtered_data)}")
    print(f"Generated critiques: {len(critique_data)}")
    print(f"Average critiques per question: {len(critique_data)/len(filtered_data):.2f}")

def analyze_critique_distribution(critique_file: str):
    """Analyze the distribution of right/wrong critiques"""
    print(f"Analyzing critique distribution from {critique_file}")
    with open(critique_file, 'r', encoding='utf-8') as f:
        critique_data = json.load(f)
    
    right_count = 0
    wrong_count = 0
    
    for item in critique_data:
        target = item.get('target', '')
        if target == 'right':
            right_count += 1
        elif target == 'wrong':
            wrong_count += 1
    
    total = len(critique_data)
    print(f"Total critiques: {total}")
    print(f"Right critiques: {right_count} ({right_count/total*100:.2f}%)")
    print(f"Wrong critiques: {wrong_count} ({wrong_count/total*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Generate critique training data')
    parser.add_argument('--filtered_file', type=str, default='../cft_data/deepscaler_train_filter.json',
                       help='Filtered training data file')
    parser.add_argument('--output_file', type=str, default='../cft_data/deepscaler_critique.json',
                       help='Output critique training data file')
    parser.add_argument('--max_samples_per_question', type=int, default=4,
                       help='Maximum number of solutions to sample per question')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze critique distribution after generation')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.filtered_file):
        print(f"Error: Input file {args.filtered_file} does not exist")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Generate critique data
    generate_critique_data(args.filtered_file, args.output_file, args.max_samples_per_question)
    
    # Analyze distribution if requested
    if args.analyze:
        analyze_critique_distribution(args.output_file)

if __name__ == "__main__":
    main() 