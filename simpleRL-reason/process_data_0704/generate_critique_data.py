#!/usr/bin/env python3
"""
Generate critique training data using vLLM with qwen-2.5-math-7b and qwen3-4b-base models
"""

import json
import os
import re
import time
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
import torch

def load_deepscaler_data(file_path: str) -> List[Dict]:
    """Load deepscaler training data"""
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data

def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format"""
    # Look for \boxed{...} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(boxed_pattern, text)
    if match:
        return match.group(1).strip()
    
    # Look for boxed{...} pattern (without backslash)
    boxed_pattern2 = r'boxed\{([^}]*)\}'
    match = re.search(boxed_pattern2, text)
    if match:
        return match.group(1).strip()
    
    # Look for \boxed{...} at the end of text
    boxed_pattern3 = r'\\boxed\{([^}]*)\}\s*$'
    match = re.search(boxed_pattern3, text)
    if match:
        return match.group(1).strip()
    
    return ""

def is_answer_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth"""
    # Clean up the answers
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()
    
    # Direct comparison
    if predicted == ground_truth:
        return True
    
    # Remove common LaTeX formatting
    predicted_clean = re.sub(r'\\text\{([^}]*)\}', r'\1', predicted)
    ground_truth_clean = re.sub(r'\\text\{([^}]*)\}', r'\1', ground_truth)
    
    if predicted_clean == ground_truth_clean:
        return True
    
    # Try numerical comparison for simple cases
    try:
        # Remove LaTeX commands and evaluate
        pred_num = eval(predicted_clean.replace('\\frac{', '(').replace('}{', '/').replace('}', ')'))
        gt_num = eval(ground_truth_clean.replace('\\frac{', '(').replace('}{', '/').replace('}', ')'))
        if abs(pred_num - gt_num) < 1e-6:
            return True
    except:
        pass
    
    return False

def generate_answers_with_vllm(model: LLM, prompts: List[str], num_samples: int = 8) -> List[List[str]]:
    """Generate answers using vLLM model"""
    print(f"Generating {num_samples} answers for {len(prompts)} prompts...")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        n=num_samples
    )
    
    results = []
    batch_size = 10  # Process in batches to avoid memory issues
    
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

def filter_data_by_accuracy(data: List[Dict], model1_results: List[List[str]], 
                           model2_results: List[List[str]], ground_truths: List[str]) -> Tuple[List[Dict], List[int]]:
    """Filter data based on model accuracy patterns"""
    print("Filtering data based on accuracy patterns...")
    
    filtered_data = []
    filtered_indices = []
    
    for i, (item, model1_answers, model2_answers, gt) in enumerate(zip(data, model1_results, model2_results, ground_truths)):
        # Check model1 accuracy
        model1_correct = sum(1 for ans in model1_answers if is_answer_correct(extract_boxed_answer(ans), gt))
        model1_accuracy = model1_correct / len(model1_answers)
        
        # Check model2 accuracy
        model2_correct = sum(1 for ans in model2_answers if is_answer_correct(extract_boxed_answer(ans), gt))
        model2_accuracy = model2_correct / len(model2_answers)
        
        # Keep data where at least one model has mixed results (not all correct or all wrong)
        if (0 < model1_accuracy < 1) or (0 < model2_accuracy < 1):
            filtered_data.append(item)
            filtered_indices.append(i)
    
    print(f"Filtered data: {len(filtered_data)}/{len(data)} records kept")
    return filtered_data, filtered_indices

def generate_critique_conclusion(model1_answers: List[str], model2_answers: List[str], 
                                ground_truth: str) -> str:
    """Generate critique conclusion based on model answers"""
    
    # Extract boxed answers
    model1_extracted = [extract_boxed_answer(ans) for ans in model1_answers]
    model2_extracted = [extract_boxed_answer(ans) for ans in model2_answers]
    
    # Count correct answers for each model
    model1_correct = sum(1 for ans in model1_extracted if is_answer_correct(ans, ground_truth))
    model2_correct = sum(1 for ans in model2_extracted if is_answer_correct(ans, ground_truth))
    
    # Determine conclusion
    if model1_correct > model2_correct:
        return "right"
    elif model2_correct > model1_correct:
        return "wrong"
    else:
        # If equal, check if both are mostly correct or mostly wrong
        if model1_correct > len(model1_answers) / 2:
            return "right"
        else:
            return "wrong"

def create_critique_training_data(filtered_data: List[Dict], model1_results: List[List[str]], 
                                 model2_results: List[List[str]], model1_name: str, model2_name: str) -> List[Dict]:
    """Create critique training data format"""
    print("Creating critique training data...")
    
    critique_data = []
    
    for i, (item, model1_answers, model2_answers) in enumerate(zip(filtered_data, model1_results, model2_results)):
        ground_truth = item['gt_answer']
        
        # Generate critique conclusion
        conclusion = generate_critique_conclusion(model1_answers, model2_answers, ground_truth)
        
        # Create critique training record
        critique_record = {
            "prompt": item['prompt'],
            "model1_name": model1_name,
            "model2_name": model2_name,
            "model1_solutions": model1_answers,
            "model2_solutions": model2_answers,
            "ground_truth": ground_truth,
            "conclusion": conclusion,
            "original_index": item['extra_info']['index'],
            "subject": item['subject'],
            "level": item['level'],
            "ability": item['ability']
        }
        
        critique_data.append(critique_record)
    
    return critique_data

def main():
    """Main function"""
    
    # File paths
    input_file = "../cft_data/deepscaler_train.json"
    model1_filtered_file = "../cft_data/deepscaler_filtered_model1.json"
    model2_filtered_file = "../cft_data/deepscaler_filtered_model2.json"
    critique_file = "../cft_data/deepscaler_critique_data.json"
    
    # Model names
    model1_name = "qwen-2.5-math-7b"
    model2_name = "qwen3-4b-base"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    # Load data
    data = load_deepscaler_data(input_file)
    
    # Extract prompts and ground truths
    prompts = [item['prompt'][0]['content'] for item in data]
    ground_truths = [item['gt_answer'] for item in data]
    
    print(f"Processing {len(data)} questions...")
    
    # Initialize models
    print("Initializing models...")
    try:
        model1 = LLM(model=model1_name, trust_remote_code=True)
        model2 = LLM(model=model2_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error initializing models: {e}")
        return
    
    # Generate answers with both models
    print(f"\nGenerating answers with {model1_name}...")
    model1_results = generate_answers_with_vllm(model1, prompts, num_samples=8)
    
    print(f"\nGenerating answers with {model2_name}...")
    model2_results = generate_answers_with_vllm(model2, prompts, num_samples=8)
    
    # Filter data
    filtered_data, filtered_indices = filter_data_by_accuracy(data, model1_results, model2_results, ground_truths)
    
    # Filter model results to match filtered data
    filtered_model1_results = [model1_results[i] for i in filtered_indices]
    filtered_model2_results = [model2_results[i] for i in filtered_indices]
    
    # Save filtered data
    print(f"\nSaving filtered data...")
    
    # Save model1 filtered data
    model1_filtered_data = []
    for i, (item, answers) in enumerate(zip(filtered_data, filtered_model1_results)):
        filtered_item = item.copy()
        filtered_item['model_solutions'] = answers
        filtered_item['model_name'] = model1_name
        model1_filtered_data.append(filtered_item)
    
    with open(model1_filtered_file, 'w', encoding='utf-8') as f:
        json.dump(model1_filtered_data, f, ensure_ascii=False, indent=2)
    
    # Save model2 filtered data
    model2_filtered_data = []
    for i, (item, answers) in enumerate(zip(filtered_data, filtered_model2_results)):
        filtered_item = item.copy()
        filtered_item['model_solutions'] = answers
        filtered_item['model_name'] = model2_name
        model2_filtered_data.append(filtered_item)
    
    with open(model2_filtered_file, 'w', encoding='utf-8') as f:
        json.dump(model2_filtered_data, f, ensure_ascii=False, indent=2)
    
    # Create critique training data
    critique_data = create_critique_training_data(
        filtered_data, filtered_model1_results, filtered_model2_results, 
        model1_name, model2_name
    )
    
    # Save critique data
    with open(critique_file, 'w', encoding='utf-8') as f:
        json.dump(critique_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Processing completed! ===")
    print(f"Files created:")
    print(f"- Model1 filtered data: {model1_filtered_file}")
    print(f"- Model2 filtered data: {model2_filtered_file}")
    print(f"- Critique training data: {critique_file}")
    print(f"- Total filtered records: {len(filtered_data)}")
    print(f"- Critique records: {len(critique_data)}")

if __name__ == "__main__":
    main() 