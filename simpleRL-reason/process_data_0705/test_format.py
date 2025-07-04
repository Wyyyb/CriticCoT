#!/usr/bin/env python3
"""
Test script for format_critique_data.py logic
"""

import json
import tempfile
import os
from format_critique_data import format_critique_data, analyze_formatted_data

def test_format_critique_data():
    """Test critique data formatting"""
    print("Testing critique data formatting:")
    
    # Create sample critique data
    sample_critique_data = [
        {
            "prompt": "You are a mathematics expert. A student is trying to solve a question. Please explain briefly whether his answer is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]'\n\nQuestion: What is 2 + 2?\n\nStudent's solution:\nThe answer is \\boxed{4}\n\nCritique:",
            "target": "right",
            "question": "What is 2 + 2?",
            "solution": "The answer is \\boxed{4}",
            "gt_answer": "4",
            "subject": "Mathematics",
            "level": 1,
            "data_source": "deepscaler_critique",
            "extra_info": {
                "original_index": 0,
                "solution_index": 0,
                "model_source": "qwen25"
            }
        },
        {
            "prompt": "You are a mathematics expert. A student is trying to solve a question. Please explain briefly whether his answer is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]'\n\nQuestion: What is 2 + 2?\n\nStudent's solution:\nThe answer is \\boxed{5}\n\nCritique:",
            "target": "wrong",
            "question": "What is 2 + 2?",
            "solution": "The answer is \\boxed{5}",
            "gt_answer": "4",
            "subject": "Mathematics",
            "level": 1,
            "data_source": "deepscaler_critique",
            "extra_info": {
                "original_index": 0,
                "solution_index": 1,
                "model_source": "qwen3"
            }
        }
    ]
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_critique_file = f.name
        json.dump(sample_critique_data, f, ensure_ascii=False, indent=2)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_output_file = f.name
    
    try:
        # Test formatting
        format_critique_data(temp_critique_file, temp_output_file)
        
        # Load and check formatted data
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            formatted_data = json.load(f)
        
        print(f"Formatted {len(formatted_data)} items")
        
        # Check structure of first item
        if formatted_data:
            first_item = formatted_data[0]
            print("\nFirst formatted item structure:")
            print(json.dumps(first_item, indent=2, ensure_ascii=False))
            
            # Check required fields
            required_fields = ['answer', 'gt_answer', 'subject', 'level', 'question', 'target', 
                             'data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
            
            print("\nField presence check:")
            for field in required_fields:
                present = field in first_item
                status = "✓" if present else "✗"
                print(f"{status} {field}: {present}")
            
            # Check specific field values
            print("\nField value check:")
            checks = [
                ("answer", "right"),
                ("gt_answer", "right"),
                ("target", "right"),
                ("data_source", "deepscaler_critique"),
                ("ability", "critique"),
                ("reward_model.ground_truth", "right"),
                ("reward_model.style", "rule")
            ]
            
            for field_path, expected_value in checks:
                if '.' in field_path:
                    # Nested field
                    parts = field_path.split('.')
                    value = first_item
                    for part in parts:
                        value = value.get(part, None)
                        if value is None:
                            break
                else:
                    value = first_item.get(field_path, None)
                
                status = "✓" if value == expected_value else "✗"
                print(f"{status} {field_path}: '{value}' (expected: '{expected_value}')")
        
        # Test analysis
        print("\n" + "="*50)
        analyze_formatted_data(temp_output_file)
        
    finally:
        # Clean up temporary files
        os.unlink(temp_critique_file)
        os.unlink(temp_output_file)

def test_format_comparison():
    """Test that formatted data matches train data structure"""
    print("\n" + "="*50)
    print("Testing format comparison with train data structure:")
    
    # Sample train data structure (from user's example)
    train_data_structure = {
        "answer": "26",
        "gt_answer": "26",
        "subject": "Mathematics",
        "level": 5,
        "question": "Doug constructs a square window...",
        "target": "26",
        "data_source": "deepscaler",
        "prompt": [
            {
                "content": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
                "role": "user"
            }
        ],
        "ability": "math",
        "reward_model": {
            "ground_truth": "26",
            "style": "rule"
        },
        "extra_info": {
            "answer": "26",
            "index": 1,
            "level": 5,
            "split": "train"
        }
    }
    
    # Sample formatted critique data structure
    formatted_critique_structure = {
        "answer": "right",
        "gt_answer": "right",
        "subject": "Mathematics",
        "level": 1,
        "question": "What is 2 + 2?",
        "target": "right",
        "data_source": "deepscaler_critique",
        "prompt": [
            {
                "content": "You are a mathematics expert...",
                "role": "user"
            }
        ],
        "ability": "critique",
        "reward_model": {
            "ground_truth": "right",
            "style": "rule"
        },
        "extra_info": {
            "answer": "right",
            "index": 0,
            "level": 1,
            "split": "train",
            "original_index": 0,
            "solution_index": 0,
            "model_source": "qwen25",
            "solution": "The answer is \\boxed{4}",
            "question_gt_answer": "4"
        }
    }
    
    # Compare structures
    train_keys = set(train_data_structure.keys())
    critique_keys = set(formatted_critique_structure.keys())
    
    print("Key comparison:")
    print(f"Train data keys: {len(train_keys)}")
    print(f"Critique data keys: {len(critique_keys)}")
    
    missing_in_critique = train_keys - critique_keys
    extra_in_critique = critique_keys - train_keys
    
    if missing_in_critique:
        print(f"Missing in critique: {missing_in_critique}")
    else:
        print("✓ All train data keys present in critique data")
    
    if extra_in_critique:
        print(f"Extra in critique: {extra_in_critique}")
    
    print("\nStructure compatibility: ✓" if not missing_in_critique else "\nStructure compatibility: ✗")

if __name__ == "__main__":
    test_format_critique_data()
    test_format_comparison()
    print("\nAll tests completed!") 