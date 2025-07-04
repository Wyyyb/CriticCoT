#!/usr/bin/env python3
"""
Test script for generate_critique_data.py logic
"""

import json
from generate_critique_data import create_critique_prompt, create_critique_target

def test_critique_prompt():
    """Test critique prompt generation"""
    print("Testing critique prompt generation:")
    
    question = "What is 2 + 2?"
    solution = "Let me solve this step by step:\n2 + 2 = 4\nTherefore, the answer is \\boxed{4}"
    
    prompt = create_critique_prompt(question, solution)
    print("Generated prompt:")
    print(prompt)
    print()

def test_critique_target():
    """Test critique target generation"""
    print("Testing critique target generation:")
    
    test_cases = [
        # Correct solutions
        ("The answer is \\boxed{4}", "4", "right"),
        ("Therefore, \\boxed{3.14}", "3.14", "right"),
        ("So the result is 100", "100", "right"),
        
        # Incorrect solutions
        ("The answer is \\boxed{5}", "4", "wrong"),
        ("Therefore, \\boxed{3.15}", "3.14", "wrong"),
        ("So the result is 99", "100", "wrong"),
        
        # Edge cases
        ("", "4", "wrong"),
        ("The answer is \\boxed{4}", "", "wrong"),
    ]
    
    for solution, ground_truth, expected in test_cases:
        target = create_critique_target(solution, ground_truth)
        status = "✓" if target == expected else "✗"
        print(f"{status} Solution: '{solution}'")
        print(f"   Ground truth: '{ground_truth}'")
        print(f"   Target: '{target}' (expected: '{expected}')")
        print()

def test_sample_data():
    """Test with sample data structure"""
    print("Testing with sample data structure:")
    
    # Create sample filtered data
    sample_filtered_data = [
        {
            'question': 'What is 2 + 2?',
            'gt_answer': '4',
            'subject': 'arithmetic',
            'level': 'elementary',
            'qwen25_solutions': [
                'The answer is \\boxed{4}',
                '2 + 2 = 4, so the answer is \\boxed{4}'
            ],
            'qwen3_solutions': [
                'Let me calculate: 2 + 2 = 4. Therefore, \\boxed{4}',
                'The answer is \\boxed{5}'  # Wrong answer
            ]
        }
    ]
    
    # Save sample data
    with open('sample_filtered.json', 'w', encoding='utf-8') as f:
        json.dump(sample_filtered_data, f, ensure_ascii=False, indent=2)
    
    print("Created sample_filtered.json")
    print("Sample data structure:")
    print(json.dumps(sample_filtered_data[0], indent=2))
    print()
    
    # Test critique generation for this sample
    question = sample_filtered_data[0]['question']
    ground_truth = sample_filtered_data[0]['gt_answer']
    
    print("Testing critique generation for sample:")
    for i, solution in enumerate(sample_filtered_data[0]['qwen25_solutions'] + sample_filtered_data[0]['qwen3_solutions']):
        prompt = create_critique_prompt(question, solution)
        target = create_critique_target(solution, ground_truth)
        
        print(f"Solution {i+1}:")
        print(f"  Solution: {solution}")
        print(f"  Target: {target}")
        print(f"  Prompt preview: {prompt[:100]}...")
        print()

if __name__ == "__main__":
    test_critique_prompt()
    test_critique_target()
    test_sample_data()
    print("All tests completed!") 