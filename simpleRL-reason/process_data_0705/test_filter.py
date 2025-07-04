#!/usr/bin/env python3
"""
Test script for filter_solutions.py logic
"""

import json
from filter_solutions import extract_answer_from_boxed, is_answer_correct, check_model_consistency

def test_answer_extraction():
    """Test answer extraction from different formats"""
    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("Therefore, \\boxed{3.14}", "3.14"),
        ("The answer is 42", "42"),
        ("Therefore, 3.14", "3.14"),
        ("So the result is 100", "100"),
        ("Hence, x = 5", "x = 5"),
        ("Thus, the answer is 7", "7"),
        ("Answer: 42", "42"),
        ("Result: 3.14", "3.14"),
    ]
    
    print("Testing answer extraction:")
    for text, expected in test_cases:
        result = extract_answer_from_boxed(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text}' -> '{result}' (expected: '{expected}')")

def test_answer_correctness():
    """Test answer correctness checking"""
    test_cases = [
        ("42", "42", True),
        ("3.14", "3.14", True),
        ("42", "43", False),
        ("3.14", "3.14159", False),
        ("x = 5", "x=5", True),
        ("5", "5.0", True),
        ("", "42", False),
        ("42", "", False),
    ]
    
    print("\nTesting answer correctness:")
    for pred, gt, expected in test_cases:
        result = is_answer_correct(pred, gt)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{pred}' vs '{gt}' -> {result} (expected: {expected})")

def test_model_consistency():
    """Test model consistency checking"""
    test_cases = [
        # All correct
        (["\\boxed{42}", "\\boxed{42}", "The answer is 42"], "42", True, False),
        # All incorrect
        (["\\boxed{43}", "\\boxed{44}", "The answer is 45"], "42", False, True),
        # Mixed
        (["\\boxed{42}", "\\boxed{43}", "The answer is 42"], "42", False, False),
        # Empty
        ([], "42", False, False),
    ]
    
    print("\nTesting model consistency:")
    for solutions, gt, expected_all_correct, expected_all_incorrect in test_cases:
        all_correct, all_incorrect = check_model_consistency(solutions, gt)
        status = "✓" if (all_correct == expected_all_correct and all_incorrect == expected_all_incorrect) else "✗"
        print(f"{status} Solutions: {solutions}")
        print(f"   Ground truth: {gt}")
        print(f"   All correct: {all_correct} (expected: {expected_all_correct})")
        print(f"   All incorrect: {all_incorrect} (expected: {expected_all_incorrect})")

def test_filtering_logic():
    """Test the filtering logic"""
    print("\nTesting filtering logic:")
    
    # Case 1: Both models all correct -> should be removed
    qwen25_all_correct, qwen25_all_incorrect = True, False
    qwen3_all_correct, qwen3_all_incorrect = True, False
    should_keep = not (qwen25_all_correct and qwen3_all_correct) and not (qwen25_all_incorrect and qwen3_all_incorrect)
    print(f"Both all correct: should_keep = {should_keep} (expected: False)")
    
    # Case 2: Both models all incorrect -> should be removed
    qwen25_all_correct, qwen25_all_incorrect = False, True
    qwen3_all_correct, qwen3_all_incorrect = False, True
    should_keep = not (qwen25_all_correct and qwen3_all_correct) and not (qwen25_all_incorrect and qwen3_all_incorrect)
    print(f"Both all incorrect: should_keep = {should_keep} (expected: False)")
    
    # Case 3: Mixed results -> should be kept
    qwen25_all_correct, qwen25_all_incorrect = False, False
    qwen3_all_correct, qwen3_all_incorrect = False, False
    should_keep = not (qwen25_all_correct and qwen3_all_correct) and not (qwen25_all_incorrect and qwen3_all_incorrect)
    print(f"Mixed results: should_keep = {should_keep} (expected: True)")

if __name__ == "__main__":
    test_answer_extraction()
    test_answer_correctness()
    test_model_consistency()
    test_filtering_logic()
    print("\nAll tests completed!") 