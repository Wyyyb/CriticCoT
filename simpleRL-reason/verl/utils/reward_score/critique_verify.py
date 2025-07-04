# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import re
from .qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
# from .qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
from functools import partial
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import threading
import logging
from typing import Optional, Callable, Any
from functools import wraps
import random
import gc 
from math_verify import parse, verify

class GlobalProcessPool:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, max_workers: int = 16, reset_threshold: int = 100000):
        self.max_workers = max_workers
        self.reset_threshold = reset_threshold
        self.task_counter = 0
        self.executor: Optional[ProcessPoolExecutor] = None
        self.logger = logging.getLogger(__name__)
        self._initialize_executor()
    
    def _initialize_executor(self) -> None:
        """Initialize a new ProcessPoolExecutor and reset task counter."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
            gc.collect() 
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_counter = 0
        self.logger.warning(f"Initialized ProcessPoolExecutor with {self.max_workers} workers")
    
    @classmethod
    def get_instance(cls, max_workers: int = 16, reset_threshold: int = 100000) -> 'GlobalProcessPool':
        """Get or create the singleton instance of GlobalProcessPool."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_workers=max_workers, reset_threshold=reset_threshold)
        return cls._instance
    
    def submit(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Submit a task to the executor with automatic recovery and periodic reset.
        
        Args:
            fn: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object representing the computation
        """
        try:
            if self.executor is None:
                with self._lock:
                    self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)
        except (Exception, RuntimeError) as e:
            self.logger.warning(f"Process pool broken, recreating: {str(e)}")
            with self._lock:
                self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)

# Create the global executor instance
global_executor = GlobalProcessPool.get_instance(max_workers=16)

def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None

def extract_conclusion_pattern(text):
    """
    提取文本中的结论模式，如 "\\n\\nConclusion: right [END]\\n\\n"
    
    Args:
        text: 输入文本
        
    Returns:
        tuple: (conclusion_text, is_correct, has_conclusion_format)
        - conclusion_text: 提取的结论文本
        - is_correct: 根据结论判断是否正确 (True/False)
        - has_conclusion_format: 是否包含结论格式 (True/False)
    """
    # 匹配结论模式，支持大小写不敏感
    pattern = r'\n\nConclusion:\s*(right|wrong|correct|incorrect)\s*\[END\]\n\n'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        conclusion_word = match.group(1).lower()
        # 判断是否正确：right 或 correct 表示正确
        is_correct = conclusion_word in ['right', 'correct']
        return match.group(0), is_correct, True
    else:
        return None, False, False

def extract_solution(solution_str):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    predict_answer = qwen_extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)
    # True means the boxed answer is correct
    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        return predict_answer, False


def hf_verify_with_try(gold, target):
    try:
        parsed_target = parse(target)    
        parsed_gold = parse(gold)
        return verify(gold=parsed_gold, target=parsed_target)
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False


def hf_math_equal_subprocess(gold, target, timeout_seconds=10):
    """
    使用 ProcessPoolExecutor 实现带超时的函数执行
    
    Args:
        gold: 参考答案
        target: 预测结果
        timeout_seconds: 超时时间(秒)
        
    Returns:
        bool: 执行结果,超时返回 False
    """
    try:
        # 提交任务到进程池
        future = global_executor.submit(hf_verify_with_try, gold=gold, target=target)
        # 等待结果,支持超时
        result = future.result(timeout=timeout_seconds)
        return result
    except TimeoutError:
        print(f"Timeout occurred for gold {gold} and target {target}.")
        return False
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False


import os 
# TODO: Might have problem in multi node ray cluster !!!!
reward_function_type = str(os.environ.get('REWORD_FUNCTION_TYPE', "mix"))
format_penalty_value = float(os.environ.get('FORMAT_PENALTY_VALUE', "-1"))

print(f"Reward function type: {reward_function_type}")
print(f"Format penalty value: {format_penalty_value}")

def compute_score(solution_str, ground_truth, method='strict'):
    """The scoring function for critique verification.

    Args:
        solution_str: the solution text containing critique conclusion
        ground_truth: the expected conclusion (e.g., "right" or "wrong")
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        
    Returns:
        dict: {"score": float, "correctness": bool}
        - score: combined score for both conclusion correctness and format correctness
        - correctness: whether the conclusion is correct (True/False)
    """
    # 提取结论模式
    conclusion_text, conclusion_is_correct, has_conclusion_format = extract_conclusion_pattern(solution_str)
    
    # 判断结论是否正确（与ground_truth比较）
    # ground_truth应该是"right"或"wrong"
    expected_correct = ground_truth.lower() in ['right', 'correct']
    conclusion_correct = conclusion_is_correct == expected_correct
    
    # 计算分数
    if reward_function_type == 'mix':
        # 只关注结论正确性
        if conclusion_correct:
            score = 1.0
        else:
            score = 0.0
    elif reward_function_type == 'independent':
        # 同时考虑结论正确性和格式规范性
        if conclusion_correct and has_conclusion_format:
            score = 1.0
        elif conclusion_correct and not has_conclusion_format:
            score = 0.5
        elif not conclusion_correct and has_conclusion_format:
            score = -0.5
        else:
            score = format_penalty_value
    else:
        raise ValueError(f"Invalid reward function type: {reward_function_type}")
            
    if random.random() < 0.05:
        # for 5% of the cases, print; otherwise, print nothing to accelerate the process 
        print(f"\n[Model Response]\n{solution_str}")
        print(f"\n[Ground Truth Conclusion]\n{ground_truth}")
        print(f"\n[Conclusion Pattern]\n{conclusion_text}")
        print(f"\n[Has Conclusion Format]\n{has_conclusion_format}")
        print(f"\n[Extracted Conclusion Is Correct]\n{conclusion_is_correct}")
        print(f"\n[Expected Conclusion Is Correct]\n{expected_correct}")
        print(f"\n[Conclusion Correct]\n{conclusion_correct}")
        print(f"\n[Combined Score]\n{score}")
    
    return {"score": score, "correctness": conclusion_correct}



if __name__ == "__main__":
    # 测试新的结论提取逻辑
    test_cases = [
        {
            "name": "Correct conclusion with right",
            "text": "The student's solution and final answer \\boxed{8\\pi i} are completely correct.\n\nConclusion: right [END]\n\n",
            "expected_correct": True,
            "expected_has_format": True
        },
        {
            "name": "Correct conclusion with correct",
            "text": "The student's solution and final answer \\boxed{8\\pi i} are completely correct.\n\nConclusion: correct [END]\n\n",
            "expected_correct": True,
            "expected_has_format": True
        },
        {
            "name": "Incorrect conclusion with wrong",
            "text": "The student's solution has errors.\n\nConclusion: wrong [END]\n\n",
            "expected_correct": False,
            "expected_has_format": True
        },
        {
            "name": "Incorrect conclusion with incorrect",
            "text": "The student's solution has errors.\n\nConclusion: incorrect [END]\n\n",
            "expected_correct": False,
            "expected_has_format": True
        },
        {
            "name": "Case insensitive test",
            "text": "Some text.\n\nConclusion: RIGHT [END]\n\n",
            "expected_correct": True,
            "expected_has_format": True
        },
        {
            "name": "No conclusion format",
            "text": "Just some regular text without conclusion format.",
            "expected_correct": False,
            "expected_has_format": False
        }
    ]
    
    print("Testing extract_conclusion_pattern function:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {repr(test_case['text'])}")
        
        conclusion_text, is_correct, has_format = extract_conclusion_pattern(test_case['text'])
        
        print(f"Extracted conclusion: {repr(conclusion_text)}")
        print(f"Is correct: {is_correct}")
        print(f"Has format: {has_format}")
        print(f"Expected correct: {test_case['expected_correct']}")
        print(f"Expected has format: {test_case['expected_has_format']}")
        
        if is_correct == test_case['expected_correct'] and has_format == test_case['expected_has_format']:
            print("✅ PASS")
        else:
            print("❌ FAIL")
    
    print("\n" + "=" * 50)
    print("Testing compute_score function:")
    
    # 测试compute_score函数 - 新的逻辑
    test_cases_score = [
        {
            "name": "Correct conclusion with right ground truth",
            "solution": "The student's solution and final answer \\boxed{8\\pi i} are completely correct.\n\nConclusion: right [END]\n\n",
            "ground_truth": "right",
            "expected_correctness": True,
            "expected_score": 1.0
        },
        {
            "name": "Correct conclusion with wrong ground truth",
            "solution": "The student's solution and final answer \\boxed{8\\pi i} are completely correct.\n\nConclusion: right [END]\n\n",
            "ground_truth": "wrong",
            "expected_correctness": False,
            "expected_score": 0.0
        },
        {
            "name": "Wrong conclusion with wrong ground truth",
            "solution": "The student's solution has errors.\n\nConclusion: wrong [END]\n\n",
            "ground_truth": "wrong",
            "expected_correctness": True,
            "expected_score": 1.0
        },
        {
            "name": "No conclusion format with right ground truth",
            "solution": "Just some regular text without conclusion format.",
            "ground_truth": "right",
            "expected_correctness": False,
            "expected_score": 0.0
        }
    ]
    
    for i, test_case in enumerate(test_cases_score, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Solution: {repr(test_case['solution'])}")
        print(f"Ground truth: {test_case['ground_truth']}")
        
        result = compute_score(test_case['solution'], test_case['ground_truth'])
        
        print(f"Result: {result}")
        print(f"Expected correctness: {test_case['expected_correctness']}")
        print(f"Expected score: {test_case['expected_score']}")
        
        if (result['correctness'] == test_case['expected_correctness'] and 
            abs(result['score'] - test_case['expected_score']) < 0.001):
            print("✅ PASS")
        else:
            print("❌ FAIL")