import json
import os
from vllm import LLM, SamplingParams
from typing import List
import time
import re


def load_vllm_model(model_path: str):
    try:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        # 初始化模型
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))  # 根据GPU数量调整
        )

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=16384,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in model_path.lower()
                else None
            ),
            top_p=0.95
        )
        return llm, sampling_params
    except Exception as e:
        print("load vllm model failed", e)
        return None, None


def batch_predict(llm, sampling_params, prompts: List[str]) -> List[str]:
    if not llm or not sampling_params:
        print("llm, sampling_params are None")
        return []
    try:
        print("Processing", len(prompts))
        outputs = llm.generate(prompts, sampling_params)
        # 提取生成的文本
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        return results
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return []


def get_prompt(question, model_type):
    question = "Question:\n" + question
    if model_type == "qwen3":
        prompt = f"<|im_start|>user\nPlease reason step by step to find a solution to the following " \
             f"question, and put your final answer within \\boxed{{}}.\n{question}<|im_end|>\n" \
             f"<|im_start|>assistant\n"
    elif model_type == "qwen2-5":
        prompt = f"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
                 f"<|im_start|>user\n{question}<|im_end|>\n" \
                 f"<|im_start|>assistant\n"
    elif model_type == "phi4":
        prompt = f"<|im_start|>system<|im_sep|>\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n" \
                 f"<|im_start|>user<|im_sep|>\n{question}<|im_end|>\n" \
                 f"<|im_start|>assistant<|im_sep|>\n"
    else:
        print("unsupported model type")
        return None
    return prompt


def single_model_inference(model_path, input_data):




def main():








