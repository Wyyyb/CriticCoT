import random

from vllm import LLM, SamplingParams
from typing import List
import os
import json
import argparse


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
            temperature=0.0,
            max_tokens=2048,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in model_path.lower()
                else None
            ),
            top_p=1.0
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


def get_prompt(question):
    question = "Question:\n" + question
    prompt = f"<|im_start|>Please reason step by step to find a solution to the following question, " \
             f"and put your final answer within \\boxed{{}}.<|im_end|>\n\n" \
             f"<|im_start|>user\n{question}<|im_end|>\n" \
             f"<|im_start|>assistant\n"
    return prompt


def main():
    input_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/ace_data_0119.jsonl"
    output_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/on_policy_data_0119/" \
                  "qwen_math_ace_80k_0119.json"
    model_path = "/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B"
    llm, sampling_params = load_vllm_model(model_path)
    ace_data = []
    with open(input_file, "r") as fi:
        for line in fi.readlines():
            ace_data.append(json.loads(line))
    random.shuffle(ace_data)
    ace_data = ace_data[:80000]
    input_data = []
    prompts = []

    for each in ace_data:
        idx = each["idx"]
        question = each["question"]
        ace_math_solution = each["ace_math_solution"]
        input_data.append({"idx": idx, "question": question, "ace_math_solution": ace_math_solution})
        prompts.append(get_prompt(question))
    print("len(prompts) = ", len(prompts))
    outputs = batch_predict(llm, sampling_params, prompts)
    if len(outputs) != len(input_data):
        print("inconsistent length", len(outputs), len(input_data))
    output_data = []
    for i, each in enumerate(outputs):
        curr = input_data[i]
        curr["qwen_2.5_math_answer"] = each
        output_data.append(curr)

    with open(output_file, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


main()
