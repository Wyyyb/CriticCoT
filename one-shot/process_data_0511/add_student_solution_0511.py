import json
import os
from vllm import LLM, SamplingParams
from typing import List
import time
import re

invalid_count = 0
valid_count = 0

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


def single_model_inference(model_path, model_name, model_type, output_path):
    global invalid_count, valid_count
    prompts = []
    questions = []
    with open(output_path, "r") as f:
        input_data = json.load(f)
    for k, v in input_data.items():
        count = 10
        if "student_solutions" not in v:
            v["student_solutions"] = {}
        if model_name not in v["student_solutions"]:
            v["student_solutions"][model_name] = []
        for each in v["student_solutions"][model_name]:
            if each.get("extracted_answer", None) is not None:
                count -= 1
            else:
                print("invalid answer", model_name, v["question"])
        prompt = get_prompt(v["question"], model_type)
        for _ in range(count):
            questions.append(v["question"])
            prompts.append(prompt)

    # batch predict
    llm, sampling_params = load_vllm_model(model_path)
    outputs = batch_predict(llm, sampling_params, prompts)
    if len(outputs) != len(questions):
        print("inconsistent output length", len(outputs), len(questions))

    # postprocess
    for i, question in enumerate(questions):
        solution = outputs[i]
        item = get_single_solution_item(solution, input_data[question]["gt_answer"])
        if item["extracted_answer"] is None:
            invalid_count += 1
            continue
        input_data[question]["student_solutions"][model_name].append(item)
        valid_count += 1
    with open(output_path, "w") as f:
        f.write(json.dumps(input_data, indent=4))
    # return input_data


def extract_boxed_answer(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def get_single_solution_item(solution, gt_answer):
    item = {}
    if "</think>" in solution:
        solution = solution.split("</think>")[-1]
    extract_answer = extract_boxed_answer(solution)
    item["extracted_answer"] = extract_answer
    item["solution"] = solution
    item["exact_match_correctness"] = extract_answer == gt_answer
    return item


def main():
    model_dir = "/data/yubo/models"
    model_info = {"Qwen2.5-Math-7B-Instruct": "qwen2-5",
                  "Qwen3-4B": "qwen3",
                  "Qwen3-8B": "qwen3",
                  "Qwen3-14B": "qwen3",
                  "Qwen3-32B": "qwen3",
                  "Phi-4-reasoning": "phi4",
                  "Phi-4-reasoning-plus": "phi4",
                  "MiMo-7B-SFT": "qwen3",
                  "MiMo-7B-RL": "qwen3",
                  "DeepSeek-R1-Distill-Qwen-32B": "qwen3"}
    output_path = "/data/yubo/CriticCoT/local_data/one_shot_data_0511/seed_questions_add_solution_0512.json"

    for model_name, model_type in model_info.items():
        print(f"processing {model_name}")
        model_path = os.path.join(model_dir, model_name)
        single_model_inference(model_path, model_name, model_type, output_path)


if __name__ == "__main__":
    main()

