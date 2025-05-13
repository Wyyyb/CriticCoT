import json
import os
from vllm import LLM, SamplingParams
from typing import List
import time
import re
import random

invalid_count = 0
valid_count = 0

def load_vllm_model(model_path: str):
    try:
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        # 初始化模型
        if "Qwen2.5-Math-7B" in model_path:
            tp_size = min(4, len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        else:
            tp_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tp_size  # 根据GPU数量调整
        )

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=4096,
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


def get_prompt(question, gt_answer, student_answer):
    prompt = f"<|im_start|>system\nI will give you a math problem, its standard answer, and a student's answer. You do not need to solve this math problem yourself. The student's answer might be in a different format than the standard answer. Please determine if the student's answer can be considered correct based only on the standard answer. And put your final answer \"Right\" or \"Wrong\" within \\boxed{{}} <|im_end|>\n" \
             f"<|im_start|>user\nQuestion:\n{question}\nStandard Answer:\n{gt_answer}\nStudent's Answer:\n{student_answer}<|im_end|>\n" \
             f"<|im_start|>assistant\n"
    return prompt


def process(model_path, input_path, output_path, batch_idx):
    output_path = output_path.replace("$", str(batch_idx))
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    process_data = []
    for each in input_data:
        if int(each["critique_id"]) // 927 == batch_idx:
            process_data.append(each)
    random.shuffle(process_data)
    process_data = process_data[:10]
    print("number of data points to be processed: ", len(process_data))
    student_prompts = []
    critique_prompts = []
    for each in process_data:
        # print("each", each)
        question = each["question"]
        gt_answer = each["gt_answer"]
        student_answer = each["student_solution"]["extracted_answer"]
        critique_answer = each["critique_extracted_answer"]
        student_prompt = get_prompt(question, gt_answer, student_answer)
        student_prompts.append(student_prompt)
        critique_prompt = get_prompt(question, gt_answer, critique_answer)
        critique_prompts.append(critique_prompt)
    llm, sampling_params = load_vllm_model(model_path)
    students_output = batch_predict(llm, sampling_params, student_prompts)
    print("number of students output: ", len(students_output))
    critique_output = batch_predict(llm, sampling_params, critique_prompts)
    print("number of critique output: ", len(critique_output))
    for i in range(len(student_prompts)):
        curr_student_res = students_output[i]
        process_data[i]["student_solution"]["judgement"] = curr_student_res
        process_data[i]["student_solution"]["judged_correctness"] = extract_conclusion(curr_student_res)
        curr_critique_res = critique_output[i]
        process_data[i]["critique_judgement"] = curr_critique_res
        process_data[i]["critique_judged_correctness"] = extract_conclusion(curr_critique_res)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(process_data, indent=4))


def extract_conclusion(text):
    if "boxed" not in text:
        return None
    text = text.split("boxed")[-1]
    if "right" in text.lower():
        return True
    elif "wrong" in text.lower():
        return False
    return None


def main():
    input_path = "../../local_data/one_shot_data_0513/merged_critique_data_50k_0513.json"
    output_path = "../../local_data/one_shot_data_0513/judge_critique_correctness_data_50k_0513_p$.json"
    batch_idx = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(batch_idx)
    model_path = "/data/yubowang/models/Qwen2.5-Math-7B-Instruct"
    process(model_path, input_path, output_path, batch_idx)


if __name__ == "__main__":
    main()

