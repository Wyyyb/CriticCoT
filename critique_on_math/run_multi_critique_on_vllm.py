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


def get_prompt(question, solution):
    input_text = f"Question:\n{question}\nSolution:\n{solution}"
    prompt = f"<|im_start|>system\nPlease critique whether the following solution to the question is " \
             f"correct.\n\nEnd your response with either **Critique Conclusion: Correct** or " \
             f"**Critique Conclusion: Incorrect**.<|im_end|>\n\n<|im_start|>user\n{input_text}<|im_end|>\n" \
             f"<|im_start|>assistant\n"
    return prompt


def single_file_critique(file_path, model_path):
    output_path = file_path.replace(".jsonl", "_add_critique.jsonl")
    llm, sampling_params = load_vllm_model(model_path)
    input_data = []
    with open(file_path, 'r', encoding='utf-8') as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            input_data.append(curr)
    prompts = []
    for each in input_data:
        question = each["question"]
        solution = each["solution"]
        prompt = get_prompt(question, solution)
        prompts.append(prompt)
    outputs = batch_predict(llm, sampling_params, prompts)
    if len(outputs) != len(prompts):
        print("inconsistent length of outputs and prompts", len(outputs), len(prompts))
    output_data = []
    for i, each in enumerate(outputs):
        critic_res = extract_critic_res(each)
        curr = input_data[i]
        score = curr["score"][0]
        critic_score = critic_res == score
        curr["critique_pred"] = critic_res
        curr["critique_output"] = each
        curr["critique_score"] = critic_score
        output_data.append(curr)
    with open(output_path, "w") as fo:
        for each in output_data:
            fo.write(json.dumps(each) + "\n")


def extract_critic_res(response):
    # 忽略大小写
    response = response.lower().replace("**", "").replace("##", "")

    # 定义正确和错误的关键词
    correct_patterns = [
        "conclusion: correct",
        "conclusion:correct",
        "conclusion: right",
        "conclusion:right",
        "conclusion: true",
        "conclusion:true",
        "critique: correct",
        "critique:correct",
        "critique: right",
        "critique:right",
        "critique: true",
        "critique:true"
    ]

    incorrect_patterns = [
        "conclusion: incorrect",
        "conclusion:incorrect",
        "conclusion: wrong",
        "conclusion:wrong",
        "conclusion: false",
        "conclusion:false",
        "critique: incorrect",
        "critique:incorrect",
        "critique: wrong",
        "critique:wrong",
        "critique: false",
        "critique:false"
    ]

    # 检查正确模式
    for pattern in correct_patterns:
        if pattern in response:
            return True

    # 检查错误模式
    for pattern in incorrect_patterns:
        if pattern in response:
            return False

    # 如果没有找到匹配模式,返回None
    return None


def main(model_path, input_dir):
    for each in os.listdir(os.path.join(input_dir, "math")):
        if not each.endswith(".jsonl") or "add_critique" in each:
            continue
        single_file_critique(os.path.join(input_dir, "math", each), model_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process math training files')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing math_train folder')

    args = parser.parse_args()
    main(args.model_path, args.input_dir)

