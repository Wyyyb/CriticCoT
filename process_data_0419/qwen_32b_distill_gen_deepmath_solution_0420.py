from vllm import LLM, SamplingParams
from typing import List
import os
import json
import argparse
import time


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
            max_tokens=8192,
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
    input_file = "../local_data/deepmath_cft_data/deepmath_cft_step_1.json"
    output_file = "../local_data/deepmath_cft_data/deepmath_qwen_32b_distill_gen_solution_step_2.json"
    model_path = "/mnt/hwfile/opendatalab/yubo/models/DeepSeek-R1-Distill-Qwen-32B"
    # model_path = "/mnt/hwfile/opendatalab/yubo/models/Qwen2.5-32B"
    llm, sampling_params = load_vllm_model(model_path)
    with open(input_file, "r") as fi:
        deepmath_data = json.load(fi)
    input_data = []
    prompts = []
    idx = 0
    for each in deepmath_data:
        idx += 1
        question = each["question"].replace("Question:\n", "")
        input_data.append({"idx": idx, "question": question})
        idx += 1
        prompts.append(get_prompt(question))
    print("len(prompts", len(prompts))
    print("prompts[0]", prompts[0])
    # batch_size = 1000
    batch_size = 40
    # batch_num = len(prompts) // batch_size
    batch_num = 1
    outputs = []
    for index in range(batch_num):
        start_time = time.time()
        start = index * batch_size
        batch_prompt = prompts[start: start + batch_size]
        batch_output = batch_predict(llm, sampling_params, batch_prompt)
        outputs += batch_output
        print("batch_size", batch_size)
        print("single batch costing time:", time.time() - start_time)
    # outputs = batch_predict(llm, sampling_params, prompts)
    if len(outputs) != len(input_data):
        print("inconsistent length", len(outputs), len(input_data))
    output_data = []
    for i, each in enumerate(outputs):
        curr = input_data[i]
        curr["solution"] = each
        output_data.append(curr)

    with open(output_file, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


main()
