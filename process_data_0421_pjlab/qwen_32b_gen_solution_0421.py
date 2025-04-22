from vllm import LLM, SamplingParams
from typing import List
import os
import json
import argparse
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
    # prompt = f"<|im_start|>You are a science expert. A student is trying to solve a question, please explain " \
    #          f"briefly whether his answer is correct or not. Finally, conclude your judgement with " \
    #          f"'Conclusion: right/wrong [END]\n\n<|im_end|>\n\n" \
    #          f"<|im_start|>user\n{question}<|im_end|>\n" \
    #          f"<|im_start|>assistant\n"
    return prompt


def get_process_data(output_data):
    process_data = []
    sta_count = {"qwen-2.5-32b_answer_valid": 0, "qwen-2.5-32b_answer_correct": 0,
                 "qwen-2.5-32b_answer_invalid": 0, "qwen-2.5-32b_answer_incorrect": 0}
    for k, v in output_data.items():
        # if "qwen-2.5-32b_answer" not in v:
        #     continue
        if "qwen-2.5-32b_answer_valid" in v and v["qwen-2.5-32b_answer_valid"] is True:
            sta_count["qwen-2.5-32b_answer_valid"] += 1
            if v.get("qwen-2.5-32b_answer_correctness") is True:
                sta_count["qwen-2.5-32b_answer_correct"] += 1
            else:
                sta_count["qwen-2.5-32b_answer_incorrect"] += 1
            continue
        sta_count["qwen-2.5-32b_answer_invalid"] += 1
        # if "DeepSeek-R1-Distill-Qwen-32B_critique" in v and
        # v.get("DeepSeek-R1-Distill-Qwen-32B_critique_valid", False) is True:
        #     continue
        process_data.append(v)
    print("new round to process data number:", len(process_data))
    print("sta_count", sta_count)
    return process_data


def filter_output_data(output_data, start, end):
    res = {}
    for k, v in output_data.items():
        if start <= int(v["idx"]) < end:
            res[k] = v
    return res


def parse_output(output):
    if not output or len(output) < 100:
        return "too short", False
    if "conclusion" in output.lower() or "right" in output.lower() or "wrong" in output.lower():
        return "format recognized", True
    else:
        return "format not recognized", True


def extract_boxed_answer(text):
    """
    从输入的字符串中提取\boxed{ANSWER}中的ANSWER部分。
    如果找不到\boxed{}的模式，返回None。
    参数:
        text (str): 输入的字符串
    返回:
        str 或 None: 提取到的答案字符串，如果没有找到则返回None
    """
    pattern = r'\\boxed\{(.*?)\}'
    match = re.search(pattern, text)

    if match:
        return match.group(1)
    else:
        return None


def main():
    input_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0421.json"
    output_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0421_add_solution_p1.json"
    start_idx, end_idx = 0, 110000
    # start_idx, end_idx = 0, 110000
    # model_path = "/map-vepfs/yubo/models/DeepSeek-R1-Distill-Qwen-32B"
    model_path = "/mnt/hwfile/opendatalab/yubo/models/Qwen2.5-32B"

    # 检查是否有中间结果文件存在
    import os
    # temp_output_file = output_file + ".temp"
    output_data = []
    processed_count = 0

    if os.path.exists(output_file):
        # 如果存在临时文件，加载已处理的数据
        with open(output_file, "r") as fi:
            output_data = json.load(fi)
        # print("sta_output_data(output_data)", sta_output_data(output_data))
    else:
        with open(input_file, "r") as fi:
            output_data = json.load(fi)
        print("input data length", len(output_data))
        output_data = filter_output_data(output_data, start=start_idx, end=end_idx)
        print("filtered output data length", len(output_data))
    llm, sampling_params = load_vllm_model(model_path)

    process_data = get_process_data(output_data)
    while len(process_data) > 500:
        prompts = []
        for each in process_data:
            question = each["question"]
            prompts.append(get_prompt(question))

        print("len(prompts)", len(prompts))
        print("prompts[0]", prompts[0])

        # 设置批处理大小
        # batch_size = 1000
        batch_size = 1000

        total_batches = (len(prompts) + batch_size - 1) // batch_size  # 向上取整

        for batch_idx in range(total_batches):
            start_time = time.time()
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompt = prompts[batch_start:batch_end]

            print(
                f"Processing batch {batch_idx + 1}/{total_batches}, items {processed_count + batch_start + 1} to {processed_count + batch_end}")
            batch_output = batch_predict(llm, sampling_params, batch_prompt)
            batch_sta = {"total_num": 0, "valid_num": 0, "invalid_num": 0,
                         "right_num": 0, "wrong_num": 0}
            # 添加这批次的结果到输出数据
            for i, output in enumerate(batch_output):
                batch_sta["total_num"] += 1
                if i == 0:
                    print("output sample:\n", output)
                question = process_data[i]["question"]
                output_data[question]["qwen-2.5-32b_answer"] = output
                extracted_answer = extract_boxed_answer(output)
                if extracted_answer is None:
                    output_data[question]["qwen-2.5-32b_answer_valid"] = False
                    batch_sta["invalid_num"] += 1
                else:
                    output_data[question]["qwen-2.5-32b_answer_valid"] = True
                    batch_sta["valid_num"] += 1
                    if extracted_answer == output_data[question]["gt_answer"]:
                        output_data[question]["qwen-2.5-32b_answer_correctness"] = True
                        batch_sta["right_num"] += 1
                    else:
                        output_data[question]["qwen-2.5-32b_answer_correctness"] = False
                        batch_sta["wrong_num"] += 1
            print("curr batch_sta:\n", batch_sta)
            # 每完成一批次就保存临时文件
            with open(output_file, "w") as fo:
                fo.write(json.dumps(output_data, indent=4))
            print(f"Batch {batch_idx + 1} complete. bs={batch_end - batch_start}. "
                  f"Progress saved. Batch processing time:", time.time() - start_time)

        process_data = get_process_data(output_data)


main()
