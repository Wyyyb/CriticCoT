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
            temperature=0.6,
            max_tokens=16384*2,
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
    # prompt = f"<|im_start|>Please reason step by step to find a solution to the following question, " \
    #          f"and put your final answer within \\boxed{{}}.<|im_end|>\n\n" \
    #          f"<|im_start|>user\n{question}<|im_end|>\n" \
    #          f"<|im_start|>assistant\n"
    prompt = f"<|im_start|>You are a science expert. A student is trying to solve a question, please explain " \
             f"briefly whether his answer is correct or not. Finally, conclude your judgement with " \
             f"'Conclusion: right/wrong [END]\n\n<|im_end|>\n\n" \
             f"<|im_start|>user\n{question}\n<|im_end|>\n" \
             f"<|im_start|>assistant\nCritique:\n"
    return prompt


def get_process_data(output_data):
    process_data = []
    sta_count = {"DeepSeek-R1-Distill-Qwen-32B_critique": 0,
                 "DeepSeek-R1-Distill-Qwen-32B_critique_valid": 0,
                 "qwen-2.5-32b_answer_valid": 0}
    for k, v in output_data.items():
        if len(v["qwen-2.5-32b_answer"]) > 50000:
            continue
        if "qwen-2.5-32b_answer" in v and v.get("qwen-2.5-32b_answer_valid") is True:
            sta_count["qwen-2.5-32b_answer_valid"] += 1
            if "DeepSeek-R1-Distill-Qwen-32B_critique" in v:
                sta_count["DeepSeek-R1-Distill-Qwen-32B_critique"] += 1
            if "DeepSeek-R1-Distill-Qwen-32B_critique" in v and \
                    v.get("DeepSeek-R1-Distill-Qwen-32B_critique_valid", False) is True:
                sta_count["DeepSeek-R1-Distill-Qwen-32B_critique_valid"] += 1
                continue
            process_data.append(v)
    print("new round to process data number:", len(process_data))
    print("sta_count", sta_count)
    return process_data


def filter_output_data(output_data, start=0, end=10000):
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


def get_cft_format_question(each):
    question = each["question"]
    solution = each["qwen-2.5-32b_answer"]
    format_question = f"Question:\n{question}\n\nStudent's Solution:\n{solution}"
    return format_question


def main():
    input_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0421_add_solution_p1.json"
    output_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0422_add_critique_p1.json"
    start_idx, end_idx = 0, 110000
    model_path = "/mnt/hwfile/opendatalab/yubo/models/DeepSeek-R1-Distill-Qwen-32B"
    # model_path = "/map-vepfs/yubo/models/Qwen2.5-32B"

    # 检查是否有中间结果文件存在
    import os
    processed_count = 0

    if os.path.exists(output_file):
        # 如果存在临时文件，加载已处理的数据
        with open(output_file, "r") as fi:
            output_data = json.load(fi)
        # print("sta_output_data(output_data)", sta_output_data(output_data))
    else:
        with open(input_file, "r") as fi:
            output_data = json.load(fi)
        output_data = filter_output_data(output_data, start=start_idx, end=end_idx)

    llm, sampling_params = load_vllm_model(model_path)

    process_data = get_process_data(output_data)
    while len(process_data) > 500:
        prompts = []
        for each in process_data:
            question = get_cft_format_question(each)
            prompts.append(get_prompt(question))

        print("len(prompts)", len(prompts))
        # print("prompts[0]", prompts[0])

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
            # batch_sta = {"critique_num": 0, "valid_critique_num": 0}
            # 添加这批次的结果到输出数据
            for i, output in enumerate(batch_output):
                if i == 0:
                    print("output[0]", output)
                question = process_data[batch_start + i]["question"]
                output_data[question]["DeepSeek-R1-Distill-Qwen-32B_critique"] = output
                output_result, output_valid = parse_output(output)
                output_data[question]["DeepSeek-R1-Distill-Qwen-32B_critique_res"] = output_result
                output_data[question]["DeepSeek-R1-Distill-Qwen-32B_critique_valid"] = output_valid

            # 每完成一批次就保存临时文件
            with open(output_file, "w") as fo:
                fo.write(json.dumps(output_data, indent=4))
            print(f"Batch {batch_idx + 1} complete. bs={batch_end - batch_start}. "
                  f"Progress saved. Batch processing time:", time.time() - start_time)

        process_data = get_process_data(output_data)


main()
