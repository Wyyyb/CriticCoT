from vllm import LLM, SamplingParams
from typing import List
import os
import json
import re
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
            max_tokens=16384,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in model_path.lower()
                else None
            ),
            top_p=0.95,
            top_k=20
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
    prompt = f"<|im_start|>user\nYou are a mathematics expert. Analyze if the student's solution is correct. " \
             f"Follow these steps:\n" \
             f"1. Identify the key mathematical concepts and correct approach.\n" \
             f"2. Check each step of the student's solution.\n" \
             f"3. If incorrect, point out errors and provide the correct solution, " \
             f"putting your final answer within \\boxed{{}}.\n" \
             f"Conclude with \"Conclusion: right/wrong [END]\"\n\n{question}\n<|im_end|>\n" \
             f"<|im_start|>assistant\n<think>\nMathematical Analysis:\n"
    return prompt


def get_process_data(output_data):
    process_data = []
    sta_count = {"DeepSeek-R1-Distill-Qwen-32B_critique": 0,
                 "DeepSeek-R1-Distill-Qwen-32B_critique_valid": 0,
                 "qwen-2.5-32b_answer_valid": 0}
    for k, v in output_data.items():
        if "qwen-2.5-32b_answer" in v and v.get("qwen-2.5-32b_answer_valid") is True:
            sta_count["qwen-2.5-32b_answer_valid"] += 1
            if "DeepSeek-R1-Distill-Qwen-32B_critique" in v:
                sta_count["DeepSeek-R1-Distill-Qwen-32B_critique"] += 1
            if "DeepSeek-R1-Distill-Qwen-32B_critique" in v and \
                    v.get("DeepSeek-R1-Distill-Qwen-32B_critique_valid", False) is True:
                sta_count["DeepSeek-R1-Distill-Qwen-32B_critique_valid"] += 1
                continue
            if len(v["qwen-2.5-32b_answer"]) > 50000:
                print("len(v[qwen-2.5-32b_answer]) > 50000", len(v["qwen-2.5-32b_answer"]))
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


def extract_boxed_answer(text):
    """
    从输入的字符串中提取\boxed{ANSWER}中的ANSWER部分。
    如果找不到\boxed{}的模式，返回None。
    参数:
        text (str): 输入的字符串
    返回:
        str 或 None: 提取到的答案字符串，如果没有找到则返回None
    """
    if len(text) > 50000:
        print("too long text", len(text))
        return None
    pattern = r'\\boxed\{(.*?)\}'
    match = re.search(pattern, text)

    if match:
        return match.group(1)
    else:
        return None


def extract_con(critique):
    # if "Conclusion" not in text:
    #     print("Conclusion not found", text)
    #     return None
    segs = critique.split("Conclusion")
    res_seg = segs[-1].lower()
    if "wrong" in res_seg or "incorrect" in res_seg:
        return False
    elif "right" in res_seg or "correct" in res_seg:
        return True
    else:
        # print("*******************recognize failed\n", res_seg, text)
        return None


def parse_critique(item, critique):
    critique_answer = extract_boxed_answer(critique)
    if critique_answer == item.get("gt_answer", None) and critique is not None:
        critique_valid = True
    else:
        critique_valid = False
    con = extract_con(critique)
    if con is None:
        return critique_answer, None, critique_valid
    if con != item.get("qwen-2.5-32b_answer_correctness"):
        return critique_answer, con, False
    if con is True:
        return critique_answer, con, True
    elif con is False:
        return critique_answer, con, critique_valid
    return None, None, False


def main():
    input_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0421_add_solution_p1.json"
    output_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0428_add_critique_p1.json"
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

    for k, v in output_data.items():
        discard_keys = ["instruction_1", "instruction_2", "instruction_3",
                        "r1_solution_1", "r1_solution_2", "r1_solution_3"]
        for each in discard_keys:
            if each in v:
                output_data[k].pop(each)

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
                critique_answer, critique_conclusion, critique_valid = \
                    parse_critique(output_data[question], output)
                # output_result, output_valid = parse_output(output)
                output_data[question]["DeepSeek-R1-Distill-Qwen-32B_critique_answer"] = critique_answer
                output_data[question]["DeepSeek-R1-Distill-Qwen-32B_critique_conclusion"] = critique_conclusion
                output_data[question]["DeepSeek-R1-Distill-Qwen-32B_critique_valid"] = critique_valid

            # 每完成一批次就保存临时文件
            with open(output_file, "w") as fo:
                fo.write(json.dumps(output_data, indent=4))
            print(f"Batch {batch_idx + 1} complete. bs={batch_end - batch_start}. "
                  f"Progress saved. Batch processing time:", time.time() - start_time)

        process_data = get_process_data(output_data)


main()
