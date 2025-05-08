import json
from tqdm import tqdm

import re


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


def integrate_deepmath_data():
    ori_data_path = "../local_data/deepmath_cft_data/deepmath_cft_step_1.json"
    output_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0421.json"
    add_qwen_32b_response_data_path = "../local_data/deepmath_cft_data/deepmath_qwen_32b_gen_solution_step_2.json.temp"
    with open(ori_data_path, "r") as fi:
        ori_data = json.load(fi)
    print("len(ori_data)", len(ori_data))
    map_data = {}
    duplicate_count = 0
    idx = 0
    for i, each in tqdm(enumerate(ori_data)):
        if each["question"] in map_data:
            # print("duplicate question", each["question"])
            duplicate_count += 1
            continue
        each["idx"] = str(idx)
        idx += 1
        map_data[each["question"]] = each
    print("duplicate_count", duplicate_count)
    print("len(map_data)", len(map_data))
    with open(add_qwen_32b_response_data_path, "r") as fi:
        add_qwen_32b_response_data = json.load(fi)
    sta_count = {"qwen-2.5-32b_answer_valid": 0, "qwen-2.5-32b_answer_correct": 0,
                 "qwen-2.5-32b_answer_invalid": 0, "qwen-2.5-32b_answer_incorrect": 0}
    for each in add_qwen_32b_response_data:
        question = each["question"]
        solution = each["solution"]
        if question not in map_data:
            print("question not in map data", question)
            continue

        map_data[question]["qwen-2.5-32b_answer"] = solution
        extracted_answer = extract_boxed_answer(solution)
        if extracted_answer is None:
            map_data[question]["qwen-2.5-32b_answer_valid"] = False
            # map_data[question]["qwen-2.5-32b_answer_correctness"] = False
            sta_count["qwen-2.5-32b_answer_invalid"] += 1
        else:
            map_data[question]["qwen-2.5-32b_answer_valid"] = True
            sta_count["qwen-2.5-32b_answer_valid"] += 1
            if extracted_answer == map_data[question]["gt_answer"]:
                map_data[question]["qwen-2.5-32b_answer_correctness"] = True
                sta_count["qwen-2.5-32b_answer_correct"] += 1
            else:
                map_data[question]["qwen-2.5-32b_answer_correctness"] = False
                sta_count["qwen-2.5-32b_answer_incorrect"] += 1

    print("sta_count:", sta_count)
    with open(output_file, "w") as fo:
        fo.write(json.dumps(map_data))


if __name__ == "__main__":
    integrate_deepmath_data()








