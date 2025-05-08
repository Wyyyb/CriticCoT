import os
import json


def format_single(item):
    prompt = f"Please reason step by step to find a solution to the following " \
             f"question, and put your final answer within \\boxed{{}}."
    return {
        "instruction": prompt,
        "input": item["question"],
        "output": "<think>" + item["qwen3-32b_answer"]
    }


def format_all_data(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    output_data = []
    for k, v in data.items():
        if not v.get("qwen3-32b_answer_valid", None):
            continue
        curr = format_single(v)
        output_data.append(curr)
    print("len(output_data)", len(output_data))
    with open(output_path, "w") as f:
        f.write(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    #format_all_data("../local_data/cft_data_0506/webinstruct_data_add_solution_0506.json",
    #                "../local_data/cft_data_0506/webinstruct_qwen3_32b_sft_data_p3334.json")
    format_all_data("../local_data/cft_data_0506/webinstruct_data_add_solution_0506_2222.json",
                    "../local_data/cft_data_0506/webinstruct_qwen3_32b_sft_data_p2222.json")



