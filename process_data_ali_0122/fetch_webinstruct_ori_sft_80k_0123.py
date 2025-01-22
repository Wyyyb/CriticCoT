import json
import os


def main():
    input_path = "../LLaMA-Factory/data/webinstruct_cft_80k_0119.json"
    output_path = "../LLaMA-Factory/data/webinstruct_ori_sft_80k_0123.json"
    with open(input_path, "r") as fi:
        cft_data = json.load(fi)
    ori_sft_data = []
    for each in cft_data:
        segs = each["input"].split("\nSolution:\n")
        question = segs[0]
        question_length = len(question)
        answer = each["input"][question_length:]
        instruction = "Please reason step by step to solve the following question."
        ori_sft_data.append({"instruction": instruction, "input": question, "output": answer})

    with open(output_path, "w") as fo:
        fo.write(json.dumps(ori_sft_data, indent=4))


main()

