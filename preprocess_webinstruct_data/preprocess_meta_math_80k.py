import json
import os
import random


def process_meta_math_qa(input_path, output_path):
    with open(input_path, "r") as fi:
        ori_data = json.load(fi)
    print("len(ori_data)", len(ori_data))
    random.shuffle(ori_data)
    ori_data = ori_data[:80000]
    output_data = []
    for each in ori_data:
        ins = "Please analyze and answer the following question step by step:"
        question = each["query"]
        output = each["response"]
        curr = {"instruction": ins, "input": question, "output": output}
        output_data.append(curr)
    with open(output_path, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


def main():
    input_path = "/data/yubo/CriticCoT/local_data/MetaMathQA-395K.json"
    output_path = "/data/yubo/CriticCoT/LLaMA-Factory/data/MetaMathQA_sample_80k_data_0118.json"
    process_meta_math_qa(input_path, output_path)


if __name__ == "__main__":
    main()



