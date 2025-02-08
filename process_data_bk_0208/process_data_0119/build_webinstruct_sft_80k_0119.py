import json
import os
from tqdm import tqdm


def main():
    input_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/webinstruct_sft_gpt4o_80k_0119_data/" \
                 "webinstruct_sft_gpt4o_80k_0119_data.jsonl"
    output_file = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/webinstruct_sft_gpt4o_80k_0119.json"

    output_data = []
    instruction = "Please analyze and answer the following question step by step:"
    with open(input_file, "r") as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            output_data.append({"instruction": instruction, "input": curr["question"], "output": curr["gpt4o_answer"]})
    with open(output_file, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


main()





