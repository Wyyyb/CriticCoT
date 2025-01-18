import json
from datasets import load_dataset
import random
import os
from tqdm import tqdm


def load_ace_data():
    no_box_count = 0
    ace_data = []
    input_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/AceMath-Instruct-Training-Data/math_sft.jsonl"
    output_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/ace_data_0119.jsonl"
    with open(input_file, "r") as fi:
        for line in tqdm(fi):
            idx = 0
            curr = json.loads(line)
            if "\\boxed{" not in curr["answer"]:
                # print("boxed not in data", curr["source"])
                no_box_count += 1
                continue
            question = curr["messages"][0]["content"]
            if len(curr["message"]) > 1:
                print(curr["message"])
                continue
            ace_math_solution = curr["answer"]
            ace_data.append({"idx": idx, "question": question, "ace_math_solution": ace_math_solution})
            idx += 1

    print("no_box_count", no_box_count)
    print("len(ace_data)", len(ace_data))
    with open(output_file, "w") as fo:
        for each in ace_data:
            fo.write(json.dumps(each) + "\n")
    # return ace_data


load_ace_data()


