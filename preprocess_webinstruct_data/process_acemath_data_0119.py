import json
from datasets import load_dataset
import random
import os
from tqdm import tqdm


def load_numina():
    no_box_count = 0
    numina_data = []
    ds = load_dataset("nvidia/AceMath-Instruct-Training-Data")
    output_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/AceMath-Instruct-Training-Data/math_sft.jsonl"
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in ds["math_sft"]:
                json_str = json.dumps(item, ensure_ascii=False)
                f.write(json_str + '\n')
    with open(output_file, "r") as fi:
        for line in tqdm(fi):
            curr = json.loads(line)
            numina_data.append(curr)
            if "\\boxed{" not in curr["answer"]:
                # print("boxed not in data", curr["source"])
                no_box_count += 1
    print("no_box_count", no_box_count)
    print("len(numina_data)", len(numina_data))
    return numina_data


load_numina()

