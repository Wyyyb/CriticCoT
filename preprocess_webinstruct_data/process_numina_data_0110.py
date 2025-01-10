import json
from datasets import load_dataset
import random
import os


def load_numina():
    no_box_count = 0
    numina_data = []
    ds = load_dataset("AI-MO/NuminaMath-CoT")
    output_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/NuminaMath-CoT/ori_numina_data.jsonl"
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in ds["train"]:
                json_str = json.dumps(item, ensure_ascii=False)
                f.write(json_str + '\n')
    with open(output_file, "r") as fi:
        for line in fi:
            curr = json.loads(line)
            numina_data.append(curr)
            if "\\boxed{" not in curr["solution"]:
                print("boxed not in data", curr["source"])
                no_box_count += 1
    print("len(numina_data)", len(numina_data))


def main():
    load_numina()


if __name__ == "__main__":
    main()

