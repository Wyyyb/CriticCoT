import json
import os
import random


def sample_80k_numina(ori_numina_path, output_path):
    with open(ori_numina_path, "r") as fi:
        ori_numina_data = json.load(fi)
    random.shuffle(ori_numina_data)
    output_data = ori_numina_data[:80000]
    print("len(output_data)", len(output_data))
    with open(output_path, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


def main():
    input_path = "/data/yubo/CriticCoT/LLaMA-Factory/data/numina_860k_data_0110.json"
    output_path = "/data/yubo/CriticCoT/LLaMA-Factory/data/numina_sample_80k_data_0118.json"
    sample_80k_numina(input_path, output_path)


if __name__ == "__main__":
    main()



