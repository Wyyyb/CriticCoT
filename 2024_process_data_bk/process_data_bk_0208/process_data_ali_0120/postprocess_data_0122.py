import json
import os
import random


def main():
    input_file_1 = "../LLaMA-Factory/data/webinstruct_cft_80k_0121_p3.json"
    input_file_2 = "../../LLaMA-Factory/data/test_sample_0122.json"
    output_file = "../LLaMA-Factory/data/webinstruct_cft_80k_0121_p3_v2.json"
    with open(input_file_1, "r") as fi:
        data_1 = json.load(fi)

    with open(input_file_2, "r") as fi:
        data_2 = json.load(fi)
    length = len(data_2 * 20)
    data = data_1[:-1*length] + data_2 * 20
    random.shuffle(data)
    print("len(data)", len(data))
    with open(output_file, "w") as fo:
        fo.write(json.dumps(data, indent=4))


main()

