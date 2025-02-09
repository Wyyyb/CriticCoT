import os
import json


def sta_average_length(input_list):
    total_length = 0.0
    for each in input_list:
        total_length += len(each["output"])
    return total_length / len(input_list)


def single_sta(input_path):
    with open(input_path, "r") as fi:
        data = json.load(fi)
    print(input_path, sta_average_length(data))


def main():
    o1_mini_path = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/webinstruct_cft_80k_o1_mini_long_0204.json"
    gpt4o_path = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/webinstruct_cft_80k_0121_p3.json"
    single_sta(o1_mini_path)
    single_sta(gpt4o_path)


main()

