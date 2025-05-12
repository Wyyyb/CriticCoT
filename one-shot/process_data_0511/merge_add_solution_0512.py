import json
import os


def merge(input_path_1, input_path_2, output_path):
    add_keys = ["MiMo-7B-SFT"]
    with open(input_path_1, "r") as f1, open(input_path_2, "r") as f2:
        input_data_1 = json.load(f1)
        input_data_2 = json.load(f2)
    for k, v in input_data_1.items():
        if k not in input_data_2:
            print("skip {}".format(k))
            continue


