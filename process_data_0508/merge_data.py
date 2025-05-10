import os
import json


def merge(input_path_1, input_path_2, output_path):
    with open(input_path_1, "r") as f:
        data_1 = json.load(f)
    with open(input_path_2, "r") as f:
        data_2 = json.load(f)
    print("len(data_1)", len(data_1))
    print("len(data_2)", len(data_2))
    output_data = {}
    for k, v in data_1.items():
        if k not in data_2:
            print("{} not in data_2".format(k))
            continue
        if v.get("qwen3-32b_answer_valid", False) is True:
            output_data[k] = v
        elif data_2[k].get("qwen3-32b_answer_valid", False) is True:
            output_data[k] = data_2[k]
    print("len(output_data)", len(output_data))
    with open(output_path, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    merge("/data/yubo/CriticCoT/local_data/cft_data_0506/webinstruct_data_add_solution_0506_2222.json",
          "/data/yubo/CriticCoT/local_data/cft_data_0506/webinstruct_data_add_solution_0506_3334.json",
          "/data/yubo/CriticCoT/local_data/cft_data_0506/webinstruct_data_add_solution_0506_merged.json")


