import json
import os
import random


def main():
    input_file = "../LLaMA-Factory/data/CriticCoT_t2_critic_data_0115.json"
    output_file_base_path = "../LLaMA-Factory/data/webinstruct_cft_80k_0121_pp"
    with open(input_file, "r", encoding="utf-8") as fi:
        ori_data = json.load(fi)
    input_length_data = sorted(ori_data, key=lambda x: len(x["input"]))
    output_length_data = sorted(ori_data, key=lambda x: len(x["output"]))
    total_length_data = sorted(ori_data, key=lambda x: len(x["input"]) + len(x["output"]))
    data_number = 120000
    edge_window = 1
    output_data = []
    # output_data[0]: input length最长的80k
    # output_data[1]: input length最短的80k
    # output_data[2]: output length最长的80k
    # output_data[3]: output length最短的80k
    # output_data[4]: input + output length平均的80k
    input_length_data = input_length_data[edge_window: -edge_window]
    output_length_data = output_length_data[edge_window: -edge_window]
    output_data.append(input_length_data[-1 * data_number:])
    output_data.append(input_length_data[:data_number])

    output_data.append(output_length_data[-1 * data_number:])
    output_data.append(output_length_data[:data_number])

    output_data.append(total_length_data[240000:360000])

    for i in range(len(output_data)):
        output_path = output_file_base_path + str(i + 1) + ".json"
        print("len(output_data[i])", len(output_data[i]))
        curr = output_data[i]
        print("i", i)
        print("min length", len(curr[0]["input"] + curr[0]["output"]))
        print("max length", len(curr[-1]["input"] + curr[-1]["output"]))
        random.shuffle(curr)
        with open(output_path, "w") as fo:
            fo.write(json.dumps(curr, indent=4))


main()


