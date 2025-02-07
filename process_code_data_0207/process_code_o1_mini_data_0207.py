import json
import os


def load_data(input_path):
    with open(input_path, "r") as fi:
        data = json.load(fi)
    print("len(data)", len(data))
    # idx_map = {}
    # for each in data:
    #     idx = each["idx"]
    #     if idx not in idx_map:
    #         idx_map[idx] = each
    #     else:
    #         print("curr", each)
    #         print("\n\nprev", idx_map[idx])
    return data


def main():
    input_path = "../local_data/cft_code_data_0206/code_o1_mini.json"
    data = load_data(input_path)
    print(data[0])
    output_data = []
    for each in data:
        output_data.append(single_format_cft_data(each["question"], each["answer"], each["model_output"]))
    output_path = "../local_data/cft_code_data_0206/opc-sft-stage2_o1_mini_cft_data_0207.json"
    with open(output_path, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


def single_format_cft_data(question, answer, critique):
    t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
               "input": f"Question:\n{question}\n\n" + f"Solution:\n{answer}\n\n", "output": f"Critique:\n{critique}"}
    return t2_curr


main()

