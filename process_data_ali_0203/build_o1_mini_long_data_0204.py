import json
import os


def main():
    input_file = "/Users/yubowang/Downloads/critique_added_output_data_o1-mini_20250203.json"
    output_file = "../local_data/webinstruct_cft_80k_o1_mini_long_0204/webinstruct_cft_80k_o1_mini_long_0204.json"
    with open(input_file, "r") as fi:
        data = json.load(fi)
    output_data = []
    total_length = 0.0
    for each in data:
        question = "Question:\n" + each["question"]
        answer = each["answer"]
        critique = each["model_output"]
        output_data.append(single_format_cft_data(question, answer, critique))
        total_length += len(critique)
    print("len(output_data)", len(output_data))
    print("average_length", total_length / len(output_data))
    with open(output_file, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


def single_format_cft_data(question, answer, critique):
    t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
               "input": question + f"\nSolution:\n{answer}\n", "output": critique}
    return t2_curr


main()
