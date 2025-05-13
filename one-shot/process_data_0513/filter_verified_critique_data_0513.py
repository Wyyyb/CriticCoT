import json
import os


def process(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    output_data = []
    for each in data:
        if each["critique_judged_correctness"] is True:
            output_data.append(each)
    print("len(output_data)", len(output_data))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    process("../../local_data/one_shot_data_0513/judge_critique_correctness_data_50k_0513_p0.json",
            "../../local_data/one_shot_data_0513/filtered_critique_data_0513.json")








