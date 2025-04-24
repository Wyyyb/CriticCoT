import os
import json


answer_file_path = "../local_data/deepmath_cft_data/deepmath_integrate_data_0421_add_solution_p1.json"
critique_file_path = "../local_data/deepmath_cft_data/deepmath_integrate_data_0422_add_critique_p1.json"
output_file_path = "../local_data/deepmath_cft_data/deepmath_integrate_data_0423_add_critique_p1.json"


with open(answer_file_path, "r") as fi:
    answer_file = json.load(fi)


with open(critique_file_path, "r") as fi:
    critique_file = json.load(fi)

update_answer_count = 0
for k, v in answer_file.items():
    if k not in critique_file:
        print("answer_file key not in critique_file")
        continue
    if "qwen-2.5-32b_answer" in v and v.get("qwen-2.5-32b_answer_valid") is True:
        if "qwen-2.5-32b_answer" in critique_file[k] and \
                critique_file[k]["qwen-2.5-32b_answer"] == v["qwen-2.5-32b_answer"]:
            continue
        update_answer_count += 1
        critique_file[k]["qwen-2.5-32b_answer"] = v["qwen-2.5-32b_answer"]
        critique_file[k]["qwen-2.5-32b_answer_valid"] = True
        critique_file[k]["qwen-2.5-32b_answer_correctness"] = v["qwen-2.5-32b_answer_correctness"]

print("update_answer_count", update_answer_count)
with open(output_file_path, "w") as fo:
    fo.write(json.dumps(critique_file, indent=4))
















