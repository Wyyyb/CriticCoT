import json

input_file_path = "../local_data/deepmath_cft_data/deepmath_integrate_data_0421_add_solution_p1.json"

with open(input_file_path, "r") as fi:
    data = json.load(fi)

valid_count = 0
for k, v in data.items():
    if v["qwen-2.5-32b_answer_valid"] is True:
        valid_count += 1

print("valid_count", valid_count)




