import json
import os


def add_ids(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    count = 0
    for k, v in data.items():
        data[k]["ori_id"] = data["id"]
        data[k]["id"] = str(count)
        count += 1
    print("count", count)
    with open(output_path, "w") as f:
        f.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    add_ids("../local_data/cft_data_0506/webinstruct_data_add_solution_0506_merged.json",
           "../local_data/cft_data_0506/webinstruct_data_add_solution_0506_ids_merged.json")

