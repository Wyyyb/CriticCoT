import os
import json


def main():
    output_file_path = "2_add_solution_data/add_solution_0526.json"
    input_dir_path = "1_seeds_data/"
    output_data = []
    for file in os.listdir(input_dir_path):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(input_dir_path, file), "r") as f:
            data = json.load(f)
            curr = {"question": data["input"].replace("Question: ", ""),
                    "source": file,
                    "gt_answer": data["target"]}
            output_data.append(curr)
    with open(output_file_path, "w") as f:
        f.write(json.dumps(output_data))


if __name__ == "__main__":
    main()

