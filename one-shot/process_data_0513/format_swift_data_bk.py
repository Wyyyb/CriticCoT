import json
import os


def format_single(item):
    instruction = "Please critique whether the following solution to the question is correct.\n\n"
    question = item.get("question", None)
    solution = item.get("student_solution", {}).get("solution", None)
    critique = item.get("critique", None)
    if not critique or not question or not solution:
        return None
    input_str = f"Question:\n{question}\n\n" + f"Solution:\n{solution}\n\n"
    output_str = f"Critique:\n{critique}\n\n"
    result = {"messages":
        [
            # {"role": "system", "content": instruction},
            {"role": "user", "content": instruction + input_str},
            {"role": "assistant", "content": output_str}
        ]
    }
    return result


def main(input_dir, output_path):
    input_data = []
    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(input_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            input_data.append(v)
    output_data = []
    for each in input_data:
        curr = format_single(each)
        if not curr:
            continue
        output_data.append(curr)
    print("len(output_data)", len(output_data))
    with open(output_path, "w") as f:
        for each in output_data:
            f.write(json.dumps(each) + "\n")


if __name__ == "__main__":
    input_dir_path = "../../local_data/one_shot_data_0513/"
    output_file_path = "../../local_data/training_data_0513/full_one-shot_train_data_0513.jsonl"
    os.makedirs("../../local_data/training_data_0513/", exist_ok=True)
    main(input_dir_path, output_file_path)

