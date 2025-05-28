import json
import os


def data_filter(item):
    critique = item.get("critique", None)
    if not critique:
        return {}
    # if "verified_as_wrong" in critique and critique["verified_as_wrong"] is True:
    #     return {}
    if "</think>" in critique:
        critique = critique.split("</think>")[-1]
    item["critique"] = critique
    return item


def format_single(item):
    instruction = "Please critique whether the following solution to the question is correct.\n\n"
    # item = data_filter(item)
    if not item:
        return None
    question = item.get("question", None)
    solution = item.get("solution", None)
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


def get_groups(input_data):
    groups = {}
    for each in input_data:
        if each["source"] not in groups:
            groups[each["source"]] = []
        groups[each["source"]].append(each)
    output_groups = []
    for k, v in groups.items():
        output_groups.append(v)
    # return output_groups
    return groups


def main(input_dir_path, output_path):
    input_data = []
    for file in os.listdir(input_dir_path):
        if file.endswith(".json"):
            with open(os.path.join(input_dir_path, file), "r") as f:
                for k, v in json.load(f).items():
                    input_data.append(v)
    groups = get_groups(input_data)
    for k, group in groups.items():
        output_data = []
        for each in group:
            curr = format_single(each)
            if not curr:
                continue
            output_data.append(curr)
        print("len(output_data)", k, len(output_data))
        with open(output_path.replace(".jsonl", f"-{k.replace(".json", "")}.jsonl"), "w") as f:
            for each in output_data:
                f.write(json.dumps(each) + "\n")


if __name__ == "__main__":
    input_path = "../local_data/bbeh_data_0526/call_api_result/"
    output_file_path = "../local_data/training_data_0528/bbeh_one-shot_train_data_0528.jsonl"
    os.makedirs("../local_data/training_data_0528/", exist_ok=True)
    main(input_path, output_file_path)

