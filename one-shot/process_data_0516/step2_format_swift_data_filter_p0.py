import json
import os


def data_filter(item):
    critique = item.get("critique", None)
    if "</think>" not in critique:
        return {}
    if not critique:
        return {}
    if "verified_as_wrong" in critique and critique["verified_as_wrong"] is True:
        return {}
    # if "</think>" in critique:
    #     critique = critique.split("</think>")[-1]
    item["critique"] = critique
    return item


def format_single(item):
    instruction = "Please critique whether the following solution to the question is correct.\n\n"
    item = data_filter(item)
    if not item:
        return None
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


def get_groups(input_data):
    groups = {}
    for each in input_data:
        if "The pressure" not in each["question"]:
            continue
        if each["question"] not in groups:
            groups[each["question"]] = []
        groups[each["question"]].append(each)
    output_groups = []
    for k, v in groups.items():
        output_groups.append(v)
    return output_groups


def main(input_file_path, output_path):
    input_data = []
    with open(input_file_path, "r") as f:
        input_data = json.load(f)
    groups = get_groups(input_data)
    for i, group in enumerate(groups):
        output_data = []
        for each in group:
            curr = format_single(each)
            if not curr:
                continue
            output_data.append(curr)
        print("len(output_data)", len(output_data))
        with open(output_path.replace(".jsonl", f"_p{i}.jsonl"), "w") as f:
            for each in output_data:
                f.write(json.dumps(each) + "\n")


if __name__ == "__main__":
    input_path = "../../local_data/1-shot-dsr-sft_data_0516/merged_1-shot_dsr_sft_data_0516.json"
    output_file_path = "../../SFT-baseline-0516/train_sft_0516/1-shot_dsr_sft_data_p0_0516.json"
    os.makedirs("../../SFT-baseline-0516/train_sft_0516/", exist_ok=True)
    main(input_path, output_file_path)

