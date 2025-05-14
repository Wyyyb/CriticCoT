import json
import os


def data_filter(item, map_info):
    question = item["question"]
    if question != map_info:
        return {}
    critique = item.get("critique", None)
    if not critique:
        return {}
    if "verified_as_wrong" in critique and critique["verified_as_wrong"] is True:
        return {}
    if "</think>" in critique:
        critique = critique.split("</think>")[-1]
    item["critique"] = critique
    return item


def format_single(item, map_info):
    instruction = "Please critique whether the following solution to the question is correct.\n\n"
    item = data_filter(item, map_info)
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


def get_map_info(input_data):
    map_info = {}
    for each in input_data:
        question = each.get("question", None)
        if question not in map_info:
            map_info[question] = {"solution_right": 0, "solution_wrong": 0,
                                  "critique_right": 0, "critique_wrong": 0}
        if each["student_solution"]["judged_correctness"] is True:
            map_info[question]["solution_right"] += 1
        elif each["student_solution"]["judged_correctness"] is False:
            map_info[question]["solution_wrong"] += 1
        if each["critique_judged_correctness"] is True:
            map_info[question]["critique_right"] += 1
        elif each["critique_judged_correctness"] is False:
            map_info[question]["critique_wrong"] += 1
    with open("map_info.json", "w") as f:
        f.write(json.dumps(map_info, indent=4))

    candidates = []
    for k, v in map_info.items():
        if v["solution_right"] > 100 and v["solution_wrong"] > 100:
            candidates.append([k, v["critique_right"]])
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected_question = candidates[2][0]
    return selected_question


def main(input_file_path, output_path):
    input_data = []
    with open(input_file_path, "r") as f:
        input_data = json.load(f)
    map_info = get_map_info(input_data)
    print("selected_question:", map_info)
    output_data = []
    for each in input_data:
        curr = format_single(each, map_info)
        if not curr:
            continue
        output_data.append(curr)
    print("len(output_data)", len(output_data))
    with open(output_path, "w") as f:
        for each in output_data:
            f.write(json.dumps(each) + "\n")


if __name__ == "__main__":
    input_path = "../../local_data/one_shot_data_0513/filtered_critique_data_0513.json"
    output_file_path = "../../local_data/training_data_0513/balance_one-shot_train_data_filtered_38k_0513.jsonl"
    os.makedirs("../../local_data/training_data_0513/", exist_ok=True)
    main(input_path, output_file_path)

