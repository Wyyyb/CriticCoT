import json
from copy import deepcopy


def remove_thinking(solution):
    if "</think>" in solution:
        solution = solution.split("</think>")[-1].strip()
    return solution


def process(input_path, output_path):
    with open(input_path, "r") as fi:
        data = json.load(fi)
    critique_id = 0
    output_data = {}
    sta_map = {}
    count = 0
    for k, v in data.items():
        for each_k, each_v in v["student_solutions"].items():
            for solution_id, solution in enumerate(each_v):
                short = remove_thinking(solution["solution"])
                if 30000 > len(short) > 10000:
                    print("len(short)", len(short))
                    count += 1
                curr = deepcopy(v)
                curr.pop("student_solutions")
                curr["solution_id"] = solution_id
                curr["student_model"] = each_k
                if each_k not in sta_map:
                    sta_map[each_k] = 0
                sta_map[each_k] += 1
                curr["student_solution"] = solution
                curr["critique_id"] = str(critique_id)
                critique_id += 1
                output_data[curr["critique_id"]] = curr
    print("len(output_data)", len(output_data))
    print("sta_map", sta_map)
    print("count", count)
    with open(output_path, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    process("../../local_data/one_shot_data_0511/seed_questions_add_solution_0512.json",
            "../../local_data/one_shot_data_0511/one_shot_data_with_solution_0512.json")








