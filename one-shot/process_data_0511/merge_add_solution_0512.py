import json
import os


def merge(input_path_1, input_path_2, output_path):
    count = 0
    add_keys = ["MiMo-7B-SFT", "Qwen3-8B"]
    with open(input_path_1, "r") as f1, open(input_path_2, "r") as f2:
        input_data_1 = json.load(f1)
        input_data_2 = json.load(f2)
    for k, v in input_data_1.items():
        if k not in input_data_2:
            print("skip {}".format(k))
            continue
        comp_value = input_data_2[k]
        for each_key in add_keys:
            if each_key not in v["student_solutions"] and each_key in comp_value["student_solutions"]:
                input_data_1[k]["student_solutions"][each_key] = comp_value["student_solutions"][each_key]
                count += len(comp_value["student_solutions"][each_key])

    with open(output_path, "w") as fo:
        fo.write(json.dumps(input_data_1, indent=4))
    print("new added count", count)


if __name__ == "__main__":
    path_1 = "/data/yubo/CriticCoT/local_data/one_shot_data_0511/seed_questions_add_solution_0512.json"
    path_2 = "/data/yubo/CriticCoT/local_data/one_shot_data_0511/seed_questions_add_solution_0512_p1.json"
    output_path_1 = "/data/yubo/CriticCoT/local_data/one_shot_data_0511/seed_questions_add_solution_0512_merged.json"
    merge(path_1, path_2, output_path_1)

