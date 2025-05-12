import json
import random


def get_deepmath_seeds(input_path):
    with open(input_path) as f:
        data = json.load(f)
    right_candidates = []
    wrong_candidates = []
    for k, v in data.items():
        if "qwen-2.5-32b_answer_correctness" not in v:
            continue
        v["data_source"] = "deepmath"
        if v["qwen-2.5-32b_answer_correctness"] is True:
            right_candidates.append(v)
        elif v["qwen-2.5-32b_answer_correctness"] is False:
            wrong_candidates.append(v)
    random.shuffle(right_candidates)
    random.shuffle(wrong_candidates)
    print("len(right_candidates), len(wrong_candidates)", len(right_candidates), len(wrong_candidates))
    return right_candidates[:20], wrong_candidates[:20]


def get_webinstruct_v_seeds(input_path):
    with open(input_path) as f:
        data = json.load(f)
    right_candidates = []
    wrong_candidates = []
    for k, v in data.items():
        if "qwen3-32b_extracted_answer" not in v:
            continue
        v["data_source"] = "webinstruct_verified"
        if v["qwen3-32b_extracted_answer"] == v["gt_answer"]:
            right_candidates.append(v)
        elif v["qwen3-32b_extracted_answer"] != v["gt_answer"]:
            wrong_candidates.append(v)
    random.shuffle(right_candidates)
    random.shuffle(wrong_candidates)
    print("len(right_candidates), len(wrong_candidates)", len(right_candidates), len(wrong_candidates))
    return right_candidates[:20], wrong_candidates[:20]


def main():
    deepmath_input_path = "/data/yubo/CriticCoT/local_data/cft_data_0506/deepmath_integrate_data_0428_add_solution.json"
    webinstruct_v_input_path = "/data/yubo/CriticCoT/local_data/cft_data_0506/webinstruct_data_add_solution_0506_ids_merged.json"
    output_path = "/data/yubo/CriticCoT/local_data/one_shot_data_0511/seed_questions_0511.json"
    output_data = []
    r, w = get_deepmath_seeds(deepmath_input_path)
    output_data.extend(r)
    output_data.extend(w)
    r, w = get_webinstruct_v_seeds(webinstruct_v_input_path)
    output_data.extend(r)
    output_data.extend(w)
    with open(output_path, "w") as f:
        f.write(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    main()

