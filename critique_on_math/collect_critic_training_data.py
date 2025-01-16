import os
import json
import random
from copy import deepcopy


def single_collect(file_path):
    collect_data = {}
    with open(file_path, 'r', encoding="utf-8") as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            real_idx = curr["real_idx"]
            if real_idx not in collect_data:
                collect_data[real_idx] = []
            collect_data[real_idx].append(curr)
    return collect_data


def merge_critic_result(ori_res, new_res):
    for k, v in ori_res.items():
        ori_res[k] += new_res[k]
    return ori_res


def format_training_data(collect_res, output_path):
    critique_right_map = {}
    critique_wrong_map = {}
    sta_right_map = {}
    sta_wrong_map = {}
    for real_idx, curr in collect_res.items():
        # critique as right or wrong, both of the two critique are right
        critique_right = []
        critique_wrong = []
        for each in curr:
            score = each["score"][0]
            critique_score = each["critique_score"]
            if not critique_score:
                continue
            pred = each["pred"][0]
            gt = each["gt"]
            if len(gt) < 20 < len(pred):
                continue
            if score:
                critique_right.append(each)
            else:
                critique_wrong.append(each)
        random.shuffle(critique_right)
        random.shuffle(critique_wrong)
        critique_right_map[real_idx] = critique_right[:4]
        critique_wrong_map[real_idx] = critique_wrong[:4]
        if len(critique_right_map[real_idx]) not in sta_right_map:
            sta_right_map[len(critique_right_map[real_idx])] = 1
        else:
            sta_right_map[len(critique_right_map[real_idx])] += 1
        if len(critique_wrong_map[real_idx]) not in sta_wrong_map:
            sta_wrong_map[len(critique_wrong_map[real_idx])] = 1
        else:
            sta_wrong_map[len(critique_wrong_map[real_idx])] += 1
    print("sta_right_map", sta_right_map)
    print("sta_wrong_map", sta_wrong_map)

    training_data = []
    for k, v in critique_right_map.items():
        for each in v:
            question = each["question"]
            solution = each["code"][0]
            critique = each["critique_output"]
            curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
                    "input": question + f"\nSolution:\n{solution}\n", "output": critique}
            training_data.append(curr)
    print("critique_right data num: ", len(training_data))
    negative_data = []
    for k, v in critique_wrong_map.items():
        for each in v:
            question = each["question"]
            solution = each["code"][0]
            critique = each["critique_output"]
            curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
                    "input": question + f"\nSolution:\n{solution}\n", "output": critique}
            training_data.append(curr)
            negative_data.append(curr)
    random.shuffle(training_data)
    print("critique_wrong data num: ", len(negative_data))
    print("critique total data num: ", len(training_data))
    with open(output_path, "w") as fo:
        fo.write(json.dumps(training_data, indent=4))
    with open(output_path.replace(".json", "_negative_samples.json"), "w") as fo:
        fo.write(json.dumps(negative_data, indent=4))


def main():
    input_dir = "/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_multi_eval_result_0116/" \
                "qwen_eval_res_0116_multi_result"
    output_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_self_critique_MATH-TRAIN-8_data_0116.json"
    collect_res = None
    for sub_dir in os.listdir(input_dir):
        sub_dir_path = os.path.join(input_dir, sub_dir, "math_train")
        for each in os.listdir(sub_dir_path):
            if each.endswith("_add_critique.jsonl"):
                file_path = os.path.join(sub_dir_path, each)
                if collect_res:
                    collect_res = merge_critic_result(collect_res, single_collect(file_path))
                else:
                    collect_res = single_collect(file_path)
    format_training_data(collect_res, output_path)


if __name__ == '__main__':
    main()







