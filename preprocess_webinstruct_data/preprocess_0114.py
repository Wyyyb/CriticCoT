import json
import os
import random


def main():
    input_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_critic_data_1231.json"
    output_1_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_t1_critic_data_0115.json"
    output_2_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_t2_critic_data_0115.json"
    output_3_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_merged_critic_data_0115.json"
    with open(input_path, "r") as fi:
        ori_data = json.load(fi)
    ins = "Please analyze and evaluate the following solution step by step:\n\n"
    new_ins = "Please reason through the following question step by step to find a solution, " \
              "then carefully critique whether your solution is correct.\n\n"
    t1_res = []
    t2_res = []
    for each in ori_data:
        question = each["instruction"].replace(ins, "") + "\n"
        solution = each["input"].replace("\nStudent's Solution:\n", "")
        critic = each["output"]
        t1_curr = {"instruction": new_ins, "input": question, "output": f"Solution:\n{solution}\n{critic}"}
        t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
                   "input": question + f"\nSolution:\n{solution}\n", "output": critic}
        t1_res.append(t1_curr)
        t2_res.append(t2_curr)
    merged_res = t1_res + t2_res
    random.shuffle(t1_res)
    random.shuffle(t2_res)
    random.shuffle(merged_res)
    with open(output_1_path, "w") as fo:
        fo.write(json.dumps(t1_res, indent=4))
    with open(output_2_path, "w") as fo:
        fo.write(json.dumps(t2_res, indent=4))
    with open(output_3_path, "w") as fo:
        fo.write(json.dumps(merged_res, indent=4))
    print("len(t1_res)", len(t1_res))
    print("len(t2_res)", len(t2_res))
    print("len(merged_res)", len(merged_res))
    print("t1_res example:\n", t1_res[0], t1_res[1])
    print("t2_res example:\n", t2_res[0], t2_res[1])


main()

