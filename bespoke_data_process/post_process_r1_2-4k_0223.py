import json
import os
import random


def format_critic_data(question, answer, critique_answer):
    question = "Question:\n" + question
    solution = answer
    critique = "Critique:\n" + critique_answer
    t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
               "input": question + f"\nSolution:\n{solution}\n", "output": critique}
    return t2_curr


def main_1():
    input_file = "/map-vepfs/yubo/CriticCoT/local_data/webinstruct_0207/add_critique_webinstruct_2k_data_0207_merged.json"
    output_file = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/webinstruct_2-4k_r1_cft_data_0223.json"
    with open(input_file, "r") as fi:
        data = json.load(fi)
    output_data = []
    for each in data:
        question = each["question"]
        answer = each["answer"]
        critique = each["critique"]
        output_data.append(format_critic_data(question, answer, critique))
    random.shuffle(output_data)
    print("output_data number", len(output_data))
    with open(output_file, "w") as fo:
        fo.write(json.dumps(output_data, indent=2))
    return output_data


web_data = main_1()



