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


def format_correct_only(question, answer):
    ins = "Please analyze and answer the following question step by step:"
    response = answer.replace("####\n", "\nSummary:\n")
    return {"instruction": ins,
            "input": question,
            "output": response}


def load_r1_data():
    input_file = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/webinstruct_2-4k_r1_cft_data_0223.json"
    with open(input_file, "r") as fi:
        data = json.load(fi)
    return data


def main():
    r1_data = load_r1_data()
    input_file = "/map-vepfs/yubo/CriticCoT/local_data/bespoke_data/bespoke_2k_data_0206.json"
    output_file_1 = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/bespoke_2k_data_0223.json"
    output_file_2 = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/bespoke_merge_r1_data_0223.json"
    with open(input_file, "r") as fi:
        data = json.load(fi)
    output_data = []
    for each in data:
        question = each["question"]
        answer = each["answer"]
        output_data.append(format_correct_only(question, answer))
    random.shuffle(output_data)
    with open(output_file_1, "w") as fo:
        fo.write(json.dumps(output_data, indent=2))
    print("output_data number", len(output_data))
    output_data += r1_data
    random.shuffle(output_data)
    with open(output_file_2, "w") as fo:
        fo.write(json.dumps(output_data, indent=2))
    print("output_data number", len(output_data))


main()

