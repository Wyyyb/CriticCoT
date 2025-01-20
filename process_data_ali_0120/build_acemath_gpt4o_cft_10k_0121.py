import json
import os
import random


def main():
    input_dir = "/cpfs/data/user/yubowang/CriticCoT/local_data/ace_10k_cft_0121"
    cft_output_file_path = "../LLaMA-Factory/data/ace_10k_gpt4o_cft_0121.json"
    cft_1_output_file_path = "../LLaMA-Factory/data/webinstruct_ace_80k_gpt4o_cft_0121.json"
    cft_data = []
    sft_data = []
    for each in os.listdir(input_dir):
        if not each.endswith(".json"):
            continue
        file_path = os.path.join(input_dir, each)
        with open(file_path, "r") as fi:
            curr = json.load(fi)
        for item in curr:
            question = "Question:\n" + item["input"]
            solution = item["output"]
            if "model_output" not in item or item["model_output"] == "":
                print("model output empty", item)
                continue
            critique = item["model_output"]
            cft_data.append(single_format_cft(question, solution, critique))
    print("len(cft_data)", len(cft_data))
    with open(cft_output_file_path, "w") as fo:
        fo.write(json.dumps(cft_data, indent=4))
    wi_data_path = "/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/data/webinstruct_cft_80k_0119.json"
    with open(wi_data_path, "r", encoding="utf-8") as fi:
        wi_data = json.load(fi)
    cft_1_data = wi_data[:70000] + cft_data
    random.shuffle(cft_1_data)
    print("len(cft_1_data)", len(cft_1_data))
    with open(cft_1_output_file_path, "w") as fo:
        fo.write(json.dumps(cft_1_data, indent=4))


def single_format_cft(question, solution, critique):
    t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
               "input": question + f"\nSolution:\n{solution}\n", "output": critique}
    return t2_curr


def single_format_sft(question, answer):
    return {"instruction": "Please analyze and answer the following question step by step:",
            "input": question, "output": answer}


main()

