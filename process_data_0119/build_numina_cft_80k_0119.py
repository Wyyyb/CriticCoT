import json
import os


def main():
    input_dir = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/on_policy_data_0119/" \
                "qwen_math_numina_80k_add_critique_0119"
    cft_output_file_path = "../LLaMA-Factory/data/numina_cft_80k_0119.json"
    sft_output_file_path = "../LLaMA-Factory/data/numina_sft_80k_0119.json"
    cft_data = []
    sft_data = []
    for each in os.listdir(input_dir):
        if not each.endswith(".json"):
            continue
        file_path = os.path.join(input_dir, each)
        with open(file_path, "r") as fi:
            curr = json.load(fi)
        for item in curr:
            question = item["question"]
            numina_solution = item["numina_solution"]
            answer = item["qwen_2.5_math_answer"]
            if "model_output" not in item or item["model_output"] == "":
                print("model output empty", item)
                continue
            critique = item["model_output"]
            cft_data.append(single_format_cft(question, answer, critique))
            sft_data.append(single_format_sft(question, numina_solution))
    with open(cft_output_file_path, "w") as fo:
        fo.write(json.dumps(cft_data, indent=4))

    with open(sft_output_file_path, "w") as fo:
        fo.write(json.dumps(sft_data, indent=4))


def single_format_cft(question, solution, critique):
    t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
               "input": question + f"\nSolution:\n{solution}\n", "output": critique}
    return t2_curr


def single_format_sft(question, answer):
    return {"instruction": "Please analyze and answer the following question step by step:",
            "input": question, "output": answer}


main()

