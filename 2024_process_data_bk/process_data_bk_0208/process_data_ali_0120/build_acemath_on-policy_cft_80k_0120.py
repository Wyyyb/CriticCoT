import json
import os


def main():
    input_file_path = "/cpfs/data/user/yubowang/CriticCoT/local_data/data_0120/" \
                      "qwen_math_ace_80k_0119_qwen_critique.json"
    cft_output_file_path = "../LLaMA-Factory/data/ace_80k_critique_qwen_on-policy_0120.json"
    cft_data = []

    with open(input_file_path, "r") as fi:
        input_data = json.load(fi)
    for item in input_data:
        question = item["question"]
        qwen_2_5_math_answer = item["qwen_2.5_math_answer"]
        critique = item["qwen_gemini_critique"]
        cft_data.append(single_format_cft(question, qwen_2_5_math_answer, critique))
    print("len(cft_data)", len(cft_data))
    with open(cft_output_file_path, "w") as fo:
        fo.write(json.dumps(cft_data, indent=4))


def single_format_cft(question, solution, critique):
    t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
               "input": question + f"\nSolution:\n{solution}\n", "output": critique}
    return t2_curr


def single_format_sft(question, answer):
    return {"instruction": "Please analyze and answer the following question step by step:",
            "input": question, "output": answer}


main()

