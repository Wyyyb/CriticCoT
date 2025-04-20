import json
import os


def load_data():
    data = []
    with open("../local_data/deepmath_ori_data/deepmath_train.jsonl", "r") as fi:
        for line in fi:
            curr = json.loads(line)
            data.append(curr)
    return data


def format_sft(data):
    output_data = []
    for each in data:
        instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        for i in range(3):
            curr = {"instruction": instruction, "input": each["question"],
                    "output": each[f"r1_solution_{str(i+1)}"]}
            output_data.append(curr)
    return output_data


def prepare_cft_data(data):
    output_data = []
    for each in data:
        instruction_1 = "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        instruction_2 = "You are a science expert. A student is trying to solve a question, please explain briefly whether his answer is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]\n\n"
        instruction_3 = "Please critique whether the following solution to the question is correct.\n\n"
        question = each["question"]
        r1_solution_1 = each["r1_solution_1"]
        r1_solution_2 = each["r1_solution_2"]
        r1_solution_3 = each["r1_solution_3"]
        curr = {"instruction_1": instruction_1, "instruction_2": instruction_2,
                "instruction_3": instruction_3, "question": question,
                "r1_solution_1": r1_solution_1, "r1_solution_2": r1_solution_2,
                "r1_solution_3": r1_solution_3}
        output_data.append(curr)
    return output_data


def main():
    ori_deepmath_train_data = load_data()
    sft_data = format_sft(ori_deepmath_train_data)
    with open("../LLaMA-Factory/data/ori_deepmath_sft_data.json") as fo:
        fo.write(json.dumps(sft_data, indent=2))

    os.makedirs("../local_data/deepmath_cft_data", exist_ok=True)
    deepmath_cft_step_1 = prepare_cft_data(ori_deepmath_train_data)
    with open("../local_data/deepmath_cft_data/deepmath_cft_step_1.json") as fo:
        fo.write(json.dumps(deepmath_cft_step_1, indent=2))


if __name__ == "__main__":
    main()


