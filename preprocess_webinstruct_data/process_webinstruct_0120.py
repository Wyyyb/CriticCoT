import os
import json
import random

from tqdm import tqdm


def load_jsonl(data_dir):
    data = []
    # critic_id = 0
    for file in os.listdir(data_dir):
        if not file.endswith(".jsonl") or "process" in file:
            continue
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                # curr["critic_id"] = str(critic_id)
                # critic_id += 1
                data.append(curr)
    return data


def parse_single_critic(text):
    segs = text.split("Conclusion")
    res_seg = segs[-1].lower()
    if "wrong" in res_seg or "incorrect" in res_seg:
        return False
    elif "right" in res_seg or "correct" in res_seg:
        return True
    else:
        # print("*******************recognize failed\n", res_seg, text)
        return False


def transfer_single(item):
    question = item.get("question", "")
    answer = item.get("answer", "")
    index = item.get("index", "")
    gpt4o_answer = item.get("gpt4o_answer", "")

    return format_critic_data(question, answer, gpt4o_answer)


def format_critic_data(question, answer, gpt4o_answer):
    question = "Question:\n" + question
    solution = answer
    critique = "Critique:\n" + gpt4o_answer
    t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
               "input": question + f"\nSolution:\n{solution}\n", "output": critique}
    return t2_curr


def main():
    input_data_dir = "/home/wenhuche/WebInstruct/gpt4-filtered"
    input_data = load_jsonl(input_data_dir)

    correct_only_data = []
    incorrect_only_data = []
    unsure_data = []
    for each in tqdm(input_data):
        gpt4o_answer = each.get("gpt4o_answer", "")
        if not gpt4o_answer:
            continue
        critic_conclusion = parse_single_critic(gpt4o_answer)
        if critic_conclusion is True:
            correct_only_data.append(transfer_single(each))
        elif critic_conclusion is False:
            incorrect_only_data.append(transfer_single(each))
        else:
            unsure_data.append(transfer_single(each))
    random.shuffle(unsure_data)
    random.shuffle(correct_only_data)
    random.shuffle(incorrect_only_data)
    print("len(correct_only_data)", len(correct_only_data))
    print("len(incorrect_only_data)", len(incorrect_only_data))
    print("len(unsure_data)", len(unsure_data))

    # os.makedirs("../local_data/WebInstructData_0119/", exist_ok=True)
    # with open("../local_data/WebInstructData_0119/correct_only_40k.json", "w") as fo:
    #     fo.write(json.dumps(correct_only_data[:40000], indent=4))
    # with open("../local_data/WebInstructData_0119/incorrect_only_40k.json", "w") as fo:
    #     fo.write(json.dumps(incorrect_only_data[:40000], indent=4))
    # merged_data = correct_only_data[:40000] + incorrect_only_data[:40000]
    # random.shuffle(merged_data)
    # with open("../local_data/WebInstructData_0119/cft_merged_80k.json", "w") as fo:
    #     fo.write(json.dumps(merged_data, indent=4))
    with open("../LLaMA-Factory/data/webinstruct_critique_right_80k_0119.json", "w") as fo:
        fo.write(json.dumps(correct_only_data[:80000], indent=4))
    with open("../LLaMA-Factory/data/webinstruct_critique_wrong_80k_0119.json", "w") as fo:
        fo.write(json.dumps(incorrect_only_data[:80000], indent=4))


if __name__ == "__main__":
    main()

