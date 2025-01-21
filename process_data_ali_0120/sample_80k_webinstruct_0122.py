import json
import os
import random
random.seed(12345)


def has_number(string):
    return any(char.isdigit() for char in string)


def main():
    input_file = "../LLaMA-Factory/data/CriticCoT_t2_critic_data_0115.json"
    output_file_base_path = "../LLaMA-Factory/data/webinstruct_cft_80k_0122_p"
    with open(input_file, "r", encoding="utf-8") as fi:
        ori_data = json.load(fi)
    curr_data = []
    for each in ori_data:
        segs = each["input"].split("\nSolution:\n")
        if len(segs) != 2:
            print("Invalid input", each["input"])
            continue
        question = segs[0]
        answer = "Solution:\n" + segs[1]
        critique = each["output"]
        critique = critique.replace(" [END]", "")
        instruction = each["instruction"]
        if not has_number(question) and not has_number(answer):
            continue
        if random.randint(0, 10) > 5:
            instruction = instruction.replac("solution", "answer")
            answer = answer.replace("Solution:\n", "Answer:\n")
        if random.randint(0, 10) > 5:
            instruction = instruction.replace("correct", "right")
        prefix = "Critique:\nLet's"
        if not critique.startswith(prefix):
            continue
        curr_data.append({"instruction": instruction, "question": question,
                          "answer": answer, "critique": critique})
    print("stage 1 data number", len(curr_data))
    answer_length_data = sorted(curr_data, key=lambda x: len(x["answer"]))
    curr_data = answer_length_data[:-1000]
    critique_length_data = sorted(curr_data, key=lambda x: len(x["critique"]))
    curr_data = critique_length_data[5000:]

    random.shuffle(curr_data)

    format_data = []
    for each in curr_data:
        format_data.append(single_format(each))
    print("length of format data", len(format_data))
    sample_num = 36
    for i in range(sample_num):
        random.shuffle(format_data)
        file_path = output_file_base_path + str(i) + ".json"
        with open(file_path, "w") as fo:
            fo.write(json.dumps(format_data[:80000], indent=4))


def single_format(item):
    instruction = item.get("instruction", "")
    question = item.get("question", "")
    answer = item.get("answer", "")
    critique = item.get("critique", "")
    res = {"instruction": instruction, "input": f"{question}\n{answer}", "output": critique}
    return res


main()
