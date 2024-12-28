import os
import json
from tqdm import tqdm


def load_jsonl(data_dir):
    data = []
    # critic_id = 0
    for file in os.listdir(data_dir):
        if not file.endswith(".jsonl"):
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
    # if "Conclusion" not in text:
    #     print("Conclusion not found", text)
    #     return None
    segs = text.split("Conclusion")
    res_seg = segs[-1].lower()
    if "wrong" in res_seg or "incorrect" in res_seg:
        return False
    elif "right" in res_seg or "correct" in res_seg:
        return True
    else:
        print("*******************recognize failed\n", res_seg, text)
        return False


def transfer_single(item, mode):
    question = item.get("question", "")
    answer = item.get("answer", "")
    # index = item.get("index", "")
    gpt4o_answer = item.get("gpt4o_answer", "")
    if gpt4o_answer is None:
        return None
    is_correct = parse_single_critic(gpt4o_answer)
    if is_correct is None:
        print("parse_single_critic failed", gpt4o_answer)
        return None
    if mode == "correct_only":
        if is_correct:
            return format_correct_only(question, answer)
        else:
            return None
    elif mode == "critic":
        return format_critic_data(question, answer, gpt4o_answer)
    else:
        print("unsupported mode", mode)
        return None


def format_correct_only(question, answer):
    question = "Please analyze and answer the following question:\n" + question
    response = answer.replace("####\n", "\nSummary:\n")
    return {"instruction": question,
            "output": response}


def format_critic_data(question, answer, gpt4o_answer):
    question = "Please analyze and evaluate the following solution:\n\nQuestion:\n" + question
    student_solution = "\nStudent's Solution:\n" + answer
    critique = "Critique:\n" + gpt4o_answer
    return {"instruction": question,
            "input": student_solution,
            "output": critique}


def main():
    input_data_dir = "/home/wenhuche/WebInstruct/gpt4-filtered"
    input_data = load_jsonl(input_data_dir)
    mode = "correct_only"
    print("creating correct_only data")
    training_data = []
    for each in tqdm(input_data):
        curr = transfer_single(each, mode)
        if curr is not None:
            training_data.append(curr)
    os.makedirs("../local_data", exist_ok=True)
    os.makedirs("../local_data/training_data_1228", exist_ok=True)
    with open("../local_data/critic_training_data_1228/CriticCoT_correct_only_data_1228.json", "w") as fo:
        fo.write(json.dumps(training_data, indent=2))
    print("length of correct_only data", len(training_data))
    mode = "critic"
    print("creating critic data")
    training_data = []
    for each in tqdm(input_data):
        curr = transfer_single(each, mode)
        if curr is not None:
            training_data.append(curr)
    with open("../local_data/critic_training_data_1228/CriticCoT_critic_data_1228.json", "w") as fo:
        fo.write(json.dumps(training_data, indent=2))
    print("length of critic data", len(training_data))


if __name__ == "__main__":
    main()

