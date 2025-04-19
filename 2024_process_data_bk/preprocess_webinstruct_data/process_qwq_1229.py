import os
import json
from tqdm import tqdm
import random


def load_qwq(dir_path):
    data = []
    print("loading qwq data...")
    for file in tqdm(os.listdir(dir_path)):
        if not file.endswith("jsonl"):
            continue
        file_path = os.path.join(dir_path, file)
        with open(file_path, "r") as fi:
            for line in fi:
                curr = json.loads(line)
                data.append(curr)
    print("data number:", len(data))
    return data


def format_qwq(ori_data):
    res_data = []
    instruction = "Please analyze and answer the following question step by step:"
    for each in tqdm(ori_data):
        message = each["api_req"]["messages"]
        if len(message) != 2:
            print("message more than 2:\n", message)
        question = message[1]["content"]
        answer = each["api_resp"]["choices"][0]["message"]["content"]
        curr = {"instruction": instruction, "input": question, "output": answer}
        res_data.append(curr)
    return res_data


def load_critic_data(data_path):
    with open(data_path, "r") as fi:
        data = json.load(fi)
    return data


def save_data(data, data_path):
    random.shuffle(data)
    with open(data_path, "w") as fo:
        fo.write(json.dumps(data, indent=2))
    print("length of data:", len(data), data_path)


def main():
    qwq_data_dir_path = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/webinstruct_qwq_462k/"
    critic_data_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_critic_data_1228.json"
    ori_qwq_data = load_qwq(qwq_data_dir_path)
    qwq_data = format_qwq(ori_qwq_data)
    critic_data = load_critic_data(critic_data_path)
    output_qwq_data_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_qwq_data_1229.json"
    output_qwq_critic_data_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/" \
                                  "data/CriticCoT_qwq_critic_data_1229.json"
    save_data(qwq_data, output_qwq_data_path)
    save_data(qwq_data + critic_data, output_qwq_critic_data_path)


if __name__ == "__main__":
    main()

