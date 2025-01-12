import json
from datasets import load_dataset
import random
import os
from tqdm import tqdm


def load_numina():
    no_box_count = 0
    numina_data = []
    ds = load_dataset("AI-MO/NuminaMath-CoT")
    output_file = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/NuminaMath-CoT/ori_numina_data.jsonl"
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in ds["train"]:
                json_str = json.dumps(item, ensure_ascii=False)
                f.write(json_str + '\n')
    with open(output_file, "r") as fi:
        for line in tqdm(fi):
            curr = json.loads(line)
            numina_data.append(curr)
            if "\\boxed{" not in curr["solution"]:
                # print("boxed not in data", curr["source"])
                no_box_count += 1
    print("no_box_count", no_box_count)
    print("len(numina_data)", len(numina_data))
    return numina_data


def trans_single(question, answer):
    ins_1 = "Please reason step by step, and put your final answer within \\boxed{}."
    ins_2 = "Please analyze and answer the following question step by step:"
    if "\\boxed{" not in answer:
        ins = ins_2
    else:
        ins = ins_1
    return {"instruction": ins,
            "input": question,
            "output": answer}


def load_critic():
    critic_data_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/CriticCoT_critic_data_1231.json"
    with open(critic_data_path, "r") as fi:
        critic_data = json.load(fi)
    return critic_data


def main():
    res_numina = []
    numina_data = load_numina()
    critic_data = load_critic()
    for each in tqdm(numina_data):
        question = each["problem"]
        answer = each["solution"]
        single = trans_single(question, answer)
        res_numina.append(single)
    random.shuffle(res_numina)
    critic_numina_260k_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/critic_numina_260k_data_0110.json"
    critic_numina_860k_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/critic_numina_860k_data_0110.json"
    numina_path = "/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/data/numina_860k_data_0110.json"
    critic_numina_260k_data = critic_data + res_numina[:260000]
    random.shuffle(critic_numina_260k_data)
    critic_numina_860k_data = critic_data + res_numina
    random.shuffle(critic_numina_860k_data)
    # random.shuffle(res_numina)
    with open(critic_numina_260k_path, "w") as fo:
        fo.write(json.dumps(critic_numina_260k_data, indent=4))
    print("len(critic_numina_260k_data)", len(critic_numina_260k_data))
    with open(critic_numina_860k_path, "w") as fo:
        fo.write(json.dumps(critic_numina_860k_data, indent=4))
    print("len(critic_numina_860k_data)", len(critic_numina_860k_data))
    with open(numina_path, "w") as fo:
        fo.write(json.dumps(res_numina, indent=4))
    print("len(numina_data)", len(res_numina))


if __name__ == "__main__":
    main()

