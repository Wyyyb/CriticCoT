import json
import random


def main():
    with open("../local_data/bespoke_data/ori_bespoke_data.json", "r") as fi:
        ori_data = json.load(fi)
    format_data = []
    for each in ori_data:
        format_data.append(single_format(each))
    random.shuffle(format_data)
    with open("../local_data/bespoke_data/bespoke_17k_format_data_0206.json", "w") as fo:
        fo.write(json.dumps(format_data, indent=4))
    format_data = sorted(format_data, key=lambda x: -len(x["answer"]))
    selected_2k = format_data[:2000]
    random.shuffle(selected_2k)
    with open("../local_data/bespoke_data/bespoke_2k_data_0206.json", "w") as fo:
        fo.write(json.dumps(selected_2k, indent=4))


def single_format(ori_single):
    question = ori_single["conversations"][0]["value"]
    answer = ori_single["conversations"][1]["value"]
    return {"question": question, "answer": answer}


main()
