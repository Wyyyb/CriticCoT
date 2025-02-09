import json
import random


def main():
    input_path = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/webinstruct_cft_80k_0121_p3.json"
    output_path = "../local_data/webinstruct_0207/webinstruct_selected_2k_data_0207_p3.json"
    with open(input_path, "r") as fi:
        data = json.load(fi)
    print(len(data))
    data = sorted(data, key=lambda x: -len(x["output"]))
    selected_2k = data[4000: 6000]
    output_data = []
    for i, each in enumerate(selected_2k):
        output_data.append(single_format(each, i + 4000))
    # random.shuffle(output_data)
    with open(output_path, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))
    print("len(output_data)", len(output_data))
    print("average answer length", sta_average_length(output_data))


def sta_average_length(input_list):
    total_length = 0.0
    for each in input_list:
        total_length += len(each["answer"])
    return total_length / len(input_list)


def single_format(item, curr_idx):
    segs = item["input"].split("\n\nSolution:\n")
    question = segs[0].replace("Question:\n", "")
    answer = segs[1].strip("\n")
    idx = item.get("idx", curr_idx)
    return {"idx": idx, "question": question, "answer": answer}


main()







