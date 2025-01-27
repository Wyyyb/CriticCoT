import json
import os
import random


def single_format_cft(question, solution, critique):
    t2_curr = {"instruction": "Please critique whether the following solution to the question is correct.\n\n",
               "input": question + f"\nSolution:\n{solution}\n", "output": critique}
    return t2_curr


def load_ori_10k_data():
    input_path = "../LLaMA-Factory/data/ace_10k_gpt4o_cft_0121.json"
    with open(input_path, "r") as fi:
        data = json.load(fi)
    return data


def main():
    input_file_1 = "/cpfs/data/user/yubowang/CriticCoT/local_data/webinstruct_mini_batch_data/batchinput-1.jsonl"
    # input_file_2 = "/cpfs/data/user/yubowang/CriticCoT/local_data/numina_batch_data_0123/batchinput-2.jsonl"
    output_file_1 = "/cpfs/data/user/yubowang/CriticCoT/local_data/webinstruct_mini_batch_data/batchoutput-1.jsonl"
    # output_file_2 = "/cpfs/data/user/yubowang/CriticCoT/local_data/numina_batch_data_0123/batchoutput-2.jsonl"
    output_file = "/cpfs/data/user/yubowang/CriticCoT/local_data/WebInstruct_40k_critique_gpt-4o-mini_0127/WebInstruct_40k_critique_gpt-4o-mini_0127.json"

    input_data = {}
    with open(input_file_1, "r") as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            request_id = curr["custom_id"]
            content = curr["body"]["messages"][1]["content"][0]["text"]
            segs = content.split("\nSolution:\n")
            if len(segs) < 2:
                print("skip it")
                continue
            question = segs[0]
            if question.startswith("\n    Question: "):
                question = question.replace("\n    Question: ", "")
            answer = segs[1]
            # critique =
            input_data[request_id] = {"question": question, "answer": answer}
    # with open(input_file_2, "r") as fi:
    #     for line in fi.readlines():
    #         curr = json.loads(line)
    #         request_id = curr["custom_id"]
    #         content = curr["body"]["messages"][1]["content"][0]["text"]
    #         segs = content.split("\n\n    Answer: ")
    #         question = segs[0]
    #         if question.startswith("\n    Question: "):
    #             question = question.replace("\n    Question: ", "")
    #         answer = segs[1]
    #         # critique =
    #         input_data[request_id] = {"question": question, "answer": answer}

    with open(output_file_1, "r") as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            request_id = curr["custom_id"]
            content = curr["response"]["body"]["choices"][0]["message"]["content"]
            critique = content
            if request_id not in input_data:
                print("request_id not in input_data", request_id)
                continue
            input_data[request_id]["critique"] = critique
    # with open(output_file_2, "r") as fi:
    #     for line in fi.readlines():
    #         curr = json.loads(line)
    #         request_id = curr["custom_id"]
    #         content = curr["response"]["body"]["choices"][0]["message"]["content"]
    #         critique = content
    #         if request_id not in input_data:
    #             print("request_id not in input_data", request_id)
    #             continue
    #         input_data[request_id]["critique"] = critique

    print("input data number", len(input_data))
    output_data = []
    for k, v in input_data.items():
        output_data.append(single_format_cft(v["question"], v["answer"], v["critique"]))
    # ori_10k_data = load_ori_10k_data()
    # output_data = output_data + ori_10k_data
    random.shuffle(output_data)
    print("output data number", len(output_data))
    with open(output_file, "w") as fo:
        fo.write(json.dumps(output_data, indent=4))


main()

