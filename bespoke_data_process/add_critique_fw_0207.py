from firework_r1_call_0207 import *
import json
import os
import time


def format_single_query(question, answer):
    ins = "You are a science expert. A student is trying to solve a question. Please " \
          "thoroughly analyze and provide detailed critique of their solution, including " \
          "their conceptual understanding, methodology, and any errors or merits, then " \
          "conclude your judgement with 'Conclusion: right/wrong [END]'."
    query = f"{ins}\n\nQuestion:\n{question}\nSolution:\n{answer}"
    return query


def load_data(input_path, res_path):
    with open(input_path, "r") as fi:
        input_data = json.load(fi)
    exist_idx = []
    if os.path.exists(res_path):
        with open(res_path, "r") as fi:
            res_data = json.load(fi)
    else:
        res_data = []
    for each in res_data:
        if each["idx"] not in exist_idx:
            exist_idx.append(each["idx"])
    return input_data, res_data, exist_idx


def single_call(item):
    start = time.time()
    query = format_single_query(item["question"], item["answer"])
    print("query", query)
    content, cost, prompt_tokens, completion_tokens = single_request(query)
    if content is not None:
        item["critique"] = content
        item["cost"] = cost
        item["prompt_tokens"] = prompt_tokens
        item["completion_tokens"] = completion_tokens
        print("cost time:", time.time() - start)
        return item
    return None


def main():
    input_data_path = "../local_data/webinstruct_0207/webinstruct_selected_2k_data_0207.json"
    res_data_path = "../local_data/webinstruct_0207/add_critique_webinstruct_2k_data_0207.json"
    input_data, res_data, exist_idx = load_data(input_data_path, res_data_path)
    for item in input_data:
        if item["idx"] in exist_idx:
            continue
        single_res = single_call(item)
        if single_res is None:
            continue
        res_data.append(single_res)
        with open(res_data_path, "w") as fo:
            fo.write(json.dumps(res_data, indent=4))


main()






