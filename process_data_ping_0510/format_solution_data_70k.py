import json
import os


def add_ids(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    data = add_extract_ans(data)
    count = 0
    for k, v in data.items():
        data[k]["ori_id"] = data[k]["id"]
        data[k]["id"] = str(count)
        count += 1
    print("count", count)
    with open(output_path, "w") as f:
        f.write(json.dumps(data, indent=4))


def extract_boxed_answer(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def parse_answer(qwen3_32b_answer):
    segments = qwen3_32b_answer.split("</think>")
    if len(segments) != 2:
        print("len(segments) != 2: ", len(segments))
        return None, None
    return "<think>\n" + segments[0].strip() + "\n</think>", segments[1].strip()


def add_extract_ans(data):
    output_data = {}
    for k, v in data.items():
        qwen3_32b_answer = v["qwen3-32b_answer"]
        thinking_content, short_answer = parse_answer(qwen3_32b_answer)
        if short_answer is None or thinking_content is None:
            continue
        extracted_answer = extract_boxed_answer(qwen3_32b_answer)
        if extracted_answer is None or extracted_answer == "":
            continue
        v["qwen3-32b_extracted_answer"] = extracted_answer
        v.pop("qwen3-32b_answer")
        v.pop("qwen3-32b_answer_valid")
        v.pop("qwen3-32b_answer_correctness")
        v["qwen3-32b_thinking_content"] = thinking_content
        v["qwen3-32b_short_answer"] = short_answer
        output_data[k] = v
    return output_data


if __name__ == "__main__":
    add_ids("../local_data/cft_data_0506/webinstruct_data_add_solution_0506_merged.json",
           "../local_data/cft_data_0506/webinstruct_data_add_solution_0506_ids_merged.json")

