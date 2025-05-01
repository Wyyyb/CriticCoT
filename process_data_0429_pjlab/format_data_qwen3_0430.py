import json


def extract_con(critique):
    # if "Conclusion" not in text:
    #     print("Conclusion not found", text)
    #     return None
    segs = critique.split("Conclusion")
    res_seg = segs[-1].lower()
    if "wrong" in res_seg or "incorrect" in res_seg:
        return False
    elif "right" in res_seg or "correct" in res_seg:
        return True
    else:
        # print("*******************recognize failed\n", res_seg, text)
        return False


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


def filter_single(item):
    if not item.get("qwen-2.5-32b_answer_valid") or not item.get("Qwen3-32B_critique_valid"):
        return False
    critique_conclusion = extract_con(item.get("Qwen3-32B_critique"))
    if critique_conclusion == item.get("qwen-2.5-32b_answer_correctness"):
        return True
    return False


def main():
    output_path = "../360-LLaMA-Factory-sp/data/deepmath_qwen3_32b_correction_data_0430.json"
    with open("/mnt/hwfile/opendatalab/yubo/CriticCoT/local_data/deepmath_cft_data/"
              "deepmath_integrate_data_0501_qwen3_add_critique.json",
              "r") as fi:
        data = json.load(fi)
    format_data = []
    invalid_count = 0
    for k, each in data.items():
        if not each.get("qwen-2.5-32b_answer_valid") or not each.get("Qwen3-32B_critique_valid"):
            invalid_count += 1
            continue
        if not filter_single(each):
            continue
        question = each["question"]
        ins = "Analyze the student's solution step-by-step, identify key concepts, " \
              "check for errors, correct them if needed, and conclude with \"Conclusion: " \
              "right/wrong [END]\" and the final answer in \\boxed{}."
        solution = each["qwen-2.5-32b_answer"]
        critique = each["Qwen3-32B_critique"]
        critique_answer = extract_boxed_answer(critique)
        if critique_answer is None:
            continue
        curr_input = f"Question:\n{question}\n\nStudent's Solution:\n{solution}\n\n"
        curr_output = f"Critique:\n{critique}"
        format_data.append({"instruction": ins, "input": curr_input, "output": curr_output})
    print("filtered invalid_count", invalid_count)
    print("filtered valid_count", len(format_data))
    with open(output_path, "w") as fo:
        fo.write(json.dumps(format_data, indent=2))


if __name__ == "__main__":
    main()


