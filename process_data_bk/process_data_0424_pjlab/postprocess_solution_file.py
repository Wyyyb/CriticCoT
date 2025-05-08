import json
import re


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


def is_same_answer(str_1, str_2):
    if not str_1 or not str_2:
        return False
    if str_1.replace("dfrac", "frac") == str_2.replace("dfrac", "frac"):
        # if str_1 != str_2:
        #     print("str_1, str_2", str_1, str_2)
        return True
    return str_1 == str_2


def main():
    input_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0421_add_solution_p1.json"
    output_file = "../local_data/deepmath_cft_data/deepmath_integrate_data_0428_add_solution.json"
    new_correct_count = 0
    with open(input_file, "r") as fi:
        data = json.load(fi)
    for k, v in data.items():
        discard_keys = ["instruction_1", "instruction_2", "instruction_3",
                        "r1_solution_1", "r1_solution_2", "r1_solution_3"]
        for each in discard_keys:
            if each in v:
                data[k].pop(each)
        ori_correctness = v.get("qwen-2.5-32b_answer_correctness", None)
        gt_answer = v["gt_answer"]
        qwen_answer = v["qwen-2.5-32b_answer"]
        qwen_short_answer = extract_boxed_answer(qwen_answer)
        data[k]["qwen-2.5-32b_short_answer"] = qwen_short_answer
        if qwen_short_answer:
            data[k]["qwen-2.5-32b_answer_valid"] = True
        if is_same_answer(gt_answer, qwen_short_answer):
            if not ori_correctness and ori_correctness is not None:
                new_correct_count += 1
            data[k]["qwen-2.5-32b_answer_correctness"] = True
        else:
            data[k]["qwen-2.5-32b_answer_correctness"] = False
    print("new_correct_count", new_correct_count)
    with open(output_file, "w") as fo:
        fo.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    main()


