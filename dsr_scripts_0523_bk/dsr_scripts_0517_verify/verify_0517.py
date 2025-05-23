import re, json, os


def extract_boxed_answer(pred_str: str):
    if "boxed" not in pred_str:
        return None
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


def filter_correct(item, p_index):
    if p_index == 0:
        if "8\\sqrt[3]{4}" in item or "8 \\sqrt[3]{4}" in item:
            return True
        elif "8 \\cdot \\sqrt[3]{4}" in item:
            return True
        elif "12.7" in item:
            return True
        else:
            return False
    if p_index == 1:
        if "48" in item:
            return True
        else:
            return False




def get_answer_map(data):
    answer_map = {}
    for each_data in data:
        critique = each_data["messages"][1]["content"]
        answer = extract_boxed_answer(critique)
        if answer not in answer_map:
            answer_map[answer] = 0
        answer_map[answer] += 1
    return answer_map



def main():
    for i in range(4):
        file_path = f"dsr_one-shot_train_data_0514_p{str(i)}.jsonl"
        input_data = []
        with open(file_path, "r") as f:
            for line in f:
                input_data.append(json.loads(line))
        print(i)
        print(get_answer_map(input_data))


if __name__ == "__main__":
    main()



