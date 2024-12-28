import os
import json
from tqdm import tqdm


def load_jsonl(data_dir):
    data = []
    # critic_id = 0
    for file in os.listdir(data_dir):
        if not file.endswith(".jsonl"):
            continue
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                # curr["critic_id"] = str(critic_id)
                # critic_id += 1
                data.append(curr)
    return data


def parse_single_critic(text):
    # if "Conclusion" not in text:
    #     print("Conclusion not found", text)
    #     return None
    segs = text.split("Conclusion")
    res_seg = segs[-1].lower()
    if "wrong" in res_seg or "incorrect" in res_seg:
        return False
    elif "right" in res_seg or "correct" in res_seg:
        return True
    else:
        # print("*******************recognize failed\n", res_seg, text)
        return False


def generate_gpt4o_answer(index):
    if int(index) == 9036097:
        return '''Your answer shows a systematic approach to calculating the addition to retained earnings, which is good. Let's analyze your steps:

1. Calculation of Total Expenses ✓
   - Operating expenses (costs): $308,000
   - Non-operating expenses: $60,000 + $40,000 = $100,000
   - Total expenses: $408,000

2. Tax Calculation ✓
   - Pre-tax income: $604,000 - $408,000 = $196,000
   - Tax (35%): $196,000 * 0.35 = $68,600

3. Net Income Calculation ✓
   - Net Income = $604,000 - $408,000 - $68,600 = $127,400

4. Addition to Retained Earnings ✓
   - Addition = Net Income - Dividends
   - $127,400 - $75,000 = $52,400

Your solution demonstrates:
- Correct organization of operating and non-operating expenses
- Proper application of the tax rate
- Accurate calculation of net income
- Appropriate consideration of dividends in determining the final addition to retained earnings

All calculations and methodology are correct.

Conclusion: right [END]'''
    if int(index) == 8614631:
        return '''Your answer correctly approaches the problem using vector components and demonstrates a thorough understanding of trigonometry and vector addition. Let's review the key steps:

Correctness of method:
- Breaking the motion into x and y components ✓
- Using cosine and sine functions appropriately ✓
- Using Pythagorean theorem for final displacement ✓

However, there are a few critical errors:

1. Angle interpretation error:
   - For the first leg (80° north of east), the angle should be measured from the x-axis, making it 10° (not 80°)
   - For the second leg (70° north of west), the correct angle from x-axis should be 160° (not 110°)

2. Calculation errors:
   - First leg x-component should be: 40 * cos(10°) = 39.39 km (not -36.87 km)
   - First leg y-component should be: 40 * sin(10°) = 6.94 km (not 38.39 km)
   - Second leg x-component should be: 40 * cos(160°) = -37.59 km (not -15.45 km)
   - Second leg y-component should be: 40 * sin(160°) = 13.68 km (not 33.68 km)

3. The final answer should be approximately 48.5 km, not 80.38 km.

While the mathematical approach and problem-solving strategy are sound, the incorrect angle interpretations led to wrong numerical results.

Conclusion: wrong [END]'''
    return None


def transfer_single(item, mode):
    question = item.get("question", "")
    answer = item.get("answer", "")
    index = item.get("index", "")
    gpt4o_answer = item.get("gpt4o_answer", "")
    if gpt4o_answer is None:
        gpt4o_answer = generate_gpt4o_answer(index)
        print("generate_gpt4o_answer result:\n", gpt4o_answer)
    if gpt4o_answer is None:
        print("gpt4o_answer is None index\n", index)
        print("gpt4o_answer is None item\n", item)
        return None
    is_correct = parse_single_critic(gpt4o_answer)
    # if is_correct is None:
    #     # print("parse_single_critic failed", gpt4o_answer)
    #     return None
    if mode == "correct_only":
        if is_correct:
            return format_correct_only(question, answer)
        else:
            return None
    elif mode == "critic":
        return format_critic_data(question, answer, gpt4o_answer)
    else:
        print("unsupported mode", mode)
        return None


def format_correct_only(question, answer):
    ins = "Please analyze and answer the following question:"
    response = answer.replace("####\n", "\nSummary:\n")
    return {"instruction": ins,
            "input": question,
            "output": response}


def format_critic_data(question, answer, gpt4o_answer):
    question = "Please analyze and evaluate the following solution:\n\nQuestion:\n" + question
    student_solution = "\nStudent's Solution:\n" + answer
    critique = "Critique:\n" + gpt4o_answer
    return {"instruction": question,
            "input": student_solution,
            "output": critique}


def main():
    input_data_dir = "/home/wenhuche/WebInstruct/gpt4-filtered"
    input_data = load_jsonl(input_data_dir)
    mode = "correct_only"
    print("creating correct_only data")
    training_data = []
    for each in tqdm(input_data):
        curr = transfer_single(each, mode)
        if curr is not None:
            training_data.append(curr)
    os.makedirs("../local_data", exist_ok=True)
    os.makedirs("../local_data/critic_training_data_1228", exist_ok=True)
    with open("../local_data/critic_training_data_1228/CriticCoT_correct_only_data_1228.json", "w") as fo:
        fo.write(json.dumps(training_data, indent=2))
    print("length of correct_only data", len(training_data))
    mode = "critic"
    print("creating critic data")
    training_data = []
    for each in tqdm(input_data):
        curr = transfer_single(each, mode)
        if curr is not None:
            training_data.append(curr)
    with open("../local_data/critic_training_data_1228/CriticCoT_critic_data_1228.json", "w") as fo:
        fo.write(json.dumps(training_data, indent=2))
    print("length of critic data", len(training_data))


if __name__ == "__main__":
    main()

