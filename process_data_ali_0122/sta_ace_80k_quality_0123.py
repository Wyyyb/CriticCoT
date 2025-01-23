import json
import os


def parse_single_critic(text):
    segs = text.split("Conclusion")
    res_seg = segs[-1].lower()
    if "wrong" in res_seg or "incorrect" in res_seg:
        return False
    elif "right" in res_seg or "correct" in res_seg:
        return True
    else:
        # print("*******************recognize failed\n", res_seg, text)
        return None


def main():
    input_file = ""
    with open(input_file, 'r') as fi:
        data = json.load(fi)
    unsure = 0.0
    correct = 0.0
    wrong = 0.0
    for each in data:
        critique = each["output"]
        critic_pred = parse_single_critic(critique)
        if critic_pred is None:
            unsure += 1
        elif critic_pred is True:
            correct += 1
        else:
            wrong += 1
    print("unsure", unsure)
    print("correct", correct)
    print("wrong", wrong)


main()


