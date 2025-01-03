from extract_answer_0103 import extract_answer_0103
import json

with open("test_data/test_1.json", "r") as fi:
    single_data = json.load(fi)

pred = extract_answer_0103("math", single_data["solution"])
print("pred", pred)
