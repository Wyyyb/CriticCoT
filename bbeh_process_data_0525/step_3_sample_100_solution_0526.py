import json
import random


def remove_think(solution):
    if "</think>" in solution:
        return solution.split("</think>")[-1]
    return solution


def main():
    with open("2_add_solution_data/add_solution_0526.json", "r") as f:
        data = json.load(f)
    output_data = []
    global_id = 0
    print(len(data))
    for k, v in data.items():
        question = v["question"]
        source = v["source"]
        gt_answer = v["gt_answer"]
        count = 0
        curr_question_solutions = []
        candidates = []
        for student, solutions in v["student_solutions"].items():
            solutions = sorted(solutions, key=lambda x: len(x["solution"]))
            valid_count = 0
            for each in solutions:
                if each["extracted_answer"] is None:
                    continue
                solution = each["solution"]
                solution = remove_think(solution)
                if len(solution) > 30000:
                    continue
                if len(each["extracted_answer"]) > 100 > len(gt_answer):
                    continue
                correctness = each["exact_match_correctness"]
                critique_id = k + "_" + str(count)
                single_critique = {"critique_id": critique_id, "question": question,
                                   "solution": solution, "gt_answer": gt_answer,
                                   "solution_correctness": correctness,
                                   "student_model": student, "student_extracted_answer": each["extracted_answer"],
                                   "source": source, "global_id": global_id}
                if valid_count >= 10:
                    candidates.append(single_critique)
                    global_id += 1
                    count += 1
                    continue
                curr_question_solutions.append(single_critique)
                valid_count += 1
                global_id += 1
                count += 1
        random.shuffle(candidates)
        if len(curr_question_solutions) < 100:
            curr_question_solutions += candidates[:100 - len(curr_question_solutions)]
        print("source", source)
        print("len(curr_question_solutions)", len(curr_question_solutions))
        output_data += curr_question_solutions
    with open("3_add_critique_data/add_critique_data_0526.json", "w") as f:
        f.write(json.dumps(output_data, indent=4))
    print("len(output_data)", len(output_data))

if __name__ == "__main__":
    main()





















