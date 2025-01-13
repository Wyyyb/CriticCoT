import json
import os
import argparse


def load_pred_res(file_dir):
    pred_res = []
    for file in os.listdir(file_dir):
        if not file.endswith("jsonl"):
            continue
        with open(os.path.join(file_dir, file), "r") as fi:
            for line in fi.readlines():
                pred_res.append(json.loads(line))
    print("load len(pred_res)", len(pred_res))
    return pred_res


def get_grade(input_int):
    ratio = 1
    while input_int > 10:
        input_int = input_int // 10
        ratio = ratio * 10
    return ratio


def single_dataset_sta(pred_dir):
    elements = pred_dir.strip("/").split("/")
    dataset_name = elements[-1]
    pred_res = load_pred_res(pred_dir)
    idx_map = {}
    max_idx = -1
    for each in pred_res:
        if each["idx"] not in idx_map:
            idx_map[each["idx"]] = each
        max_idx = max(each["idx"], max_idx)
    interval = get_grade(max_idx)
    real_idx_map = {}
    for idx, each in idx_map.items():
        real_idx = idx % interval
        if real_idx not in real_idx_map:
            real_idx_map[real_idx] = [each]
        else:
            real_idx_map[real_idx].append(each)
        real_idx_map[real_idx] = sorted(real_idx_map[real_idx], key=lambda x: x["idx"])
    critic_res_map = {}
    total_right = 0.0
    total_wrong = 0.0
    for real_idx, pred_list in real_idx_map.items():
        if len(pred_list) != 8:
            print("length of pred_list", len(pred_list))
            print("pred_list", pred_list)
        if len(pred_list) != args.candidate_num:
            pred_list = pred_list[:args.candidate_num]
        critic_results = []
        preds = []
        scores = []
        vote_res = 0
        responses = []
        question = pred_list[0]["question"]
        answer = pred_list[0]["answer"]
        for pred_info in pred_list:
            if question != pred_info["question"]:
                print("inconsistent question", question, pred_info["question"])
            if answer != pred_info["answer"]:
                print("inconsistent answer", answer, pred_info["answer"])
            response = pred_info["code"][0]
            pred = pred_info["pred"][0]
            score = pred_info["score"][0]
            critic_result = extract_critic_res(response)
            critic_results.append(critic_result)
            preds.append(pred)
            scores.append(score)
            responses.append(response)
            if critic_result:
                if score:
                    vote_res += 1
                else:
                    vote_res -= 1
        if vote_res < 0:
            final_score = False
            total_wrong += 1
        elif vote_res > 0:
            final_score = True
            total_right += 1
        else:
            print("same number of correct and incorrect!")
            final_score = False
        critic_res_map[real_idx] = {"idx": real_idx, "question": question, "answer": answer,
                                    "preds": preds, "scores": scores, "critic_results": critic_results,
                                    "final_score": final_score}
        if real_idx < 200:
            critic_res_map[real_idx]["responses"] = responses
    accuracy = total_right / (total_wrong + total_right)
    return dataset_name, accuracy, critic_res_map


def extract_critic_res(response):
    # 忽略大小写
    response = response.lower()

    # 定义正确和错误的关键词
    correct_patterns = [
        "conclusion: correct",
        "conclusion:correct",
        "conclusion: right",
        "conclusion:right",
        "conclusion: true",
        "conclusion:true",
    ]

    incorrect_patterns = [
        "conclusion: incorrect",
        "conclusion:incorrect",
        "conclusion: wrong",
        "conclusion:wrong",
        "conclusion: false",
        "conclusion:false",
    ]

    # 检查正确模式
    for pattern in correct_patterns:
        if pattern in response:
            return True

    # 检查错误模式
    for pattern in incorrect_patterns:
        if pattern in response:
            return False

    # 如果没有找到匹配模式,返回None
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="", type=str)
    parser.add_argument("--summary_path", default="", type=str)
    parser.add_argument("--candidate_num", default=8, type=int)
    return parser.parse_args()


def main():
    for sub_dir in os.listdir(args.input_dir):
        if sub_dir not in ["math", "minerva_math"]:
            print("unsupported dataset", sub_dir)
            continue
        print("Processing dataset", sub_dir)
        curr_dir = os.path.join(args.input_dir, sub_dir)
        dataset_name, accuracy, critic_res_map = single_dataset_sta(curr_dir)
        summary_content = args.input_dir + "\t" + dataset_name + "\tAccuracy:" + str(accuracy) + "\n"
        result_path = os.path.join(curr_dir, f"{dataset_name}-critique_result_file.json")
        with open(result_path, "w") as fo:
            fo.write(json.dumps(critic_res_map, indent=4))
        with open(args.summary_path, "a") as fo:
            fo.write(summary_content)


if __name__ == "__main__":
    args = parse_args()
    main()

