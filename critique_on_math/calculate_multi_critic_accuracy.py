import json
import os
import argparse


def load_res(file_path):
    res = {}
    with open(file_path, "r", encoding='utf-8') as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            real_idx = curr["real_idx"]
            if real_idx not in res:
                res[real_idx] = []
            res[real_idx].append(curr)
    return res


def main(input_dir, summary_path):
    original_res = {}
    candidate_res = {}
    for file in os.listdir(input_dir):
        if not file.endswith("_add_critique.jsonl"):
            continue
        if "t0.0" in file:
            original_res = load_res(os.path.join(input_dir, file))
        elif "t0.6" in file:
            candidate_res = load_res(os.path.join(input_dir, file))
        else:
            print("unsupported input file", file)
    final_res = {}
    total_right = 0.0
    total_wrong = 0.0
    for real_idx, res_list in original_res.items():
        ori_res = res_list[0]
        if ori_res["critique_pred"]:
            final_res[real_idx] = ori_res
            if ori_res["score"][0]:
                total_right += 1
            else:
                total_wrong += 1
        else:
            candidates = candidate_res[real_idx]
            flag = False
            for i, cand in enumerate(candidates):
                if cand["critique_pred"]:
                    flag = True
                    final_res[real_idx] = cand
                    if cand["score"][0]:
                        total_right += 1
                    else:
                        total_wrong += 1
            if flag:
                continue
            if not flag:
                final_res[real_idx] = ori_res
                if ori_res["score"][0]:
                    total_right += 1
                else:
                    total_wrong += 1

    final_res_file_path = os.path.join(input_dir, "final_res_0117.json")
    with open(final_res_file_path, "w") as fo:
        fo.write(json.dumps(final_res, indent=2))
    accuracy = total_right / (total_right + total_wrong)
    with open(summary_path, "a") as fo:
        fo.write(f"total_right: {total_right}\ntotal_wrong: {total_wrong}\naccuracy: {accuracy}\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process math training files')
    parser.add_argument('--summary_path', type=str, required=True,
                        help='Path to the summary')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing math_train folder')

    args = parser.parse_args()
    main(args.input_dir, args.summary_path)



