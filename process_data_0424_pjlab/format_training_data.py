import json


def main():
    output_path = "../LLaMA-Factory/data/deepmath_qwen_32b_distill_cft_0424.json"
    with open("/mnt/hwfile/opendatalab/yubo/CriticCoT/local_data/deepmath_cft_data/"
              "deepmath_integrate_data_0423_add_critique_p1.json",
              "r") as fi:
        data = json.load(fi)
    format_data = []
    invalid_count = 0
    for k, each in data.items():
        if not each.get("qwen-2.5-32b_answer_valid") or each.get("DeepSeek-R1-Distill-Qwen-32B_critique_valid"):
            invalid_count += 1
            continue
        question = each["question"]
        ins = each["instruction_3"]
        solution = each["qwen-2.5-32b_answer"]
        critique = each["DeepSeek-R1-Distill-Qwen-32B_critique"]
        curr_input = f"Question:\n{question}\n\nStudent's Solution:\n{solution}\n\n"
        curr_output = f"Critique:\n{critique}"
        format_data.append({"instruction": ins, "input": curr_input, "output": curr_output})
    print("invalid_count", invalid_count)
    print("valid_count", len(format_data))
    with open(output_path, "w") as fo:
        fo.write(json.dumps(format_data, indent=2))


if __name__ == "__main__":
    main()


