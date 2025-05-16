import json, os
from datasets import load_dataset


def get_message(question, solution):
    instruction = "Please reason step by step to solve the following question, putting your final answer within \\boxed{{}}.\n\n"
    input_str = "Question:\n" + question
    output_str = "Answer:\n" + solution
    result = {"messages":
        [
            # {"role": "system", "content": instruction},
            {"role": "user", "content": instruction + input_str},
            {"role": "assistant", "content": output_str}
        ]
    }
    return result


def fetch_deepscaler_preview_data(output_file="../../local_data/sft_data_0516/dsr_full_40k_sft_data.json"):
    os.makedirs("../../local_data/sft_data_0516/", exist_ok=True)
    try:
        # 从Hugging Face加载数据集
        dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset")
        data_list = [item for item in dataset["train"]]
        output_data = []
        for item in data_list:
            question = item["problem"]
            solution = item["solution"]
            curr_data = get_message(question, solution)
            output_data.append(curr_data)

        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"数据已成功保存到 {output_file}")
        return True

    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        return False


fetch_deepscaler_preview_data()
