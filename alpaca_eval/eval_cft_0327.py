import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alpaca_eval import evaluate

# 设置环境变量（可选）
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 使用的GPU编号

# 加载模型和分词器
model_name = "TIGER-Lab/Qwen2.5-Math-7B-CFT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)


# 生成回答的函数
def generate_response(prompt, max_new_tokens=2048):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


# 准备评估数据
# 可以使用alpaca_eval自带的数据集或自定义数据集
# 这里我们使用alpaca_eval的标准评估集

# 为模型创建输出文件
output_file = "qwen25_math_7b_responses.jsonl"


# 读取评估数据集并生成回答
def generate_model_outputs(data_path="alpaca_eval_data.jsonl"):
    with open(output_file, "w") as f_out:
        with open(data_path, "r") as f_in:
            for line in tqdm(f_in):
                item = json.loads(line)
                instruction = item["instruction"]
                prompt = f"以下是一个指令，请提供恰当的回应。\n\n指令: {instruction}\n\n回应:"
                response = generate_response(prompt)

                output_item = {
                    "instruction": instruction,
                    "output": response,
                    "generator": "qwen25_math_7b"
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")


# 运行评估
def run_evaluation():
    # 步骤1：生成模型输出（如果还没有）
    if not os.path.exists(output_file):
        generate_model_outputs()

    # 步骤2：使用alpaca_eval评估输出
    results = evaluate(
        output_file,
        annotators_config="alpaca_eval_gpt4",  # 使用GPT-4作为评判
        num_threads=4,  # 调整线程数
        batch_size=10,  # 调整批次大小
        model_outputs=True  # 保存模型输出
    )

    # 打印评估结果摘要
    print(results)

    # 保存详细评估结果
    with open("qwen25_math_7b_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_evaluation()