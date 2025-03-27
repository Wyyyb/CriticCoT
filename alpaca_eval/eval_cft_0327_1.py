import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from alpaca_eval import evaluate

# 加载模型和tokenizer
model_name = "TIGER-Lab/Qwen2.5-Math-7B-CFT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

# 定义生成回答的函数
def qwen_generate(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

# 使用alpaca_eval评测模型
results = evaluate(
    model=qwen_generate,
    model_name="Qwen2.5-Math-7B-CFT",
    tasks=["alpaca_eval_gpt4"],  # 使用alpaca_eval标准基准数据集
    limit=100  # 根据需要设置评测题目数量
)

# 输出评测结果
print(results)