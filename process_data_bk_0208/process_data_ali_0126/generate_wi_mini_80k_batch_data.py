import json
import os


def example_prompt_func(item):
    # question = item["input"]
    # qwen_math_answer = item["output"]
    # question = f"""
    # Question: {question}
    #
    # Answer: {qwen_math_answer}
    # """
    question = item["input"]
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a science expert. A student is trying to solve the a question, please explain briefly whether his answer is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]'."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                }
            ]
        },
    ]
    return chat_prompt


def main():
    input_path = "/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/data/webinstruct_cft_80k_0119.json"
    output_path_1 = "/cpfs/data/user/yubowang/CriticCoT/local_data/webinstruct_mini_batch_data/batchinput-1.jsonl"
    output_path_2 = "/cpfs/data/user/yubowang/CriticCoT/local_data/webinstruct_mini_batch_data/batchinput-2.jsonl"
    with open(input_path, "r") as fi:
        sft_data = json.load(fi)
    # sft_data = sft_data[10000:]
    batch_data = []
    for i, each in enumerate(sft_data):
        custom_id = f"request-{str(i)}"
        messages = example_prompt_func(each)
        curr = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body":
                {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "max_tokens": 3200,
                    "temperature": 0.3,
                    "top_p": 0.95
                }
            }
        batch_data.append(curr)
    with open(output_path_1, "w") as fo:
        for each in batch_data[:40000]:
            fo.write(json.dumps(each) + "\n")

    with open(output_path_2, "w") as fo:
        for each in batch_data[40000:]:
            fo.write(json.dumps(each) + "\n")


main()


