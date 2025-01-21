import json
import os


def example_prompt_func(item):
    question = item["input"]
    qwen_math_answer = item["output"]
    question = f"""
    Question: {question}

    Answer: {qwen_math_answer}
    """
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
    input_path = "/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/data/ace_80k_sft_0120.json"
    output_path = "/cpfs/data/user/yubowang/CriticCoT/local_data/batch_data_0122/batchinput.jsonl"
    with open(input_path, "r") as fi:
        sft_data = json.load(fi)
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
                    "model": "gpt-4o-2024-11-20",
                    "messages": messages,
                    "max_tokens": 3200,
                    "temperature": 0.3,
                    "top_p": 0.95
                }
            }
        batch_data.append(curr)
    with open(output_path, "w") as fo:
        for each in batch_data:
            fo.write(json.dumps(each) + "\n")


main()






















