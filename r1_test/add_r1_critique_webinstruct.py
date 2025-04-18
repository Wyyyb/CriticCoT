import json
import os
from multiprocessing import Process
import time
from tqdm import tqdm
from typing import Callable
from openai import OpenAI
from pathlib import Path
import requests


def process_chunk(start_idx: int,
                  end_idx: int,
                  input_path: str,
                  output_dir: str,
                  prompt_func: Callable,
                  process_id: int):
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{start_idx + 1}-{end_idx}.json')

    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = {item['idx']: item for item in json.load(f)}

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunk_data = data[start_idx:end_idx]

    # client = OpenAI()

    results = []

    for item in tqdm(chunk_data, desc=f'Process {process_id}'):
        if item['idx'] in existing_results:
            results.append(existing_results[item['idx']])
            continue

        try:
            messages = prompt_func(item)
            url = "https://api.siliconflow.cn/v1/chat/completions"
            payload = {
                "model": "deepseek-ai/DeepSeek-R1",
                "messages": messages,
                "stream": False,
                "max_tokens": 4096,
                "stop": ["[END]"],
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "response_format": {"type": "text"},
            }
            k = "sk-ht"
            headers = {
                "Authorization": f"Bearer {k}kongektpfxqolvifmbozbvfsjpjdosfhzwuseeuxiibvpc",
                "Content-Type": "application/json"
            }
            res_text = requests.request("POST", url, json=payload, headers=headers).text
            print("res_text", res_text)
            response = json.loads(res_text)
            item['model_output'] = response["choices"][0]["message"]["content"]
            print("\n\nmodel_output", item['model_output'])
            results.append(item)

            if len(results) % 1 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing item {item['idx']}: {str(e)}")
            continue
            # item['error'] = str(e)
            # results.append(item)

        time.sleep(0.1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_large_dataset(input_path: str,
                          output_dir: str,
                          prompt_func: Callable,
                          num_processes: int = 1):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        total_items = len(data)
    print("len(data)", len(data))
    # data = data[:100]
    # add idx
    for i, each in enumerate(data):
        if "idx" not in each:
            data[i]["idx"] = i
    with open(input_path, "w") as fo:
        fo.write(json.dumps(data, indent=4))

    total_items = 200
    chunk_size = total_items // num_processes

    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else total_items

        p = Process(
            target=process_chunk,
            args=(start_idx, end_idx, input_path, output_dir, prompt_func, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes completed!")


def example_prompt_func(item):
    # question = item["question"]
    # answer = item["answer"]
    segs = item["input"].split("\n\nSolution:\n")
    question = segs[0].replace("Question:\n", "")
    answer = segs[1]
    question = f"""
    Question: {question}

    Answer: {answer}
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


if __name__ == "__main__":
    input_path = "/map-vepfs/yubo/CriticCoT/LLaMA-Factory/data/webinstruct_cft_80k_0121_p3.json"
    output_dir = "../local_data/r1_critique_80k_0203"
    os.makedirs(output_dir, exist_ok=True)
    process_large_dataset(
        input_path=input_path,
        output_dir=output_dir,
        prompt_func=example_prompt_func
    )


