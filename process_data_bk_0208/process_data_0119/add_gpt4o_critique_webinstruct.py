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

    client = OpenAI()

    results = []

    for item in tqdm(chunk_data, desc=f'Process {process_id}'):
        if item['idx'] in existing_results:
            results.append(existing_results[item['id']])
            continue

        try:
            messages = prompt_func(item)

            # 调用GPT-4
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=messages,
                temperature=0.3,
                max_tokens=3200,
                top_p=0.95
            )
            item['model_output'] = completion.choices[0].message.content
            item['cost'] = completion.usage.completion_tokens * 10 / 1e6 + completion.usage.prompt_tokens * 2.5 / 1e6
            results.append(item)

            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing item {item['id']}: {str(e)}")
            continue
            # item['error'] = str(e)
            # results.append(item)

        time.sleep(0.1)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_large_dataset(input_path: str,
                          output_dir: str,
                          prompt_func: Callable,
                          num_processes: int = 200):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        total_items = len(data)
    print("len(data)", len(data))
    # add idx
    for i, each in enumerate(data):
        if "idx" not in each:
            data[i]["idx"] = i
    with open(input_path, "w") as fo:
        fo.write(json.dumps(data, indent=4))

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
    question = item["question"]
    qwen_math_answer = item["answer"]
    question = f"""
    Question: {question}

    Answer: {qwen_math_answer}
    """
    chat_prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a science expert. A student is trying to solve the a question, please explain briefly whether his answer is correct or not. Finally, conclude your judgement with 'Conclusion: right/wrong [END]'.\n\n" + question
                }
            ]
        },
    ]
    return chat_prompt


if __name__ == "__main__":
    input_path = "path/to/original_dataset.json"
    output_dir = "output_dir/"
    os.makedirs(output_dir, exist_ok=True)
    process_large_dataset(
        input_path=input_path,
        output_dir=output_dir,
        prompt_func=example_prompt_func
    )


