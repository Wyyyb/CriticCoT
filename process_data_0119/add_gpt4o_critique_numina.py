import json
import os
from multiprocessing import Process
import time
from tqdm import tqdm
from typing import Callable
from openai import OpenAI
from pathlib import Path


def process_chunk(start_idx: int,
                  end_idx: int,
                  input_path: str,
                  output_dir: str,
                  prompt_func: Callable,
                  process_id: int):
    """
    处理数据的一个chunk
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构建输出文件路径
    output_file = os.path.join(output_dir, f'{start_idx + 1}-{end_idx}.json')

    # 读取已有的输出结果（如果存在）
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = {item['idx']: item for item in json.load(f)}

    # 读取输入数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 选择当前进程需要处理的数据
    chunk_data = data[start_idx:end_idx]

    # 初始化OpenAI客户端
    client = OpenAI()

    # 存储结果
    results = []

    # 处理每条数据
    for item in tqdm(chunk_data, desc=f'Process {process_id}'):
        # 检查是否已经处理过
        if item['idx'] in existing_results:
            results.append(existing_results[item['id']])
            continue

        # 构造prompt并调用API
        try:
            messages = prompt_func(item)

            # 调用GPT-4
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=3200,
                top_p=0.95
            )
            # 保存结果
            item['model_output'] = completion.choices[0].message.content
            item['cost'] = completion.usage.completion_tokens * 10 / 1e6 + completion.usage.prompt_tokens * 2.5 / 1e6
            results.append(item)

            # 定期保存结果
            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing item {item['id']}: {str(e)}")
            continue
            # 保存错误信息
            # item['error'] = str(e)
            # results.append(item)

        # 添加短暂延迟避免API限制
        time.sleep(0.1)

    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_large_dataset(input_path: str,
                          output_dir: str,
                          prompt_func: Callable,
                          num_processes: int = 20):
    """
    主函数：处理大型数据集

    Args:
        input_path: 输入JSON文件路径
        output_dir: 输出目录
        prompt_func: 构造prompt的函数
        num_processes: 进程数量
    """
    # 获取数据总量
    with open(input_path, 'r', encoding='utf-8') as f:
        total_items = len(json.load(f))

    # 计算每个进程处理的数据量
    chunk_size = total_items // num_processes

    # 创建进程
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

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("All processes completed!")


# 使用示例
def example_prompt_func(item):
    question = item["question"]
    qwen_math_answer = item["qwen_2.5_math_answer"]
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


if __name__ == "__main__":
    # 使用示例
    input_path = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/on_policy_data_0119/" \
                 "qwen_math_numina_80k_0119.json"
    output_dir = "/gpfs/public/research/xy/yubowang/CriticCoT/local_data/on_policy_data_0119/" \
                 "qwen_math_numina_80k_add_critique_0119"

    process_large_dataset(
        input_path=input_path,
        output_dir=output_dir,
        prompt_func=example_prompt_func
    )


