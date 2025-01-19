from datasets import Dataset
import os
from openai import OpenAI
from multiprocessing import Process
import json
import time
import tqdm


client = OpenAI()


def prompt(dataset: Dataset, index: int):
    handle = open(f'gpt4-generated/outputs.process_{index}.jsonl', 'w')
    for entry in tqdm.tqdm(dataset, desc=f"Processing process_{index}"):
        question = f"""
Question: {entry['question']}
"""
        #Prepare the chat prompt
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a science expert. You need to answer a given question clearly and conclude your answer with 'Answer: [YOUR ANSWER]' in the end."
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
        # Include speech result if speech is enabled
        messages = chat_prompt
        # Generate the completion
        max_retries = 8
        base_delay = 2
        retry_count = 0

        while retry_count < max_retries:
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=3200,
                    temperature=0.3,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={"type": "text"},
                    stop=None,
                    stream=False
                )
                entry['gpt4o_answer'] = completion.choices[0].message.content
                entry['cost'] = completion.usage.completion_tokens * 10 / 1e6 + completion.usage.prompt_tokens * 2.5 / 1e6
                break
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Failed after {max_retries} retries. Final error: {e}")
                    entry['gpt4o_answer'] = None
                    entry['cost'] = 0
                    break

                delay = base_delay ** retry_count
                print(f"Error on attempt {retry_count}/{max_retries}: {e}")
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        handle.write(json.dumps(entry) + '\n')

    handle.close()


def split_dataset(dataset, num_splits):
    split_size = len(dataset) // num_splits
    splits = [
        dataset.select(range(i * split_size, (i + 1) * split_size)) for i in range(num_splits - 1)
    ]
    splits.append(dataset.select(range((num_splits - 1) * split_size, len(dataset))))
    return splits


if __name__ == "__main__":

    with open('cft_merged_80k.json', 'r') as f:
        dataset = json.load(f)

    # Use only the question part
    new_dataset = []
    for entry in dataset:
        question = entry['input'].split('Solution:')[0]
        new_dataset.append({'question': question})
    dataset = new_dataset

    dataset = Dataset.from_list(new_dataset)
    dataset = dataset.select(range(0, 50000))

    num_processes = 100
    dataset_splits = split_dataset(dataset, num_processes)

    processes = []
    for i, split in enumerate(dataset_splits):
        p = Process(target=prompt, args=(split, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()