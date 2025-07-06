""" Preprocess dataset for swe repair task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
# from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

AGENTLESS_REPAIR = """We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs.

--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first localize the bug based on the issue statement step by step, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap each *SEARCH/REPLACE* edit in a code block as shown in the example above. If you have multiple *SEARCH/REPLACE* edits, use a separate code block for each one."""

CODE_FILE = """
### {path}
{content}"""

THINKING_SYSTEM = """
A user will ask you to solve a task. You should first draft your thinking process (inner monologue). Then, generate the solution.

Your response format must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct solution.
</think>
<solution>
Final solution presented to the user.
</solution>
""".strip()

THINKING_SYSTEM_0321 = """
You are a helpful programming assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. \
Now the user asks you to solve an issue within a repository. After thinking about the fault within <think> </think> tags and being confident to generate a correct solution, please clearly present your *SEARCH/REPLACE* edits to fix the issue within <answer> </answer> tags. i.e., \
<think>
Your reasoning process for fault localization here.
</think>
<answer>
```python
### file_path
<<<<<<< SEARCH
A contiguous chunk of lines to search for in the existing source code
=======
The lines to replace into the source code
>>>>>>> REPLACE
```
...
</answer>.
""".strip()

THINKING_SYSTEM_0324 = """
You are a helpful programming assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. \
Now the user asks you to solve an issue within a repository. After thinking, when you are confident to generate a correct solution, please clearly present your *SEARCH/REPLACE* edits to fix the issue within <answer> </answer> tags. i.e., \
<answer>
```python
### file_path
<<<<<<< SEARCH
A contiguous chunk of lines to search for in the existing source code
=======
The lines to replace into the source code
>>>>>>> REPLACE
```
</answer>.
""".strip()

THINKING_SYSTEM_0325 = """
You are a helpful programming assistant. \
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. \
Now the user asks you to solve an issue within a repository. After thinking, when you are confident to generate a correct solution, please clearly present your *SEARCH/REPLACE* edits to fix the issue.

Your response must follow the format below:
<think>
Your reasoning process localize the bug based on the issue statement.
</think>
<answer>
Final edits presented to the user.
</answer>.
""".strip()

THINKING_SYSTEM_BASE = """
The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a issue within a repository. After thinking, when you finally reach a conclusion, clearly state the edits within <answer> </answer> tags.
""".strip()


def make_messages(dp):
    # combine changed_file_contents and unchanged_file_contents dicts
    file_contents = json.loads(dp['file_contents'])
    content = "\n".join(
        [
            CODE_FILE.format(
                path=path, 
                content=content
            ) 
            for path, content in file_contents.items()
        ]
    )
    question = AGENTLESS_REPAIR.format(
        problem_statement=dp['problem_statement'], 
        content=content
    )

    messages = [
        # {
        #     "role": "system",
        #     "content": THINKING_SYSTEM_0325
        # },
        {
            "role": "user",
            "content": question
        }
    ]
    return messages

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/minimax-dialogue/users/xiancai//verl/data/r/0417')
    parser.add_argument('--data_path', default='/minimax-dialogue/users/xiancai/hf_datasets/SWE-bench_Lite/swe-bench-lite-test-claude-file-contents.jsonl')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--data_source', type=str, default='swe_rl_data')
    parser.add_argument('--dataset_name', type=str, default='swe_bench_lite')
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    split = args.split
    data_source = args.data_source
    dataset_name = args.dataset_name
    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                data["file_contents"] = json.dumps(data["file_contents"])
                yield data
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))


    def make_map_fn(split):
        def process_fn(example, idx):
            messages = make_messages(example)
            gt_patch = example['patch']
            solution = {
                "run_type": "swe",
                "dataset_name": dataset_name,
                "instance_id": example['instance_id'],
                "file_contents": example['file_contents'],
                "split": split
            }
            data = {
                "data_source": data_source,
                "prompt": messages,
                "ability": "swe",
                "reward_model": {
                    "style": "sandbox",
                    "solution": json.dumps(solution),
                    "answer": gt_patch,
                },
                "extra_info": {
                    "repo": example['repo'],
                    # "pull_number": example['pull_number'],
                    # "instance_id": example['instance_id'],
                    "base_commit": example['base_commit'],
                    # "file_contents": example['file_contents'],
                    "idx": idx,
                    "split": split
                }
            }
            return data
        return process_fn

    raw_dataset = raw_dataset.map(function=make_map_fn(split), with_indices=True)
    local_dir = args.local_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    raw_dataset.to_parquet(os.path.join(local_dir, f'{split}.parquet'))