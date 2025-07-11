""" Preprocess dataset for swe repair task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
# from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import re

system_prompt = """
You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

We have access to the following functions:

–– BEGIN FUNCTION #1: file_editor ––
Description:
Custom editing tool for viewing, creating and editing files
•	State is persistent across command calls and discussions with the user
•	If path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
•	The create command cannot be used if the specified path already exists as a file
•	If a command generates a long output, it will be truncated and marked with <response clipped>
•	The undo_edit command will revert the last edit made to the file at path

Notes for using the str_replace command:
•	The old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
•	If the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique
•	The new_str parameter should contain the edited lines that should replace the old_str

Parameters:
1.	command (string, required)
Allowed values: [view, create, str_replace, insert, undo_edit]
The command to run.
2.	path (string, required)
Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
3.	file_text (string, optional)
Required for the create command. Contains the content of the file to be created.
4.	old_str (string, optional)
Required for the str_replace command. The exact string in path to replace.
5.	new_str (string, optional)
•	Optional for the str_replace command to specify the replacement string.
•	Required for the insert command to specify the string to insert.
6.	insert_line (integer, optional)
Required for the insert command. The new_str will be inserted after the line number specified here.
7.	view_range (array, optional)
•	Optional for the view command (when path is a file).
•	If provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
•	[start_line, -1] will show all lines from start_line to the end of file.
8.	concise (boolean, optional)
•	Optional for the view command.
•	Defaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

–– END FUNCTION #1 ––

–– BEGIN FUNCTION #2: execute_bash ––
Description:
Execute a bash command in the terminal.

Behavior notes:
•	If a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.
•	If the bash command returns exit code -1, it means the process is still running. The assistant may:
•	Call this function again with command as an empty string ("") to retrieve additional logs.
•	Send more input to STDIN of the running process by calling this function again with command set to the text input.
•	Send command="ctrl+c" to interrupt the currently running process.
•	If the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.

Parameters:
1.	cmd (string, required)
The bash command (and optional arguments) to execute.
•	Can be empty ("") to retrieve more logs if the process is still running.
•	Can be "ctrl+c" to interrupt the running process.

–– END FUNCTION #2 ––

–– BEGIN FUNCTION #3: search ––
Description:
Search for a term in a directory or a single file.
•	If path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.
•	If path points to a file, it runs a grep -n in that file to show line numbers matching the search term.
•	If more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
•	If no matches are found, it will inform you as well.

Parameters:
1.	search_term (string, required)
The term or string to search for in files.
2.	path (string, optional)
The file or directory to search in. Defaults to . if not specified.

–– END FUNCTION #3 ––

–– BEGIN FUNCTION #4: finish ––
Description:
Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
•	The submit command finalizes your output.

Parameters:
1.	command (string, required)
Currently allowed value: [submit]
2.	result (string, optional)
The result text or final message to submit. Defaults to an empty string if not provided.

–– END FUNCTION #4 ––

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.
""".strip()

instance_prompt_template = """
Consider the following github issue:
<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

IMPORTANT TIP:
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
6. When viewing large files, use specific line-ranges, usually within 50 to 100 lines) as required
7. NOTE: The repository is at '/testbed' and the current working directory is already '/testbed', so DO NOT include 'testbed/' or 'testbed.' in relative paths in bash commands or reproduction python files. 
""".strip()

def get_task_instruction(ds) -> str:
    content = ds["problem_statement"]
    match = re.search(r"\[ISSUE\](.*?)\[/ISSUE\]", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/data/minimax-dialogue/users/xiancai//verl/data/r2e/0623')
    parser.add_argument('--data_path', default='/data/minimax-dialogue/users/xiancai/hf_datasets/R2E-Gym/SWE-Bench-Verified/swe-bench-verified-test.jsonl')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--data_source', type=str, default='r2e_rl_data')
    parser.add_argument('--dataset_name', type=str, default='swe_bench_verified')
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
                yield data
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))


    def make_map_fn(split):
        def process_fn(example, idx):
            problem_statement = get_task_instruction(example)
            user_prompt = instance_prompt_template.format(
                problem_statement=problem_statement
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            data = {
                "data_source": data_source,
                "prompt": messages,
                "ability": "r2e",
                "reward_model": {
                    "style": "sandbox",
                    "ground_truth": ""
                },
                "raw_dataset": json.dumps(dict(example)),
                "extra_info": {
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