import os
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder, Repository
import json


def upload_jsonl_dataset_to_hf(local_dir, repo_id):
    dataset_dict = {}

    # 遍历所有 jsonl 文件
    for filename in os.listdir(local_dir):
        if filename.endswith(".jsonl"):
            split_name = filename.replace(".jsonl", "").replace("-", "_").lower()
            file_path = os.path.join(local_dir, filename)

            # 读取 jsonl 文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]
                dataset = Dataset.from_list(lines)
                dataset_dict[split_name] = dataset

    # 创建 DatasetDict
    dataset_dict = DatasetDict(dataset_dict)

    # 推送到 Hugging Face Hub
    dataset_dict.push_to_hub(repo_id)


upload_jsonl_dataset_to_hf(
    local_dir="formal_release",  # 替换为你本地的目录
    repo_id="TIGER-Lab/One-Shot-CFT-Data"
    #repo_id="ubowang/One-Shot-CFT-Data"
)

