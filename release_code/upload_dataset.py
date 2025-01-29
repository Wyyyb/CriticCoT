from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
import json
import pandas as pd


def load_json_data(file_path):
    """加载JSON格式的数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_and_upload_dataset():
    # 登录到Hugging Face
    login()

    # 配置数据集路径和配置名称
    configs = {
        "WebInstruct-CFT-600K": "../local_data/WebInstruct-CFT-600K.json",
        "WebInstruct-CFT-50K": "../local_data/WebInstruct-CFT-50K.json",
        "WebInstruct-CFT-4K": "../local_data/WebInstruct-CFT-4K.json"
    }

    # 为每个配置创建数据集
    dataset_dict = {}
    for config_name, file_path in configs.items():
        print(f"Processing {config_name} configuration...")
        try:
            data = load_json_data(file_path)
            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)
            dataset_dict[config_name] = DatasetDict({
                "train": dataset
            })
        except Exception as e:
            print(f"Error processing {config_name}: {str(e)}")

    # 上传到Hub
    for config_name, dataset in dataset_dict.items():
        dataset.push_to_hub(
            "TigerLab/WebInstruct-CFT",
            config_name=config_name,
            private=True
        )


if __name__ == "__main__":
    create_and_upload_dataset()

