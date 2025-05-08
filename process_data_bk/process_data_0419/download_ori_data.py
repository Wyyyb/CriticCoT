from datasets import load_dataset
import json
import os


def download_and_save_deepmath_dataset(output_dir="../local_data/deepmath_ori_data",
                                       splits=["train", "validation", "test"]):
    """
    从Hugging Face下载zwhe99/DeepMath-103K数据集并以jsonl格式保存到本地

    参数:
        output_dir (str): 保存jsonl文件的目录
        splits (list): 要下载的数据集分割部分，默认["train", "validation", "test"]

    返回:
        list: 保存的jsonl文件路径列表
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    print(f"正在从Hugging Face下载zwhe99/DeepMath-103K数据集...")
    dataset = load_dataset("zwhe99/DeepMath-103K")

    saved_files = []

    # 遍历每个分割并保存为jsonl
    for split in splits:
        if split in dataset:
            output_file = os.path.join(output_dir, f"deepmath_{split}.jsonl")
            print(f"正在将{split}分割保存到{output_file}...")

            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            saved_files.append(output_file)
            print(f"成功保存{split}分割，共{len(dataset[split])}条数据")
        else:
            print(f"警告：数据集中不存在{split}分割")

    print(f"完成！所有数据已保存到{output_dir}目录")
    return saved_files


# 使用示例
if __name__ == "__main__":
    saved_files = download_and_save_deepmath_dataset()
    print(f"保存的文件: {saved_files}")


