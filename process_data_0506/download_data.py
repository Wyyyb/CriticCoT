import os
import json
from datasets import load_dataset


def download_webinstruct_verified(output_path="../local_data/cft_data_0506/webinstruct_verified_data_0506.json", split="train", num_samples=None):
    """
    从Hugging Face下载TIGER-Lab/WebInstruct-verified数据集并保存为JSON格式

    参数:
        output_path (str): 保存JSON文件的路径
        split (str): 要下载的数据集分割，例如'train'或'test'
        num_samples (int, 可选): 限制下载样本的数量，如果为None则下载所有样本

    返回:
        str: 保存的JSON文件路径
    """
    try:
        os.makedirs("../local_data/cft_data_0506/", exist_ok=True)
        # 从Hugging Face加载数据集
        print(f"正在加载TIGER-Lab/WebInstruct-verified数据集的{split}分割...")
        dataset = load_dataset("TIGER-Lab/WebInstruct-verified", split=split)

        # 如果指定了样本数量限制
        if num_samples and num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # 将数据集转换为列表
        data_list = [item for item in dataset]
        output_data = []
        for item in data_list:
            if item.get("category") != "Mathematics":
                continue
            if item.get("answer_type") == "Multiple Choice":
                continue
            item["gt_answer"] = item["answer"]
            item.pop("answer")
            output_data.append(item)
        print("len(output_data)", len(output_data))
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"成功下载并保存了{len(data_list)}条数据到{output_path}")
        return output_path

    except Exception as e:
        print(f"下载数据时出错: {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # 下载训练集，限制为1000个样本
    download_webinstruct_verified("../local_data/cft_data_0506/webinstruct_train.json", "train")

    # 下载测试集，不限制样本数量
    # download_webinstruct_verified("webinstruct_test.json", "test")