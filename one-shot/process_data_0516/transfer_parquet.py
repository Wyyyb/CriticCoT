import pandas as pd
import json
import os


def parquet_to_json(parquet_file_path, json_file_path=None):
    """
    将Parquet文件转换为JSON文件

    参数:
        parquet_file_path: Parquet文件的路径
        json_file_path: 输出JSON文件的路径，如果不指定，将使用与parquet文件相同的名称和路径，但扩展名为.json

    返回:
        输出的JSON文件路径
    """
    # 如果未指定JSON文件路径，则生成默认路径
    if json_file_path is None:
        base_name = os.path.splitext(parquet_file_path)[0]
        json_file_path = f"{base_name}.json"

    # 读取Parquet文件
    try:
        df = pd.read_parquet(parquet_file_path)
    except Exception as e:
        raise Exception(f"读取Parquet文件时出错: {e}")

    # 将DataFrame转换为JSON并写入文件
    try:
        # 将DataFrame转换为JSON字符串列表
        json_records = df.to_json(orient='records', force_ascii=False)

        # 写入JSON文件
        with open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json_records)

        print(f"成功将Parquet文件转换为JSON文件: {json_file_path}")
        return json_file_path

    except Exception as e:
        raise Exception(f"转换为JSON文件时出错: {e}")

# 使用示例
parquet_to_json("merge_pi1_pi2_pi13_pi1209_r128.parquet", "merge_pi1_pi2_pi13_pi1209_r128.json")

