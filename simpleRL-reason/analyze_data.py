import pandas as pd
import json
import numpy as np

# 读取数据
df = pd.read_parquet('data/train.parquet')

print('=== 数据概览 ===')
print(f'总行数: {len(df)}')
print(f'列数: {len(df.columns)}')

print('\n=== 列名 ===')
print(df.columns.tolist())

print('\n=== 各列的唯一值数量 ===')
for col in df.columns:
    try:
        unique_count = df[col].nunique()
        print(f'{col}: {unique_count}')
    except:
        print(f'{col}: 无法计算唯一值 (可能是numpy数组)')

print('\n=== data_source分布 ===')
print(df['data_source'].value_counts())

print('\n=== ability分布 ===')
print(df['ability'].value_counts())

print('\n=== prompt字段的详细结构 ===')
print('第一行的prompt:')
print(df.iloc[0]['prompt'])
print(f'类型: {type(df.iloc[0]["prompt"])}')
if isinstance(df.iloc[0]['prompt'], np.ndarray):
    print(f'数组形状: {df.iloc[0]["prompt"].shape}')
    print(f'数组内容: {df.iloc[0]["prompt"].tolist()}')

print('\n=== reward_model字段的详细结构 ===')
print('第一行的reward_model:')
print(df.iloc[0]['reward_model'])
print(f'类型: {type(df.iloc[0]["reward_model"])}')

print('\n=== 前3行的完整数据 ===')
for i in range(min(3, len(df))):
    print(f'\n--- 第{i+1}行 ---')
    row = df.iloc[i]
    for col in df.columns:
        if col in ['prompt', 'reward_model', 'extra_info']:
            print(f'{col}: {type(row[col])}')
            if isinstance(row[col], np.ndarray):
                print(f'  数组内容: {row[col].tolist()}')
            else:
                print(f'  内容: {row[col]}')
        else:
            print(f'{col}: {row[col]}')

print('\n=== 数据类型 ===')
print(df.dtypes)

print('\n=== 数据样例 ===')
print('第一行的问题:')
print(df.iloc[0]['question'])
print('\n第一行的答案:')
print(df.iloc[0]['answer'])
print('\n第一行的目标答案:')
print(df.iloc[0]['target'])

print('\n=== 前3行的JSON格式美化输出 ===')
for i in range(min(3, len(df))):
    row = df.iloc[i].copy()
    # 将numpy数组的prompt转为list
    if isinstance(row['prompt'], np.ndarray):
        row['prompt'] = row['prompt'].tolist()
    # 只保留可序列化的内容
    try:
        json_str = json.dumps(row.to_dict(), ensure_ascii=False, indent=2)
        print(f'\n--- 第{i+1}行 ---')
        print(json_str)
    except Exception as e:
        print(f'第{i+1}行无法序列化: {e}') 