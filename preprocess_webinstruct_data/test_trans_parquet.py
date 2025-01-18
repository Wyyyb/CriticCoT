import pandas as pd
import json

# 读取parquet文件
df = pd.read_parquet('math_sft.parquet')

# # 转换为JSON文件
output_file = '../local_data/AceMath-Instruct-Training-Data/math_sft.jsonl'
# with open(output_file, 'w', encoding='utf-8') as f:
#     # 逐行写入JSONL格式
#     for _, row in df.iterrows():
#         json_str = json.dumps(row.to_dict(), ensure_ascii=False)
#         f.write(json_str + '\n')

#df = pd.read_parquet('math_sft.parquet')

# 直接转换为JSONL
df.to_json(output_file, orient='records', lines=True, force_ascii=False)

