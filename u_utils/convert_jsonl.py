import json
import pandas as pd

# 读取第一个文件并打印长度
with open("../verl_data/training_data/deepscaler_critique_formatted/deepscaler_critique_formatted.json", "r") as fi:
    critique_data = json.load(fi)
    print(len(critique_data))

# 读取第二个文件并打印长度
with open("../verl_data/training_data/deepscaler_train_filter/deepscaler_train_filter.json", "r") as fi:
    train_data = json.load(fi)
    print(len(train_data))

# 将第一个文件转换为DataFrame，然后保存为Parquet格式
critique_df = pd.DataFrame(critique_data)
critique_df.to_parquet("../verl_data/training_data/deepscaler_critique_formatted/deepscaler_critique_formatted.parquet", index=False)

# 将第二个文件转换为DataFrame，然后保存为Parquet格式
train_df = pd.DataFrame(train_data)
train_df.to_parquet("../verl_data/training_data/deepscaler_train_filter/deepscaler_train_filter.parquet", index=False)

print("转换完成！")