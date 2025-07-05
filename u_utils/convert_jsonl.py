import json

# 读取第一个文件并打印长度
with open("deepscaler_critique_formatted.json", "r") as fi:
    critique_data = json.load(fi)
    print(len(critique_data))

# 读取第二个文件并打印长度
with open("deepscaler_train_filter.json", "r") as fi:
    train_data = json.load(fi)
    print(len(train_data))

# 将第一个文件转换为JSONL格式
with open("deepscaler_critique_formatted.jsonl", "w") as fo:
    for item in critique_data:
        fo.write(json.dumps(item, ensure_ascii=False) + "\n")

# 将第二个文件转换为JSONL格式
with open("deepscaler_train_filter.jsonl", "w") as fo:
    for item in train_data:
        fo.write(json.dumps(item, ensure_ascii=False) + "\n")

print("转换完成！")