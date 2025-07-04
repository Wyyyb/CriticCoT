# 使用 JSON 格式训练数据

本文档说明如何将训练数据从 Parquet 格式转换为 JSON 格式，并使用 JSON 格式进行训练。

## 1. 数据格式转换

### 1.1 使用转换脚本

我们提供了一个转换脚本来将 Parquet 文件转换为 JSON 格式：

```bash
# 转换单个文件
python convert_parquet_to_json.py --input data/train.parquet --output data/train.json

# 转换为 JSONL 格式（每行一个 JSON 对象）
python convert_parquet_to_json.py --input data/train.parquet --output data/train.jsonl --jsonl

# 批量转换目录中的所有 parquet 文件
python convert_parquet_to_json.py --input data/ --output json_data/ --batch
```

### 1.2 JSON 格式说明

支持两种 JSON 格式：

#### 1.2.1 标准 JSON 格式
```json
[
  {
    "answer": "118",
    "gt_answer": "118",
    "subject": "Intermediate Algebra",
    "level": 5,
    "question": "Let $a$ and $b$ be the two real values...",
    "target": "118",
    "data_source": "simplelr_qwen",
    "prompt": [
      {
        "content": "<|im_start|>system\nYou are a helpful assistant...",
        "role": "user"
      }
    ],
    "ability": "math",
    "reward_model": {
      "ground_truth": "118",
      "style": "rule"
    },
    "extra_info": {
      "answer": "118",
      "index": 0,
      "level": 1,
      "question": "Let $a$ and $b$ be the two real values...",
      "split": "train"
    }
  }
]
```

#### 1.2.2 JSONL 格式（每行一个 JSON 对象）
```jsonl
{"answer": "118", "gt_answer": "118", "prompt": [{"content": "...", "role": "user"}], ...}
{"answer": "5", "gt_answer": "5", "prompt": [{"content": "...", "role": "user"}], ...}
```

## 2. 修改训练配置

### 2.1 使用 JSON 配置文件

使用提供的 JSON 配置文件：

```bash
python -m verl.trainer.main_ppo --config-path verl/trainer/config --config-name ppo_trainer_json
```

### 2.2 修改现有配置

在现有的配置文件中，将数据文件路径从 `.parquet` 改为 `.json`：

```yaml
data:
  train_files: ~/data/rlhf/gsm8k/train.json  # 从 .parquet 改为 .json
  val_files: ~/data/rlhf/gsm8k/test.json     # 从 .parquet 改为 .json
  # ... 其他配置保持不变
```

### 2.3 修改训练脚本

在 `train_grpo_math_tune_ray.sh` 脚本中，修改数据路径：

```bash
# 将
data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \

# 改为
data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.json \
data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.json \
```

## 3. 实现细节

### 3.1 新增的 JSONDataset 类

我们在 `verl/utils/dataset/rl_dataset.py` 中添加了 `JSONDataset` 类，支持：

- 标准 JSON 文件（包含对象数组）
- JSONL 格式（每行一个 JSON 对象）
- 自动检测文件格式
- 与现有 RLHFDataset 相同的接口

### 3.2 自动格式检测

训练器会自动检测文件扩展名：
- `.json` 或 `.jsonl` → 使用 JSONDataset
- `.parquet` → 使用 RLHFDataset（原有行为）

### 3.3 数据兼容性

JSONDataset 完全兼容现有的数据格式，包括：
- `prompt` 字段的对话格式
- `reward_model` 字段的字典格式
- `extra_info` 字段的元数据

## 4. 使用示例

### 4.1 转换当前数据

```bash
# 转换训练数据
python convert_parquet_to_json.py --input data/train.parquet --output data/train.json

# 转换验证数据
python convert_parquet_to_json.py --input data/test.parquet --output data/test.json
```

### 4.2 使用 JSON 格式训练

```bash
# 使用 JSON 配置文件
python -m verl.trainer.main_ppo --config-path verl/trainer/config --config-name ppo_trainer_json

# 或者直接指定 JSON 文件路径
python -m verl.trainer.main_ppo \
  data.train_files=data/train.json \
  data.val_files=data/test.json \
  # ... 其他参数
```

## 5. 优势

使用 JSON 格式的优势：

1. **可读性更好**：JSON 格式是人类可读的
2. **调试方便**：可以直接查看和编辑 JSON 文件
3. **工具支持**：大多数编程语言和工具都支持 JSON
4. **版本控制友好**：JSON 文件在版本控制系统中更容易比较差异
5. **跨平台兼容**：JSON 格式在不同系统间完全兼容

## 6. 注意事项

1. **文件大小**：JSON 文件通常比 Parquet 文件大
2. **加载速度**：JSON 文件加载可能比 Parquet 慢
3. **内存使用**：JSON 格式可能需要更多内存
4. **向后兼容**：现有代码仍然支持 Parquet 格式

## 7. 故障排除

### 7.1 常见问题

1. **JSON 解析错误**：检查 JSON 文件格式是否正确
2. **内存不足**：对于大文件，考虑使用 JSONL 格式
3. **编码问题**：确保 JSON 文件使用 UTF-8 编码

### 7.2 调试技巧

```python
# 验证 JSON 文件格式
import json
with open('data/train.json', 'r') as f:
    data = json.load(f)
print(f"Loaded {len(data)} records")
print(f"First record keys: {list(data[0].keys())}")
``` 