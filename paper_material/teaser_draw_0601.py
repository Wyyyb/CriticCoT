import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['Qwen2.5-Math-1.5B', 'Llama3.2-3B-Instruct', 'Qwen2.5-Math-7B', 'Qwen2.5-14B']

# 每个模型的方法(注意Qwen2.5-14B只有4个方法)
methods = ['base', 'base (sober)', 'SFT (1 ex)', 'SFT (full)', 'RL (1 ex)', 'CFT (1 ex)']

# 每个模型每种方法的平均分数
scores = [
    # Qwen2.5-Math-1.5B
    [21.1, 25.0, 18.5, 18.8, 33.8, 32.8],
    # Llama3.2-3B-Instruct (没有base (sober)数据，用0代替)
    [17.5, 0, 15.4, 16.5, 19.0, 21.7],
    # Qwen2.5-Math-7B
    [27.3, 32.2, 22.9, 25.6, 40.2, 42.2],
    # Qwen2.5-14B (没有base (sober)和RL (1 ex)数据，用0代替)
    [27.1, 0, 24.6, 25.8, 0, 36.0]
]

# 设置图形大小
plt.figure(figsize=(14, 8))

# 设置条形的宽度
bar_width = 0.13

# 设置条形的位置
r = np.arange(len(models))

# 绘制每种方法的条形图
for i, method in enumerate(methods):
    valid_scores = []
    valid_positions = []
    valid_models = []

    for j, model in enumerate(models):
        if scores[j][i] > 0:  # 只绘制有数据的方法
            valid_scores.append(scores[j][i])
            valid_positions.append(r[j] + (i - 2.5) * bar_width)  # 居中对齐
            valid_models.append(model)

    bars = plt.bar(valid_positions, valid_scores, width=bar_width, label=method)

    # 在条形上方添加数值标签
    for bar, score in zip(bars, valid_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{score}', ha='center', va='bottom', fontsize=9)

# 添加标签和标题
plt.xlabel('Models', fontweight='bold', fontsize=12)
plt.ylabel('Average Score (%)', fontweight='bold', fontsize=12)
plt.title('Average Performance on Mathematical Benchmarks', fontweight='bold', fontsize=14)
plt.xticks(r, models, fontsize=10)
plt.ylim(0, 50)  # 设置y轴范围

# 添加图例
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()