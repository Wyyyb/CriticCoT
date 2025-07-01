import matplotlib.pyplot as plt
import numpy as np

# 数据分组（只保留average）
# 按照三个不同数据集的微调分组
groups = {
    'Causal Understanding': ['Qwen2.5-Math-7B', 'SFT (1 ex)', 'CFT (1 ex)'],
    'DisambiguationQA': ['Qwen2.5-Math-7B', 'SFT (1 ex)', 'CFT (1 ex)'],
    'Time Arithmetic': ['Qwen2.5-Math-7B', 'SFT (1 ex)', 'CFT (1 ex)'],
    'All Tasks (3 ex)': ['Qwen2.5-Math-7B', 'SFT (3 ex)', 'CFT (3 ex)']
}

# 平均分数数据
average_scores = {
    'Causal Understanding': [10.5, 13.7, 25.2],
    'DisambiguationQA': [10.5, 10.6, 20.4],
    'Time Arithmetic': [10.5, 13.4, 26.4],
    'All Tasks (3 ex)': [10.5, 15.9, 26.8]
}

# 创建画布
fig, ax = plt.subplots(figsize=(14, 8))

# 设置条形的宽度和位置
bar_width = 0.2
x = np.arange(len(groups))

# 为每个模型类型创建条形
for i, model_type in enumerate(['Qwen2.5-Math-7B', 'SFT', 'CFT']):
    scores = [average_scores[group][i] for group in groups.keys()]
    position = x - bar_width + i * bar_width
    bars = ax.bar(position, scores, bar_width,
                  label=model_type,
                  alpha=0.8)

    # 在条形上方添加数值标签
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{height}',
                ha='center', va='bottom', fontsize=9)

# 添加标题和轴标签
ax.set_title('Average Performance Across Different Fine-tuning Settings', fontsize=16)
ax.set_ylabel('Average Score', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(groups.keys(), fontsize=12)

# 添加图例
ax.legend(fontsize=12)

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 设置y轴范围
ax.set_ylim(0, max([max(scores) for scores in average_scores.values()]) + 5)

# 调整布局
fig.tight_layout()

plt.show()