import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 创建一个包含两个子图的大图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 第一个子图 - 只保留Qwen2.5-Math-7B的数据
# 方法名称
methods = ['base', 'base(sober)', 'SFT(1ex)', 'SFT(full)', 'RL(1ex)', 'CFT(1ex)']

# Qwen2.5-Math-7B的各种方法的平均分数
scores = [27.3, 32.2, 22.9, 28.9, 40.2, 42.2]

# 生成颜色映射
colors = cm.viridis(np.linspace(0, 0.9, len(methods)))

# 柱子宽度为0.5，并且靠在一起
width = 0.5  # 柱子宽度

# 修改后的位置计算方式 - 连续放置柱子，不留间隙
x_positions = np.arange(len(methods)) * 0.78 + 0.8 # 0, 1, 2, 3, 4, 5

# 绘制条形图
bars = ax1.bar(x_positions, scores, width=width, color=colors, edgecolor='none')

# 在条形上方添加数值标签
for i, bar in enumerate(bars):
    ax1.text(bar.get_x() + width / 2, bar.get_height() + 0.8,
             f'{scores[i]}', ha='center', va='bottom', fontsize=15, fontweight='bold')

# 添加标签和标题
# ax1.set_xlabel('Methods', fontweight='bold', fontsize=18)
ax1.set_ylabel('Average Score (%)', fontweight='bold', fontsize=16)
ax1.set_title('1-shot CFT on Mathematical Reasoning', fontweight='bold', fontsize=16)
ax1.set_xticks(x_positions)  # 将刻度放在柱子中间
ax1.set_xticklabels(methods, fontsize=15, rotation=20)
ax1.set_xlim(-0.25, len(methods) - 0.25)  # 调整x轴范围，给两边留一点空间
ax1.set_ylim(0, 50)  # 设置y轴范围

# 添加网格线
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 第二个子图 - 不同微调数据集上的表现 (Logic Reasoning)
# 数据分组
groups = {
    'Causal Understanding': ['Qwen2.5-Math-7B', 'SFT (1 ex)', 'CFT (1 ex)'],
    'DisambiguationQA': ['Qwen2.5-Math-7B', 'SFT (1 ex)', 'CFT (1 ex)'],
    'Time Arithmetic': ['Qwen2.5-Math-7B', 'SFT (1 ex)', 'CFT (1 ex)']
}

# 平均分数数据
average_scores = {
    'Causal Understanding': [10.5, 13.7, 25.2],
    'DisambiguationQA': [10.5, 10.6, 20.4],
    'Time Arithmetic': [10.5, 13.4, 26.4]
}

# 设置条形的宽度和位置
bar_width = 0.25
x = np.arange(len(groups))

# 定义颜色
method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 为每个模型类型创建条形
for i, model_type in enumerate(['Qwen2.5-Math-7B', 'SFT (1 ex)', 'CFT (1 ex)']):
    scores = [average_scores[group][i] for group in groups.keys()]
    position = x - bar_width + i * bar_width
    bars = ax2.bar(position, scores, bar_width,
                   label=model_type,
                   color=method_colors[i],
                   alpha=0.8)

    # 在条形上方添加数值标签
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.8,
                 f'{height}',
                 ha='center', va='bottom', fontsize=15, fontweight='bold')

# 添加标题和轴标签
ax2.set_title('1-shot CFT on Logic Reasoning', fontweight='bold', fontsize=18)
ax2.set_ylabel('Average Score (%)', fontweight='bold', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(groups.keys(), fontsize=15)
ax2.set_ylim(0, 30)  # 设置y轴范围

# 添加网格线
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 将图例放在图内部 (右上角并往右移)
ax2.legend(loc='upper right', fontsize=16, bbox_to_anchor=(0.87, 0.99))

# 调整布局
fig.tight_layout()

# 保存为PNG图片
plt.savefig('teaser_figure.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()