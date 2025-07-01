import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 创建一个包含两个子图的大图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1.3, 0.7]})

# 第一个子图 - 使用表格数据展示不同模型的平均分数
# 模型名称
models = ['Qwen2.5-Math-1.5B', 'Llama3.2-3B-Instruct', 'Qwen2.5-Math-7B', 'Qwen2.5-14B']

# 每个模型的方法对应的平均分数
methods = ['base', 'base (sober)', 'SFT (1 ex)', 'SFT (full)', 'RL (1 ex)', 'CFT (1 ex)']

# 从表格提取的平均分数数据（AVG列）
scores = {
    'Qwen2.5-Math-1.5B': {
        'base': 21.1,
        'base (sober)': 25.0,
        'SFT (1 ex)': 18.5,
        'SFT (full)': 22.2,
        'RL (1 ex)': 33.8,
        'CFT (1 ex)': 32.8
    },
    'Llama3.2-3B-Instruct': {
        'base': 17.5,
        'SFT (1 ex)': 15.4,
        'SFT (full)': 16.5,
        'RL (1 ex)': 19.0,
        'CFT (1 ex)': 21.7
    },
    'Qwen2.5-Math-7B': {
        'base': 27.3,
        'base (sober)': 32.2,
        'SFT (1 ex)': 22.9,
        'SFT (full)': 28.9,
        'RL (1 ex)': 40.2,
        'CFT (1 ex)': 42.2
    },
    'Qwen2.5-14B': {
        'base': 27.1,
        'SFT (1 ex)': 24.6,
        'SFT (full)': 25.8,
        'CFT (1 ex)': 36.0
    }
}

# 设置不同方法的颜色
method_colors = {
    'base': '#1f77b4',
    'base (sober)': '#aec7e8',
    'SFT (1 ex)': '#ff7f0e',
    'SFT (full)': '#ffbb78',
    'RL (1 ex)': '#2ca02c',
    'CFT (1 ex)': '#98df8a'
}

# 为图例创建空的句柄和标签列表
legend_handles = []
legend_labels = []

# 计算每个模型的方法数量和位置
bar_width = 0.5
group_spacing = 0.5
model_positions = []
start_pos = 0

for model in models:
    model_methods = [m for m in methods if m in scores[model]]
    num_methods = len(model_methods)
    total_width = num_methods * bar_width
    model_positions.append(start_pos + total_width / 2)

    # 在模型组内绘制每个方法的条形
    for i, method in enumerate(model_methods):
        method_pos = start_pos + i * bar_width
        score = scores[model][method]
        bar = ax1.bar(method_pos, score, width=bar_width, color=method_colors[method],
                      label="")

        # 为图例收集句柄和标签（只添加一次）
        if method not in legend_labels:
            legend_handles.append(bar)
            legend_labels.append(method)

        # 在条形上方添加数值标签
        ax1.text(method_pos, score + 0.8, f'{score}', ha='center', va='bottom',
                 fontsize=11, fontweight='normal')

    start_pos += total_width + group_spacing

# 添加标签和标题
ax1.set_ylabel('Average Score (%)', fontweight='bold', fontsize=15)
ax1.set_title('Mathematical Reasoning', fontweight='bold', fontsize=18)

# 添加x轴标签（模型名称）
ax1.set_xticks(model_positions)
ax1.set_xticklabels(models, fontsize=13, rotation=10)
ax1.set_ylim(10, 45)  # 设置y轴范围

# 添加网格线
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 将图例分为三部分，每部分2个
legend1 = ax1.legend([legend_handles[0], legend_handles[1]],
                     [legend_labels[0], legend_labels[1]],
                     loc='upper left', fontsize=12, ncol=1,
                     frameon=True, framealpha=0.7)

legend2 = ax1.legend([legend_handles[2], legend_handles[3]],
                     [legend_labels[2], legend_labels[3]],
                     loc='upper left', fontsize=12, ncol=1,
                     bbox_to_anchor=(0.22, 1),
                     frameon=True, framealpha=0.7)

legend3 = ax1.legend([legend_handles[4], legend_handles[5]],
                     [legend_labels[4], legend_labels[5]],
                     loc='upper left', fontsize=12, ncol=1,
                     bbox_to_anchor=(0.415, 1),
                     frameon=True, framealpha=0.7)

# 需要添加第一个和第二个图例，否则会被覆盖
ax1.add_artist(legend1)
ax1.add_artist(legend2)

# 第二个子图 - 使用in-domain数据（对角线高亮的数据）
# 数据分组
groups = {
    'Causal Understanding': ['Base', 'SFT (1 ex)', 'CFT (1 ex)'],
    'DisambiguationQA': ['Base', 'SFT (1 ex)', 'CFT (1 ex)'],
    'Time Arithmetic': ['Base', 'SFT (1 ex)', 'CFT (1 ex)']
}

# 从表格中提取的in-domain分数数据（对角线高亮的数据）
in_domain_scores = {
    'Causal Understanding': [24.0, 27.5, 41.5],
    'DisambiguationQA': [5.0, 9.2, 24.2],
    'Time Arithmetic': [2.5, 5.0, 14.0]
}

# 设置条形的宽度和位置
bar_width = 0.2
x = np.arange(len(groups))

# 定义颜色
method_colors_right = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 为每个模型类型创建条形
for i, model_type in enumerate(['Qwen-2.5-Math-7B', 'SFT (1 ex)', 'CFT (1 ex)']):
    scores = [in_domain_scores[group][i] for group in groups.keys()]
    position = x - bar_width + i * bar_width
    bars = ax2.bar(position, scores, bar_width,
                   label=model_type,
                   color=method_colors_right[i],
                   alpha=0.8)

    # 在条形上方添加数值标签
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.8,
                 f'{height}',
                 ha='center', va='bottom', fontsize=11, fontweight='normal')

# 添加标题和轴标签
ax2.set_title('Logic Reasoning', fontweight='bold', fontsize=18)
ax2.set_ylabel('In-Domain Score (%)', fontweight='bold', fontsize=15)
ax2.set_xticks(x)
ax2.set_xticklabels(groups.keys(), fontsize=13, rotation=10)
ax2.set_ylim(0, 45)  # 设置y轴范围，调整以适应更高的分数

# 添加网格线
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 将图例放在图内部 (右上角并往右移)
ax2.legend(loc='upper right', fontsize=12, bbox_to_anchor=(0.99, 0.99), frameon=True, framealpha=0.7)

# 调整布局
fig.tight_layout()

# 保存为PNG图片
plt.savefig('teaser_figure_4.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()