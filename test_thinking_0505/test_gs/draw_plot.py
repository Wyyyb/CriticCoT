import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def convert_to_relative_days(data_dict):
    # 获取第一个日期作为参考点
    start_date = datetime.strptime(min(data_dict.keys()), '%Y-%m-%d')

    # 创建新字典，键为相对天数
    relative_dict = {}

    for date_str, value in data_dict.items():
        current_date = datetime.strptime(date_str, '%Y-%m-%d')
        days_diff = (current_date - start_date).days + 1  # +1 使得第一天为1而不是0
        relative_dict[days_diff] = value

    return relative_dict


def plot_weekly_citations(data_dict):
    # 将原始日期转换为相对天数
    relative_data = convert_to_relative_days(data_dict)

    # 创建DataFrame
    df = pd.DataFrame(list(relative_data.items()), columns=['day', 'citations'])

    # 按周分组
    df['week'] = ((df['day'] - 1) // 7) + 1  # 计算周数，第一周为1

    # 对每周取最大值
    weekly_data = df.groupby('week').agg({
        'citations': 'max',
        'day': 'max'  # 取每周的最后一天
    }).reset_index()

    # 计算每周的增量
    weekly_data['prev_citations'] = weekly_data['citations'].shift(1).fillna(0)
    weekly_data['weekly_increase'] = weekly_data['citations'] - weekly_data['prev_citations']

    # 创建图形
    plt.figure(figsize=(15, 8))
    bars = plt.bar(weekly_data['week'], weekly_data['weekly_increase'], width=0.7, color='skyblue')

    # 添加累计曲线（使用右y轴）
    ax2 = plt.twinx()
    ax2.plot(df['week'].unique(), df.groupby('week')['citations'].max(), 'r-', linewidth=2, label='累计引用量')
    ax2.set_ylabel('累计引用量', color='r', fontsize=12)

    # 设置图表标题和标签
    plt.title('每周引用量增长（相对于第一天2024-09-09）', fontsize=16)
    plt.xlabel('周数', fontsize=12)
    plt.ylabel('每周增加的引用量', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 设置x轴刻度
    plt.xticks(weekly_data['week'])

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # 只在正值上添加标签
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # 为x轴添加每周对应的天数标签
    week_day_labels = [f"{week}周\n(第{day}天)" for week, day in zip(weekly_data['week'], weekly_data['day'])]
    plt.gca().set_xticklabels(week_day_labels)

    # 添加图例
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 紧凑布局
    plt.tight_layout()

    # 显示图表
    plt.show()

    # 返回处理后的数据字典，方便后续使用
    return relative_data, weekly_data


# 示例数据字典
data = {
    "2024-09-09": 69,
    "2024-09-29": 77,
    "2024-10-16": 91,
    "2024-10-18": 100,
    "2024-10-21": 108,
    "2024-10-25": 114,
    "2024-10-26": 123,
    "2024-11-03": 132,
    "2024-11-13": 145,
    "2024-11-24": 156,
    "2024-12-02": 163,
    "2024-12-08": 176,
    "2024-12-19": 185,
    "2024-12-29": 196,
    "2025-01-12": 211,
    "2025-01-20": 224,
    "2025-01-31": 241,
    "2025-02-05": 254,
    "2025-02-13": 273,
    "2025-02-23": 301,
    "2025-02-28": 323,
    "2025-03-03": 333,
    "2025-03-09": 356,
    "2025-03-16": 372,
    "2025-03-22": 401,
    "2025-03-28": 414,
    "2025-04-03": 427,
    "2025-04-07": 444,
    "2025-04-14": 459,
    "2025-04-17": 464,
    "2025-04-20": 483,
    "2025-04-29": 501,
    "2025-05-05": 513,
    "2025-05-14": 526,
    "2025-05-20": 540,
    "2025-05-24": 574,
    "2025-05-27": 595,
    "2025-05-29": 606,
    "2025-05-30": 633,
    "2025-05-31": 642,
    "2025-06-04": 649,
    "2025-06-05": 660,
    "2025-06-06": 683,
    "2025-06-07": 692,
    "2025-06-09": 704,
    "2025-06-11": 711,
    "2025-06-13": 723,
    "2025-06-15": 729,
    "2025-06-18": 746,
    "2025-06-19": 756,
    "2025-06-22": 783,
    "2025-06-26": 786,
    "2025-06-29": 797,
    "2025-07-04": 814,
    "2025-07-07": 829
}

# 使用函数
relative_data, weekly_data = plot_weekly_citations(data)

# 打印转换后的相对日期字典
print("相对日期字典:")
print(relative_data)