#!/bin/bash
set -ex

#source /gpfs/public/research/miniconda3/bin/activate
#conda activate lf_yubo
#
#export CUDA_VISIBLE_DEVICES=0

cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
# 接收两个参数：models_dir 和 summary_path
if [ $# -ne 2 ]; then
    echo "Usage: $0 <models_dir> <summary_path>"
    exit 1
fi

models_dir=$1
summary_path=$2

# 获取models_dir的文件夹名称（不是完整路径）
models_dir_name=$(basename "$models_dir")

# 获取summary_path的上一层目录
summary_parent_dir=$(dirname "$summary_path")

for checkpoint_dir in ${models_dir}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then  # 确保是目录
        # 获取checkpoint号码
        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)

        # 检查checkpoint_num是否[0, 30)
        if [ "$checkpoint_num" -ge 10 ] && [ "$checkpoint_num" -lt 200 ]; then
            # 设置输出目录
            output_dir="${summary_parent_dir}/${models_dir_name}-checkpoint-${checkpoint_num}/"

            echo "Processing checkpoint-${checkpoint_num}"
            echo "Model path: ${checkpoint_dir}"
            echo "Output dir: ${output_dir}"

            # 执行评估脚本
            bash eval_full.sh "$checkpoint_dir" "$output_dir" "$summary_path"
        else
            echo "Skipping checkpoint-${checkpoint_num} as it's >= 200"
        fi
    fi
done