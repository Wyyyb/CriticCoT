#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=1
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh


models_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models/qwen2.5-math-7B_critic_numina_860k_0110"
summary_path="../math_eval_result_0111/ckpts_summary_0112_numina_860k_full.txt"

# 获取models_dir的文件夹名称（不是完整路径）
models_dir_name=$(basename "$models_dir")

# 获取summary_path的上一层目录
summary_parent_dir=$(dirname "$summary_path")

# 遍历所有以checkpoint开头的文件夹
for checkpoint_dir in ${models_dir}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then  # 确保是目录
        # 获取checkpoint号码
        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)

        # 设置输出目录
        output_dir="${summary_parent_dir}/${models_dir_name}-checkpoint-${checkpoint_num}/"

        echo "Processing checkpoint-${checkpoint_num}"
        echo "Model path: ${checkpoint_dir}"
        echo "Output dir: ${output_dir}"

        # 执行评估脚本
        bash eval_full.sh "$checkpoint_dir" "$output_dir" "$summary_path"
    fi
done