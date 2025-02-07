#!/bin/bash
set -ex

source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

summary_path="/map-vepfs/yubo/CriticCoT/0207_eval_results_qwen2.5-32B-Instruct-webinstruct_cft_80k_o1_mini_long_0204_0/summary.txt"
model_dir="/map-vepfs/yubo/CriticCoT/LLaMA-Factory/output_models_0207/qwen2.5-32B-Instruct-webinstruct_cft_80k_o1_mini_long_0204_0"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /map-vepfs/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
# 获取models_dir的文件夹名称（不是完整路径）
models_dir_name=$(basename "$models_dir")

# 获取summary_path的上一层目录
summary_parent_dir=$(dirname "$summary_path")

for checkpoint_dir in ${models_dir}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then  # 确保是目录
        # 获取checkpoint号码
        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)

        # 检查checkpoint_num是否[0, 30)
        if [ "$checkpoint_num" -ge 0 ] && [ "$checkpoint_num" -lt 1000 ]; then
            # 设置输出目录
            output_dir="${summary_parent_dir}/${models_dir_name}-checkpoint-${checkpoint_num}/"

            echo "Processing checkpoint-${checkpoint_num}"
            echo "Model path: ${checkpoint_dir}"
            echo "Output dir: ${output_dir}"

            # 执行评估脚本
            bash eval_full.sh "$checkpoint_dir" "$output_dir" "$summary_path"
        else
            echo "Skipping checkpoint-${checkpoint_num}"
        fi
    fi
done