#!/bin/bash
set -ex  # 这很好，可以显示执行的命令并在出错时退出

# 激活conda环境
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 切换到正确的目录
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"
# 定义基础变量
BASE_DIR="../math_eval_result_0112_critic"
summary_path="${BASE_DIR}/summary_0112_test_critic.txt"

# 确保输出目录存在
mkdir -p "${BASE_DIR}"

model_sub_dir="qwen2.5-math-7B_critic_1231-0109/checkpoint-200"
model_path="${model_dir}/${model_sub_dir}"
# 使用循环来处理不同的temperature值
for temp in 0 0.3 0.6 0.9; do
    # 替换小数点，使其适合作为目录名
    temp_str=$(echo $temp | tr '.' '_')
    output_dir="${BASE_DIR}/qwen_eval_res_0112_test_critic/qwen2.5-math-7B_critic_1231-0109-checkpoint-200-t_${temp_str}/"

    echo "Running evaluation with temperature ${temp}"
    bash eval_math_critic.sh "$model_path" "$output_dir" "$summary_path" "$temp"
done


model_sub_dir="qwen2.5-math-7B_critic_1231-0109/checkpoint-400"
model_path="${model_dir}/${model_sub_dir}"
# 使用循环来处理不同的temperature值
for temp in 0 0.3 0.6 0.9; do
    # 替换小数点，使其适合作为目录名
    temp_str=$(echo $temp | tr '.' '_')
    output_dir="${BASE_DIR}/qwen_eval_res_0112_test_critic/qwen2.5-math-7B_critic_1231-0109-checkpoint-400-t_${temp_str}/"

    echo "Running evaluation with temperature ${temp}"
    bash eval_math_critic.sh "$model_path" "$output_dir" "$summary_path" "$temp"
done

