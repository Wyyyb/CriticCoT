#!/bin/bash
set -ex

# 激活conda环境
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 切换到正确的目录
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0114"
# 定义基础变量
BASE_DIR="../math_multi_eval_result_0116"

# 确保输出目录存在
mkdir -p "${BASE_DIR}"


ckpt="checkpoint-380"
model_sub_dir="qwen2.5-math-7B_t2_critic_0114/${ckpt}"
model_path="${model_dir}/${model_sub_dir}"
# 使用循环来处理不同的temperature值
for temp in 0.9 0.1; do
    # 替换小数点，使其适合作为目录名
    temp_str=$(echo $temp | tr '.' '-')
    output_dir="${BASE_DIR}/qwen_eval_res_0116_multi_result/qwen2.5-math-7B_t2_critic_test_0114-${ckpt}-t_${temp_str}/"
    summary_path="${BASE_DIR}/summary_0116_${ckpt}_t-${temp_str}.txt"
    echo "Running evaluation with temperature ${temp}"
    bash eval_math_multiple_results.sh "$model_path" "$output_dir" "$summary_path" "$temp"
done

