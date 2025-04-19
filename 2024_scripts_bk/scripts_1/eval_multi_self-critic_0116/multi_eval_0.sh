#!/bin/bash
set -ex

# 激活conda环境
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 切换到正确的目录
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0116"
# 定义基础变量
BASE_DIR="../math_eval_result_self-critic_multi_0116"

ckpt="checkpoint-80"
model_sub_dir="qwen2.5-math-7B_self_critique_ckpt-80_MATH-TRAIN-8_data_0116/${ckpt}"
model_path="${model_dir}/${model_sub_dir}"
output_dir="${BASE_DIR}/qwen_eval_res_0116_multi_result/qwen2.5-math-7B_multi_self-critic-0116-${ckpt}/"

temp=0.0
echo "Running evaluation with temperature ${temp}"
temp_str=$(echo $temp | tr '.' '-')
summary_path="${BASE_DIR}/summary_0116_${ckpt}_t-${temp_str}.txt"
bash critique_eval_math.sh "$model_path" "$output_dir" "$summary_path" "$temp" 1

temp=0.6
echo "Running evaluation with temperature ${temp}"
temp_str=$(echo $temp | tr '.' '-')
summary_path="${BASE_DIR}/summary_0116_${ckpt}_t-${temp_str}.txt"
bash critique_eval_math.sh "$model_path" "$output_dir" "$summary_path" "$temp" 4


