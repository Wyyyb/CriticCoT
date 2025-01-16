#!/bin/bash
set -ex

# 激活conda环境
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd /gpfs/public/research/xy/yubowang/CriticCoT/critique_on_math

input_dir="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_eval_result_self-critic_multi_0116/qwen_eval_res_0116_multi_result/qwen2.5-math-7B_multi_self-critic-0116-checkpoint-80/math/"
summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_eval_result_self-critic_multi_0116/qwen_eval_res_0116_multi_result/qwen2.5-math-7B_multi_self-critic-0116-checkpoint-80/math/summary.txt"

python calculate_multi_critic_accuracy.py \
    --input_dir $input_dir \
    --summary_path $summary_path

