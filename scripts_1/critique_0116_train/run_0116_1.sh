#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /gpfs/public/research/xy/yubowang/CriticCoT/critique_on_math/


input_base_dir="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_multi_eval_result_0116"
sub_dir="qwen_eval_res_0116_multi_result/qwen2.5-math-7B_t2_critic_test_0114-checkpoint-80-t_0-1"
input_dir="${input_base_dir}/${sub_dir}"

model_base_path="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0114"
model_sub_path="qwen2.5-math-7B_t2_critic_0114/checkpoint-80"
model_path="${model_base_path}/${model_sub_path}"

python critique_vllm.py \
  --model_path $model_path \
  --input_dir $input_dir


