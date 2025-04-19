#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh


summary_path="../math_eval_result_0112/summary_0112_test.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"
model_sub_dir="qwen2.5-math-7B_critic_1231-0109/checkpoint-200"
model_path="${model_dir}/${model_sub_dir}"

n_shot=1
output_dir="../math_eval_result_0112/qwen_eval_res_0112_test_n-shot/qwen2.5-math-7B_critic_1231-0109-checkpoint-200-1-shot/"
bash eval_math_n-shot.sh $model_path $output_dir $summary_path $n_shot

n_shot=2
output_dir="../math_eval_result_0112/qwen_eval_res_0112_test_n-shot/qwen2.5-math-7B_critic_1231-0109-checkpoint-200-2-shot/"
bash eval_math_n-shot.sh $model_path $output_dir $summary_path $n_shot

n_shot=3
output_dir="../math_eval_result_0112/qwen_eval_res_0112_test_n-shot/qwen2.5-math-7B_critic_1231-0109-checkpoint-200-3-shot/"
bash eval_math_n-shot.sh $model_path $output_dir $summary_path $n_shot

n_shot=4
output_dir="../math_eval_result_0112/qwen_eval_res_0112_test_n-shot/qwen2.5-math-7B_critic_1231-0109-checkpoint-200-4-shot/"
bash eval_math_n-shot.sh $model_path $output_dir $summary_path $n_shot

n_shot=5
output_dir="../math_eval_result_0112/qwen_eval_res_0112_test_n-shot/qwen2.5-math-7B_critic_1231-0109-checkpoint-200-5-shot/"
bash eval_math_n-shot.sh $model_path $output_dir $summary_path $n_shot

n_shot=0
output_dir="../math_eval_result_0112/qwen_eval_res_0112_test_n-shot/qwen2.5-math-7B_critic_1231-0109-checkpoint-200-0-shot/"
bash eval_math_n-shot.sh $model_path $output_dir $summary_path $n_shot
