#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"


model_sub_dir="CriticCoT_correct_only_data_1228/checkpoint-400"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1228/qwen2.5_7B_correct_only_1228_ckpt_400/"
bash scripts/run_eval.sh cot $model_path $output_path


model_sub_dir="CriticCoT_correct_only_data_1228/checkpoint-800"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1228/qwen2.5_7B_correct_only_1228_ckpt_400/"
bash scripts/run_eval.sh cot $model_path $output_path


model_sub_dir="CriticCoT_correct_only_data_1228/checkpoint-1200"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1228/qwen2.5_7B_correct_only_1228_ckpt_400/"
bash scripts/run_eval.sh cot $model_path $output_path


model_sub_dir="CriticCoT_correct_only_data_1228"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1228/qwen2.5_7B_correct_only_1228/"
bash scripts/run_eval.sh cot $model_path $output_path


model_sub_dir="CriticCoT_critic_data_1228/checkpoint-800"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1228/qwen2.5_7B_critic_1228_ckpt_800/"
bash scripts/run_eval.sh cot $model_path $output_path
