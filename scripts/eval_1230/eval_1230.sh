#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness

mkdir -p ../math_eval_result/eval_res_1230/

model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B"
output_dir="../math_eval_result/eval_res_1230/ori_qwen2.5_7B/"
bash scripts/run_eval.sh cot $model_path $output_dir


model_sub_dir="CriticCoT_critic_data_1228/checkpoint-800"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1230/qwen2.5_7B_critic_1228_ckpt_800/"
bash scripts/run_eval.sh cot $model_path $output_path

model_sub_dir="CriticCoT_critic_data_1228/checkpoint-1600"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1230/qwen2.5_7B_critic_1228_ckpt_1600/"
bash scripts/run_eval.sh cot $model_path $output_path

model_sub_dir="CriticCoT_critic_data_1228"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1230/qwen2.5_7B_critic_1228/"
bash scripts/run_eval.sh cot $model_path $output_path


model_sub_dir="CriticCoT_correct_only_data_1228/checkpoint-400"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1230/qwen2.5_7B_correct_only_1228_ckpt_400/"
bash scripts/run_eval.sh cot $model_path $output_path


model_sub_dir="CriticCoT_correct_only_data_1228/checkpoint-800"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1230/qwen2.5_7B_correct_only_1228_ckpt_800/"
bash scripts/run_eval.sh cot $model_path $output_path


model_sub_dir="CriticCoT_correct_only_data_1228/checkpoint-1200"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1230/qwen2.5_7B_correct_only_1228_ckpt_1200/"
bash scripts/run_eval.sh cot $model_path $output_path



