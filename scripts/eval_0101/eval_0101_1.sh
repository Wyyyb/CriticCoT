#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=1
datasets="math,gsm8k,minerva_math,sat_math,mmlu_stem"

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness
use_ins="ins"

mkdir -p ../math_eval_result/eval_res_0101_ins/

model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"

model_sub_dir="CriticCoT_correct_only_data_1231/checkpoint-723"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101_ins/qwen2.5_7B_correct_only_1231_ckpt_723/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins

model_sub_dir="CriticCoT_correct_only_data_1231/checkpoint-1446"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101_ins/qwen2.5_7B_correct_only_1231_ckpt_1446/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins

model_sub_dir="CriticCoT_correct_only_data_1231/checkpoint-2169"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101_ins/qwen2.5_7B_correct_only_1231_ckpt_2169/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


use_ins="original"

model_sub_dir="CriticCoT_critic_add_50k_correct_data_1230/checkpoint-800"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1230_ckpt_800/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


model_sub_dir="CriticCoT_critic_add_50k_correct_data_1230/checkpoint-1600"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1230_ckpt_1600/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


model_sub_dir="CriticCoT_critic_add_50k_correct_data_1230/checkpoint-2400"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1230_ckpt_2400/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


# qwq model eval

model_sub_dir="CriticCoT_qwq_critic_data_1229/checkpoint-6000"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_qwq_critic_1229_ckpt_6000/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


model_sub_dir="CriticCoT_qwq_critic_data_1229/checkpoint-8000"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_qwq_critic_1229_ckpt_8000/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


model_sub_dir="CriticCoT_qwq_data_1229/checkpoint-6000"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_qwq_1229_ckpt_6000/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins

