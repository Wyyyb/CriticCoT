#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0
datasets="math,gsm8k,minerva_math,sat_math,mmlu_stem"

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness
use_ins="original"

mkdir -p ../math_eval_result/eval_res_0101/

model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"

model_sub_dir="CriticCoT_correct_only_data_1231/checkpoint-723"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_correct_only_1231_ckpt_723/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins

model_sub_dir="CriticCoT_correct_only_data_1231/checkpoint-1446"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_correct_only_1231_ckpt_1446/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins

model_sub_dir="CriticCoT_correct_only_data_1231/checkpoint-2169"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_correct_only_1231_ckpt_2169/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins

model_sub_dir="CriticCoT_critic_add_50k_correct_data_1231/checkpoint-1269"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_1269/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins

model_sub_dir="CriticCoT_critic_add_50k_correct_data_1231/checkpoint-2538"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_2538/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins

#model_sub_dir="CriticCoT_critic_add_50k_correct_data_1231/checkpoint-3807"
#model_path="${model_dir}/${model_sub_dir}"
#output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_3807/"
#bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


model_sub_dir="CriticCoT_critic_data_1231/checkpoint-1171"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_1231_ckpt_1171/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


model_sub_dir="CriticCoT_critic_data_1231/checkpoint-2342"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_1231_ckpt_2342/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins


#model_sub_dir="CriticCoT_critic_data_1231/checkpoint-3513"
#model_path="${model_dir}/${model_sub_dir}"
#output_path="../math_eval_result/eval_res_0101/qwen2.5_7B_critic_1231_ckpt_3513/"
#bash scripts/run_eval.sh cot $model_path $output_path $datasets $use_ins






