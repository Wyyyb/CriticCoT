#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0
cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness-main-new/scripts


summary_path="../harness_math_eval_result_0108_part2_dev/summary_0108_part2.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"


model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B-Instruct"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/ori_Qwen2.5-Math-7B-Instruct/"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_critic_data_1228/checkpoint-800"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/qwen2.5_7B_critic_1228_ckpt_800/"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_critic_data_1228/checkpoint-1600"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/qwen2.5_7B_critic_1228_ckpt_1600/"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_critic_data_1228/checkpoint-2340"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/qwen2.5_7B_critic_1228_ckpt_2340/"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_qwq_critic_data_1229/checkpoint-8000"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/qwen2.5_7B_qwq_critic_1229_ckpt_8000/"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_qwq_data_1229/checkpoint-8000"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/qwen2.5_7B_qwq_1229_ckpt_8000/"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_path="/gpfs/public/research/xy/yubowang/models/deepseek-math-7b-base"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/ori_deepseek-math-7b-base/"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_path="/gpfs/public/research/xy/yubowang/models/Mathstral-7B-v0.1"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/ori_Mathstral-7B-v0.1/"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_correct_only_data_1228/checkpoint-400"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/qwen2.5_7B_correct_only_1228_ckpt_400/"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_correct_only_data_1228/checkpoint-800"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/qwen2.5_7B_correct_only_1228_ckpt_800/"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_correct_only_data_1228/checkpoint-1200"
output_dir="../harness_math_eval_result_0108_part2_dev/eval_res_0108_new_math/qwen2.5_7B_correct_only_1228_ckpt_1200/"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path
