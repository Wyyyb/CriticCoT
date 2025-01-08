#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness-main-new/scripts


summary_path="../harness_math_eval_result_0108_debug/summary_0108_debug.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"

model_sub_dir="output_models/CriticCoT_correct_only_data_1231/checkpoint-723"
output_dir="../harness_math_eval_result_0108_debug/qwen_eval_res_0108_new_math/qwen2.5_7B_correct_only_1231_ckpt_723"
model_path="${model_dir}/${model_sub_dir}"
bash run_eval_0108.sh $model_path $output_dir $summary_path

