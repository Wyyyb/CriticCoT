#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval


summary_path="../math_eval_result_v1_test/summary_0105_test_theoremqa.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"


model_sub_dir="output_models/CriticCoT_critic_data_1231/checkpoint-2342"
output_dir="../math_eval_result_v1_test/eval_res_0105_theoremqa/qwen2.5_7B_critic_1231_ckpt_2342/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_0105_test_theoremqa.sh $model_path $output_dir $summary_path


model_sub_dir="output_models/CriticCoT_critic_data_1231/checkpoint-3513"
output_dir="../math_eval_result_v1_test/eval_res_0105_theoremqa/qwen2.5_7B_critic_1231_ckpt_3513/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_0105_test_theoremqa.sh $model_path $output_dir $summary_path
