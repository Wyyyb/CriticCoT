#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd ../../math_eval

model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"

#model_sub_dir="output_models/CriticCoT_critic_data_1228/checkpoint-800"
#output_dir="../math_eval_result_new/eval_res_0102/qwen2.5_7B_critic_1228_ckpt_800/"
#model_path="${model_dir}/${model_sub_dir}"
#bash eval_math.sh $model_path $output_dir


model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B"
output_dir="../math_eval_result_new/eval_res_0102/ori_qwen2.5_7B/"
bash eval_math.sh $model_path $output_dir


model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B"
output_dir="../math_eval_result_new/eval_res_0102/ori_qwen2.5_math_7B/"
bash eval_math.sh $model_path $output_dir


model_path="/gpfs/public/research/xy/yubowang/models/deepseek-math-7b-base"
output_dir="../math_eval_result_new/eval_res_0102/ori_deepseek-math-7b-base/"
bash eval_math.sh $model_path $output_dir

