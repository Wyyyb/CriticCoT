#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval


model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"

model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B"
output_dir="../math_eval_result_new/eval_res_0105_v1/ori_qwen2.5_7B/"
bash eval_math_v2.sh $model_path $output_dir


model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B"
output_dir="../math_eval_result_new/eval_res_0105_v1/ori_qwen2.5_math_7B/"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_critic_data_1231/checkpoint-1171"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_critic_1231_ckpt_1171/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_critic_data_1231/checkpoint-2342"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_critic_1231_ckpt_2342/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_critic_data_1231/checkpoint-3513"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_critic_1231_ckpt_3513/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-1269"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_critic_add_50k_correct_1231_ckpt_1269/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-2538"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_critic_add_50k_correct_1231_ckpt_2538/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-3807"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_critic_add_50k_correct_1231_ckpt_3807/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_correct_only_data_1231/checkpoint-723"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_correct_only_1231_ckpt_723"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_correct_only_data_1231/checkpoint-1446"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_correct_only_1231_ckpt_1446"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="qwen_math_output_models/CriticCoT_correct_only_data_1231/checkpoint-2169"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_math_7B_correct_only_1231_ckpt_2169"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_critic_data_1231/checkpoint-1171"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_critic_1231_ckpt_1171/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_critic_data_1231/checkpoint-2342"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_critic_1231_ckpt_2342/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_critic_data_1231/checkpoint-3513"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_critic_1231_ckpt_3513/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-1269"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_1269/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-2538"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_2538/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-3807"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_3807/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_correct_only_data_1231/checkpoint-723"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_correct_only_1231_ckpt_723"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_correct_only_data_1231/checkpoint-1446"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_correct_only_1231_ckpt_1446"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_correct_only_data_1231/checkpoint-2169"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_correct_only_1231_ckpt_2169"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_qwq_critic_data_1229/checkpoint-8000"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_qwq_critic_1229_ckpt_8000/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir


model_sub_dir="output_models/CriticCoT_qwq_data_1229/checkpoint-8000"
output_dir="../math_eval_result_new/eval_res_0105_v1/qwen2.5_7B_qwq_1229_ckpt_8000/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_math_v2.sh $model_path $output_dir

