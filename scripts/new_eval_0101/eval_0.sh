#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd ../../math_eval

model_sub_dir="output_models/CriticCoT_correct_only_data_1231/checkpoint-723"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_correct_only_1231_ckpt_723"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_correct_only_data_1231/checkpoint-1446"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_correct_only_1231_ckpt_1446"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_correct_only_data_1231/checkpoint-2169"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_correct_only_1231_ckpt_2169"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-1269"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_1269/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-2538"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_2538/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_critic_add_50k_correct_data_1231/checkpoint-3807"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_critic_add_50k_correct_1231_ckpt_3807/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_critic_data_1231/checkpoint-1171"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_critic_1231_ckpt_1171/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_critic_data_1231/checkpoint-2342"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_critic_1231_ckpt_2342/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_critic_data_1231/checkpoint-3513"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_critic_1231_ckpt_3513/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_qwq_critic_data_1229/checkpoint-10000"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_qwq_critic_1229_ckpt_10000/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_qwq_data_1229/checkpoint-10000"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_qwq_1229_ckpt_10000/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_qwq_critic_data_1229/checkpoint-8000"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_qwq_critic_1229_ckpt_8000/"
bash eval_math.sh $model_sub_dir $output_dir


model_sub_dir="output_models/CriticCoT_qwq_data_1229/checkpoint-8000"
output_dir="../math_eval_result_new/eval_res_0101/qwen2.5_7B_qwq_1229_ckpt_8000/"
bash eval_math.sh $model_sub_dir $output_dir