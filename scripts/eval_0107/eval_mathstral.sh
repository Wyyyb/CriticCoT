#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval

n_shot=5
summary_path="../math_eval_result_0107/summary_mathstral_0107_job.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"


model_path="/gpfs/public/research/xy/yubowang/models/Mathstral-7B-v0.1"
output_dir="../math_eval_result_0107/eval_res_0107_mathstral_job/ori_Mathstral-7B-v0.1/"
bash eval_full.sh $model_path $output_dir $summary_path $n_shot


model_sub_dir="Mathstral_math_output_models/CriticCoT_critic_data_0106/checkpoint-1171"
output_dir="../math_eval_result_0107/eval_res_0107_mathstral_job/Mathstral_7B_critic_0106_ckpt_1171/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_full.sh $model_path $output_dir $summary_path $n_shot


model_sub_dir="Mathstral_math_output_models/CriticCoT_correct_only_data_0106/checkpoint-723"
output_dir="../math_eval_result_0107/eval_res_0107_mathstral_job/Mathstral_7B_correct_only_0106_ckpt_723/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_full.sh $model_path $output_dir $summary_path $n_shot


model_sub_dir="Mathstral_math_output_models/CriticCoT_correct_only_data_0106/checkpoint-1446"
output_dir="../math_eval_result_0107/eval_res_0107_mathstral_job/Mathstral_7B_correct_only_0106_ckpt_1446/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_full.sh $model_path $output_dir $summary_path $n_shot


model_sub_dir="Mathstral_math_output_models/CriticCoT_correct_only_data_0106/checkpoint-2169"
output_dir="../math_eval_result_0107/eval_res_0107_mathstral_job/Mathstral_7B_correct_only_0106_ckpt_2169/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_full.sh $model_path $output_dir $summary_path $n_shot


model_sub_dir="Mathstral_math_output_models/CriticCoT_critic_add_50k_correct_data_0106/checkpoint-1269"
output_dir="../math_eval_result_0107/eval_res_0107_mathstral_job/Mathstral_7B_critic_add_50k_correct_0106_ckpt_1269/"
model_path="${model_dir}/${model_sub_dir}"
bash eval_full.sh $model_path $output_dir $summary_path $n_shot




