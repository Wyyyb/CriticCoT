#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=1
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh


summary_path="../math_eval_result_0110/summary_0112_baseline_p1.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_bk"


#model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B"
#output_dir="../math_eval_result_0110/qwen_eval_res_0110_new_math/ori_qwen2.5_7B/"
#bash eval_full.sh $model_path $output_dir $summary_path
#
#
#model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B"
#output_dir="../math_eval_result_0110/qwen_eval_res_0110_new_math/ori_qwen2.5_math_7B/"
#bash eval_full.sh $model_path $output_dir $summary_path
#
#
#model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B-Instruct"
#output_dir="../math_eval_result_0110/qwen_eval_res_0110_new_math/ori_qwen2.5_math_7B_instruct/"
#bash eval_full.sh $model_path $output_dir $summary_path


model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B-Instruct"
output_dir="../math_eval_result_0110/qwen_eval_res_0110_new_math/ori_qwen2.5_7B_instruct/"
bash eval_full.sh $model_path $output_dir $summary_path


model_path="/gpfs/public/research/xy/yubowang/models/deepseek-math-7b-base"
output_dir="../math_eval_result_0110/qwen_eval_res_0110_new_math/ori_deepseek-math-7b-base/"
bash eval_deepseek_math.sh $model_path $output_dir $summary_path

