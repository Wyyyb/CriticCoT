#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=1
cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval


summary_path="../math_eval_result_0106/summary_0106_ori_models.txt"
n_shot=5
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"

model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B-Instruct"
output_dir="../math_eval_result_0106/eval_res_0106_ori_models/ori_qwen2.5_math_7B_instruct/"
bash eval_full.sh $model_path $output_dir $summary_path $n_shot

n_shot=0
model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B-Instruct"
output_dir="../math_eval_result_0106/eval_res_0106_ori_models/ori_qwen2.5_math_7B_instruct/"
bash eval_full.sh $model_path $output_dir $summary_path $n_shot


