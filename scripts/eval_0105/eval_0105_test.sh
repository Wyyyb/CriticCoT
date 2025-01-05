#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval


summary_path="../math_eval_result_v1_test/summary_0105_test.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"


model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B"
output_dir="../math_eval_result_v1_job/eval_res_0105/ori_qwen2.5_7B/"
bash eval_math_test_0105.sh $model_path $output_dir $summary_path


model_path="/gpfs/public/research/xy/yubowang/models/Qwen2.5-Math-7B"
output_dir="../math_eval_result_v1_job/eval_res_0105/ori_qwen2.5_math_7B/"
bash eval_math_test_0105.sh $model_path $output_dir $summary_path

