#!/bin/bash
#source /gpfs/public/research/miniconda3/bin/activate
#conda activate lf_yubo
export CUDA_VISIBLE_DEVICES=0
cd /data/yubowang/CriticCoT/math-evaluation-harness
datasets="gsm8k,minerva_math"

mkdir -p ../math_eval_result/eval_res_1231_ori/

model_path="/data/yubowang/models/OLMo-2-1124-7B"
output_dir="../math_eval_result/eval_res_1231_ori/ori_OLMo-2-1124-7B"
bash scripts/run_eval.sh cot $model_path $output_dir $datasets

model_path="/data/yubowang/models/deepseek-math-7b-base"
output_dir="../math_eval_result/eval_res_1231_ori/ori_deepseek-math-7b-base"
bash scripts/run_eval.sh cot $model_path $output_dir $datasets




