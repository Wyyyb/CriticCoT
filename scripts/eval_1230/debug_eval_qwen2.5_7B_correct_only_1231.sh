#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo
export CUDA_VISIBLE_DEVICES=1

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"
datasets="math"

model_sub_dir="CriticCoT_correct_only_data_1228/checkpoint-1200"
model_path="${model_dir}/${model_sub_dir}"
output_path="../math_eval_result/eval_res_1228/debug_qwen2.5_7B_correct_only_1228_ckpt_1200/"
bash scripts/run_eval.sh cot $model_path $output_path $datasets


