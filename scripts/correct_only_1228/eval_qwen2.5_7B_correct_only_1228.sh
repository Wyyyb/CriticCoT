#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd /gpfs/public/research/xy/yubowang/CriticCoT/math-evaluation-harness
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"
model_path="CriticCoT_correct_only_data_1228/checkpoint-400"
bash scripts/run_eval.sh cot "${model_dir}/${model_path}"


model_path="CriticCoT_correct_only_data_1228/checkpoint-800"
bash scripts/run_eval.sh cot "${model_dir}/${model_path}"


model_path="CriticCoT_correct_only_data_1228/checkpoint-1200"
bash scripts/run_eval.sh cot "${model_dir}/${model_path}"


model_path="CriticCoT_correct_only_data_1228"
bash scripts/run_eval.sh cot "${model_dir}/${model_path}"



