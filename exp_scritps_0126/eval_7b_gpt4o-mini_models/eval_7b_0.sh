#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_7b_gpt4o-mini_models

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/0127_eval_results_qwen2.5-math-7B-WebInstruct_40k_critique_gpt-4o-mini_0127/summary.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0127/qwen2.5-math-7B-WebInstruct_40k_critique_gpt-4o-mini_0127"

bash eval_dir_models_math.sh $model_dir $summary_path

