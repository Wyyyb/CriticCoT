#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_7b_models

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/0127_eval_results_qwen2.5-math-7B-Instruct_webinstruct_cft_80k_0119/summary.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0126/qwen2.5-math-7B-Instruct_webinstruct_cft_80k_0119"

sleep 18000
bash eval_dir_models_math.sh $model_dir $summary_path

