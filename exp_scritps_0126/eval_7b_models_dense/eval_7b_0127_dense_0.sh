#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/dense_multi_eval_math_scripts_1_step

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/0127_eval_results_qwen2.5-32B-Instruct_webinstruct_cft_10k_0127_dense_constant_with_warmup/summary.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0127/qwen2.5-32B-Instruct_webinstruct_cft_10k_0127_dense_constant_with_warmup"

bash eval_2_card_start.sh $summary_path $model_dir

