#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/dense_multi_eval_math_scripts

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/0127_eval_results_qwen2.5-math-7B-instruct_webinstruct_cft_80k_0119_0127_dense/summary.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0126/qwen2.5-math-7B-instruct_webinstruct_cft_80k_0119_0127_dense"

bash eval_4_card_start.sh $summary_path $model_dir

