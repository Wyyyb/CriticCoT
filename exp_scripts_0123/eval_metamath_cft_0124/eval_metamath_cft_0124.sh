#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

# export CUDA_VISIBLE_DEVICES=0
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /cpfs/data/user/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/multi_eval_math_scripts

summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_qwen-math-7b_MetaMath_80k_critique_gpt-4o-1120_0124/summary.txt"
model_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0123/qwen-math-7b_MetaMath_80k_critique_gpt-4o-1120_0124"

bash eval_8_card_start.sh $summary_path $model_dir
