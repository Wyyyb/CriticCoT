#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /cpfs/data/user/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/multi_eval_math_scripts/

summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_qwen2.5-math-7B_webinstruct_cft_80k_0121_p3/summary.txt"
model_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0124/qwen-math-7b_Numina_80k_critique_gpt-4o-1120_0125"

bash eval_4_card_start.sh $summary_path $model_dir
