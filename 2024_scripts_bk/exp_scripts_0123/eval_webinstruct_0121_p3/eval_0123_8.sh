#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=7
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /cpfs/data/user/yubowang/CriticCoT/exp_scripts_0123/eval_webinstruct_0121_p3/

summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_qwen2.5-math-7B_webinstruct_cft_80k_0121_p3_t2/summary.txt"
root_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0121"

find "$root_dir" -type d -name "qwen2.5-math-7B_webinstruct_cft_80k_0121_p3" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math_4.sh $model_dir $summary_path
done
