#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /cpfs/data/user/yubowang/CriticCoT/ali_scripts/eval_numina_cft_0120/

summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_qwen2.5-math-7B_numina_cft_gpt4o-1120_on-policy_50k_0120_constant-lr/summary.txt"
root_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0120"

find "$root_dir" -type d -name "qwen2.5-math-7B_numina_cft_gpt4o-1120_on-policy_50k_0120_constant-lr" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math.sh $model_dir $summary_path
done
