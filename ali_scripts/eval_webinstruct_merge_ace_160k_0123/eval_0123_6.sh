#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=5
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /cpfs/data/user/yubowang/CriticCoT/ali_scripts/eval_webinstruct_merge_ace_160k_0123/

summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_webinstruct_merge_ace_cft_160k_0123/summary.txt"
root_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0123"

find "$root_dir" -type d -name "webinstruct_merge_ace_cft_160k_0123" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math_6.sh $model_dir $summary_path
done
