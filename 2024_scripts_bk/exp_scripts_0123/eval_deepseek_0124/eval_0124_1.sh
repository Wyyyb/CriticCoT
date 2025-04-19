#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /cpfs/data/user/yubowang/CriticCoT/exp_scripts_0123/eval_deepseek_0124/

summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_deepseek-math-7b-base_webinstruct_cft_80k_0121_p3_ori_eval_5shot/summary.txt"
root_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0124"

find "$root_dir" -type d -name "deepseek-math-7b-base_webinstruct_cft_80k_0121_p3" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_deepseek_models_math.sh $model_dir $summary_path
done
