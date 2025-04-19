#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /cpfs/data/user/yubowang/CriticCoT/exp_scripts_0123/train_cft_2models_and_eval_0124/

summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_qwen-7b-base_webinstruct_cft_80k_0121_p3/summary.txt"
root_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0124"

find "$root_dir" -type d -name "qwen-7b-base_webinstruct_cft_80k_0121_p3" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math.sh $model_dir $summary_path
done
