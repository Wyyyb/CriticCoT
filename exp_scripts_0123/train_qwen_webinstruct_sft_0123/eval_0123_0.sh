#!/bin/bash
set -ex

source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /cpfs/data/user/yubowang/CriticCoT/ali_scripts/train_qwen_webinstruct_sft_0123/

summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_qwen-7b_webinstruct_ori_sft_80k_0123/summary.txt"
root_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0123"

find "$root_dir" -type d -name "qwen-7b_webinstruct_ori_sft_80k_0123" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math.sh $model_dir $summary_path
done
