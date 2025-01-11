#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=1
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh


summary_path="../math_eval_result_0111/ckpts_summary_0111_others.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"

find "$root_dir" -type d -name "qwen2.5*" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_others.sh $model_dir $summary_path
done