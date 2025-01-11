#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh


summary_path="../math_eval_result_0111/ckpts_summary_0111_math.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models"

find "$root_dir" -type d -name "qwen2.5*" | while read -r qwen_dir; do
  model_dir = "$root_dir/$qwen_dir"
  echo $model_dir
  bash eval_dir_models_math.sh $model_dir $summary_path

done