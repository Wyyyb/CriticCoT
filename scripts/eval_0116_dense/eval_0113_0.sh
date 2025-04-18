#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts/eval_0116_dense/

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_eval_result_0116_deepseek/deepseek_ckpts_summary_0116_dense_p0_new.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/deepseek_math_output_models"

find "$root_dir" -type d -name "CriticCoT_correct_only_data_0114" | while read -r model_dir; do
  echo $model_dir
  bash deepseek_eval_dir_models_math.sh $model_dir $summary_path
done
