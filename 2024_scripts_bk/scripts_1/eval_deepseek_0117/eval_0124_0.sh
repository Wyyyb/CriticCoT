#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_1/eval_deepseek_0117/

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/math_eval_result_0117_dense_deepseek_math_eval_0124/deepseek_math_output_models_summary_0.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/deepseek_math_output_models"

find "$root_dir" -type d -name "CriticCoT_critic_data_0114" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_deepseek_models_math.sh $model_dir $summary_path
done
