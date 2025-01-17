#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_1/eval_dense_0117/

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/math_eval_result_0117_dense_qwen/dense_qwen_t2_summary_0117_math_eval_critic.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0117"

find "$root_dir" -type d -name "qwen2.5-7B_t2_critic_0117" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math.sh $model_dir $summary_path
done

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/math_eval_result_0117_dense_deepseek_math_eval/dense_deepseek_summary_0117_math_eval_critic.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0117"

find "$root_dir" -type d -name "deepseek-math-7B_t2_critic_0117" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_deepseek_models_math.sh $model_dir $summary_path
done

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/math_eval_result_0117_dense_deepseek_math_eval/dense_deepseek_summary_0117_math_eval_correct_only.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/deepseek_math_output_models"

find "$root_dir" -type d -name "CriticCoT_correct_only_data_0114" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_deepseek_models_math.sh $model_dir $summary_path
done

