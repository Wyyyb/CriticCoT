#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts/eval_0114_dense/


summary_path="../math_eval_result_0114/ckpts_summary_0114_dense_p3_new.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_dense_0112_qwen_math"

find "$root_dir" -type d -name "qwen2.5-math-7B_critic_1231-0111" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math.sh $model_dir $summary_path
done