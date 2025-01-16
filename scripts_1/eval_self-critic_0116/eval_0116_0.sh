#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_1/eval_self-critic_0116/

summary_path="../math_eval_result_self-critic_0116/ckpts_summary_0116_dense.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0116"

find "$root_dir" -type d -name "qwen2.5-math-7B_self_critique_ckpt-80_MATH-TRAIN-8_data_0116" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math.sh $model_dir $summary_path
done
