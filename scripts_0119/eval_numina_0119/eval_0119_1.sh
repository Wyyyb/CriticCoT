#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /gpfs/public/research/xy/yubowang/CriticCoT/scripts_0119/eval_numina_0119

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/math_eval_result_qwen2.5-math-7B_numina_sft_80k_0119/summary.txt"
root_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0119"

find "$root_dir" -type d -name "qwen2.5-math-7B_numina_sft_80k_0119" | while read -r model_dir; do
  echo $model_dir
  bash eval_dir_models_math.sh $model_dir $summary_path
done
