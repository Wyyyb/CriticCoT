#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
# cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_32B_models
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/0127_eval_results_Qwen_32B_Instruct/summary.txt"
output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/0127_eval_results_Qwen_32B_Instruct"
checkpoint_dir="/gpfs/public/research/xy/yubowang/models/Qwen2.5-32B-Instruct"

bash eval_full.sh "$checkpoint_dir" "$output_dir" "$summary_path"
