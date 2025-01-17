#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh

checkpoint_dir="/gpfs/public/research/xy/yubowang/models/deepseek-math-7b-base"
output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_eval_result_0118_ori_deepseek"
summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_eval_result_0118_ori_deepseek/ori_deepseek_summary_0118.txt"

bash eval_deepseek_full.sh "$checkpoint_dir" "$output_dir" "$summary_path"
