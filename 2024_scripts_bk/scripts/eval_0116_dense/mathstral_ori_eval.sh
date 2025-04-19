#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=1
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh

checkpoint_dir="/gpfs/public/research/xy/yubowang/models/Mathstral-7B-v0.1"
output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_eval_result_0116_ori_Mathstral"
summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/math_eval_result_0116_Mathstral/ori_Mathstral_ckpts_summary_0116.txt"

bash eval_mathtral_full.sh "$checkpoint_dir" "$output_dir" "$summary_path"
