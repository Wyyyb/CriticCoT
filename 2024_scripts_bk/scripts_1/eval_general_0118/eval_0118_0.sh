#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1
cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval_original/

checkpoint_dir="/gpfs/public/research/xy/yubowang/models/deepseek-math-7b-base"
output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/eval_general_0118"
summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/eval_general_0118/summary_general_0118.txt"

bash eval_full.sh "$checkpoint_dir" "$output_dir" "$summary_path" 5
