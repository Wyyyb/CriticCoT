#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gpfs/public/research/xy/yubowang/CriticCoT/math_eval_original/

checkpoint_dir="/gpfs/public/research/xy/yubowang/models/deepseek-math-7b-base"
output_dir="/gpfs/public/research/xy/yubowang/CriticCoT/math_eval_result_0117_ori_deepseek_math_eval"
summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/math_eval_result_0117_ori_deepseek_math_eval/ori_deepseek_ckpts_summary_0117_math_eval.txt"

bash eval_full.sh "$checkpoint_dir" "$output_dir" "$summary_path" 5
