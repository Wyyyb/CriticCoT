#!/bin/bash
set -ex

#source /gpfs/public/research/miniconda3/bin/activate
#conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_32B_models
cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
summary_path="/data/yubo/CriticCoT/0129_eval_results_qwen2.5-7b-math-cft-gpt-4o-0128-t_0_5/summary.txt"
output_dir="/data/yubo/CriticCoT/0129_eval_results_qwen2.5-7b-math-cft-gpt-4o-0128-t_0_5"
checkpoint_dir="/data/yubo/models/qwen2.5-7b-math-cft-gpt-4o-0128"

#bash eval_compe.sh "$checkpoint_dir" "$output_dir" "$summary_path"
bash eval_full.sh "$checkpoint_dir" "$output_dir" "$summary_path" 0.5
