#!/bin/bash
set -ex

#source /gpfs/public/research/miniconda3/bin/activate
#conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=1
# cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_32B_models
cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
summary_path="/data/yubo/CriticCoT/0129_eval_results_CFT-Webinstruct-0121-ckpt-t_1/summary.txt"
output_dir="/data/yubo/CriticCoT/0129_eval_results_CFT-Webinstruct-0121-ckpt-t_1"
checkpoint_dir="/data/yubo/models/CFT-Webinstruct-0121-ckpt"

#bash eval_compe.sh "$checkpoint_dir" "$output_dir" "$summary_path"
bash eval_math_temp.sh "$checkpoint_dir" "$output_dir" "$summary_path" 0.1
