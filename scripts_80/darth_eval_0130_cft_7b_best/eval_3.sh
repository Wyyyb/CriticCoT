#!/bin/bash
set -ex

#source /gpfs/public/research/miniconda3/bin/activate
#conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=7
# cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_32B_models
cd /data/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
summary_path="/data/yubowang/CriticCoT/0129_eval_results_CFT-Webinstruct-0121-ckpt-t_0_8_aime/summary.txt"
output_dir="/data/yubowang/CriticCoT/0129_eval_results_CFT-Webinstruct-0121-ckpt-t_0_8_aime"
checkpoint_dir="/data/yubowang/models/CFT-Webinstruct-0121-ckpt"

#bash eval_compe.sh "$checkpoint_dir" "$output_dir" "$summary_path"
bash eval_aime_temp.sh "$checkpoint_dir" "$output_dir" "$summary_path" 0.8
