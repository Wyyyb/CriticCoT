#!/bin/bash
set -ex

#source /gpfs/public/research/miniconda3/bin/activate
#conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
# cd /gpfs/public/research/xy/yubowang/CriticCoT/exp_scritps_0126/eval_32B_models
cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
summary_path="/data/yubo/CriticCoT/0128_eval_results_Qwen2.5-32B-Instruct/summary.txt"
output_dir="/data/yubo/CriticCoT/0128_eval_results_Qwen2.5-32B-Instruct"
checkpoint_dir="/data/yubo/models/Qwen2.5-32B-Instruct"

#bash eval_compe.sh "$checkpoint_dir" "$output_dir" "$summary_path"
bash eval_math.sh "$checkpoint_dir" "$output_dir" "$summary_path"
