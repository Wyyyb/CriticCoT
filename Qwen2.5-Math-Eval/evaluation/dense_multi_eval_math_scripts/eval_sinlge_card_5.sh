#!/bin/bash
set -ex
summary_path=$1
model_dir=$2

#source /cpfs/data/shared/public/miniconda3/bin/activate
#conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=4
# cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/multi_eval_math_scripts

#summary_path="/cpfs/data/user/yubowang/CriticCoT/math_eval_result_qwen2.5-math-7B_webinstruct_cft_80k_0121_p3_t1/summary.txt"
#root_dir="/cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/output_models_0121"

bash eval_dir_models_math_5.sh $model_dir $summary_path

