#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gpfs/public/research/xy/yubowang/CriticCoT/Qwen2.5-Math-Eval/evaluation/sh

checkpoint_dir=""
output_dir=""
summary_path=""

bash eval_mathtral_full.sh "$checkpoint_dir" "$output_dir" "$summary_path"
