#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd ../../

# python -m simple-evals.simple_evals --list-models

model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory"
model_sub_dir="qwen_math_output_models/CriticCoT_critic_data_1231/checkpoint-1171"
model_path="${model_dir}/${model_sub_dir}"

python -m simple-evals.simple_evals --model $model_path --examples 500

