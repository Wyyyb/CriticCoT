#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate

conda activate py310

cd /gpfs/public/research/xy/yubowang/CriticCoT/SkyThought-main/skythought/tools
export CUDA_VISIBLE_DEVICES=0,1

summary_path="/gpfs/public/research/xy/yubowang/CriticCoT/0126_eval_results_qwen2.5-32B_t2_critic_0126/summary.txt"
model_dir="/gpfs/public/research/xy/yubowang/CriticCoT/LLaMA-Factory/output_models_0126/qwen2.5-32B_t2_critic_0126"

bash eval_model_dir.sh $model_dir $summary_path
