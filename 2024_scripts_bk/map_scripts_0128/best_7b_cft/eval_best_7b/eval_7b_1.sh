#!/bin/bash
set -ex

source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

export CUDA_VISIBLE_DEVICES=0
cd /map-vepfs/yubo/CriticCoT/map_scripts_0128/best_7b_cft/eval_best_7b

summary_path="/map-vepfs/yubo/CriticCoT/0128_eval_results_qwen2.5-math-7B-webinstruct_cft_80k_0119_0128/summary.txt"
model_dir="/map-vepfs/yubo/CriticCoT/LLaMA-Factory/output_models_0128/qwen2.5-math-7B-webinstruct_cft_80k_0119_0128"

bash eval_dir_models_math_1.sh $model_dir $summary_path
