#!/bin/bash
set -ex

source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

export CUDA_VISIBLE_DEVICES=1
cd /map-vepfs/yubo/CriticCoT/map_scripts_0128/eval_o1_mini_0203/

summary_path="/map-vepfs/yubo/CriticCoT/0203_eval_results_qwen2.5-math-7B-webinstruct_cft_40k_o1_mini_brief/summary.txt"
model_dir="/map-vepfs/yubo/CriticCoT/LLaMA-Factory/output_models_0203/qwen2.5-math-7B-webinstruct_cft_40k_o1_mini_brief"

bash eval_dir_models_math_2.sh $model_dir $summary_path

