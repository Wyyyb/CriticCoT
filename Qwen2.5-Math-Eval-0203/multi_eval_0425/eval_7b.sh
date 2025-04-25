#!/bin/bash
set -ex

#source /map-vepfs/miniconda3/bin/activate
#conda activate yubo_lf

summary_path=$1
model_dir=$2
start=$3
end=$4

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /map-vepfs/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/multi_eval_0209_dense

#summary_path="/map-vepfs/yubo/CriticCoT/0203_eval_results_qwen2.5-math-7B-webinstruct_cft_40k_o1_mini_brief/summary.txt"
#model_dir="/map-vepfs/yubo/CriticCoT/LLaMA-Factory/output_models_0203/qwen2.5-math-7B-webinstruct_cft_40k_o1_mini_brief"

bash eval_dir_models_math.sh $model_dir $summary_path $start $end
