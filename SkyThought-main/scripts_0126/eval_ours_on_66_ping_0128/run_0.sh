#!/bin/bash
#source /map-vepfs/miniconda3/bin/activate
#conda activate yubo_lf
conda activate llamaFactory

cd /data/yubo/CriticCoT/SkyThought-main/skythought/tools
export CUDA_VISIBLE_DEVICES=0,1,2,3
model_path="/data/yubo/models/Qwen2.5-32B-Instruct-Critique-0128"
python eval.py --model ${model_path} --evals=AIME,GPQADiamond --tp=4 --output_file="Qwen2.5-32B-Instruct-Critique-0128_results.txt" --temperatures 0.7
