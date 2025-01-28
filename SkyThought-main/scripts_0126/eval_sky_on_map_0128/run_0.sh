#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

cd /map-vepfs/yubo/CriticCoT/SkyThought-main/skythought/tools
export CUDA_VISIBLE_DEVICES=6,7
model_path="/map-vepfs/yubo/models/Sky-T1-32B-Preview"
python eval.py --model ${model_path} --evals=AIME,GPQADiamon --tp=2 --output_file="Sky-T1-32B-Preview_results.txt" --temperatures 0.7
