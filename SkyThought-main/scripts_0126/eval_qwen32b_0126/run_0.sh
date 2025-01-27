#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate

conda activate py310

cd /gpfs/public/research/xy/yubowang/CriticCoT/SkyThought-main/skythought/tools
export CUDA_VISIBLE_DEVICES=0,1
model_path="/gpfs/public/research/xy/yubowang/models/Sky-T1-32B-Preview"
python eval.py --model ${model_path} --evals=GPQADiamond --tp=2 --output_file="Sky-T1-32B-Preview_results.txt"
