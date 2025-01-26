#!/bin/bash
source /gpfs/public/research/miniconda3/bin/activate

conda activate lf_yubo

cd /gpfs/public/research/xy/yubowang/CriticCoT/SkyThought-main/skythought/tools
export CUDA_VISIBLE_DEVICES=0,1
model_path="/gpfs/public/research/xy/yubowang/models/MAmmoTH-Critique-1"
python eval.py --model ${model_path} --evals=AIME,MATH500,GPQADiamond --tp=2 --output_file=results.txt
