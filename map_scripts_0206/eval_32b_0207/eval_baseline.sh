#!/bin/bash
set -ex

source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

cd /map-vepfs/yubo/CriticCoT/SkyThought-main/skythought
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m skythought_evals.eval \
    --model /map-vepfs/yubo/models/Qwen2.5-Coder-32B-Instruct \
    --evals=livecodebench,aime,math500,gpqa_diamond \
    --tp=8 --output_file=results.txt






