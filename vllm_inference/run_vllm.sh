#!/bin/bash
set -ex

# 激活conda环境
source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1

python qwen_math_run_numina_80k_0119.py

