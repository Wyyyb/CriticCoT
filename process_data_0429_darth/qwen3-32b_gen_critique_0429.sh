#!/bin/bash

# 激活您的环境（如果有）
#source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
#conda activate cft

# 设置CUDA可见设备
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /data/yubowang/CriticCoT/process_data_0429_darth
# 运行您的Python脚本，并将标准输出和标准错误重定向到指定文件
python -u qwen3_32b_gen_critique_0429.py

