#!/bin/bash

source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

cd /map-vepfs/yubo/CriticCoT/LLaMA-Factory/
PROJECT_NAME="critic_cot"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=f318ffd0dcf5d31701fd33aee12e57e9cf15444f
export WANDB_PROJECT=$PROJECT_NAME
# export WANDB_MODE=disabled
export WANDB_DISABLED="true"
export MASTER_PORT=$(shuf -i 30000-40000 -n 1)
export MASTER_ADDR="127.0.0.1"

FORCE_TORCHRUN=1 llamafactory-cli train ../map_scripts_0206/train_32b_cft_0207_o1-mini_long/qwen2.5-32b-instruct_80k_long_0207_0.yaml
