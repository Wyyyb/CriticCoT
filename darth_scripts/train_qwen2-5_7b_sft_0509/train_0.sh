#!/bin/bash

conda activate cft

cd /data/yubo/CriticCoT/360-LLaMA-Factory-sp/
PROJECT_NAME="critic_cot"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=f318ffd0dcf5d31701fd33aee12e57e9cf15444f
export WANDB_PROJECT=$PROJECT_NAME
# export WANDB_MODE=disabled
export WANDB_DISABLED="true"
export MASTER_PORT=$(shuf -i 30000-40000 -n 1)
export MASTER_ADDR="127.0.0.1"

FORCE_TORCHRUN=1 llamafactory-cli train ../darth_scripts/train_qwen2-5_7b_sft_0509/qwen2.5-7b_wv_sft.yaml

