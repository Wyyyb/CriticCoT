#!/bin/bash
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

cd /data/yubo/CriticCoT/LLaMA-Factory/
PROJECT_NAME="critic_cot"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=f318ffd0dcf5d31701fd33aee12e57e9cf15444f
export WANDB_PROJECT=$PROJECT_NAME
export WANDB_MODE=disabled
MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="critic-proj"
export WANDB_RUN_NAME=$MODEL_NAME

FORCE_TORCHRUN=1 llamafactory-cli train ../map_scripts_0128/gpt-4o-mini-cft/train_7b_gpt4o-mini/qwen2.5-7B-WebInstruct_40k_critique_gpt-4o-mini_0128.yaml

