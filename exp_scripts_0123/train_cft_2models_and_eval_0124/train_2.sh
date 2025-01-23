#!/bin/bash
source /cpfs/data/shared/public/miniconda3/bin/activate
conda activate lf_yubo

# cd /data/yubo/CriticCoT/LLaMA-Factory/
cd /cpfs/data/user/yubowang/CriticCoT/LLaMA-Factory/
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

FORCE_TORCHRUN=1 llamafactory-cli train ../exp_scripts_0123/train_cft_2models_and_eval_0124/deepseek-math-7B_webinstruct_cft_80k_0123_2.yaml

bash ../exp_scripts_0123/train_cft_2models_and_eval_0124/eval_0124_2.sh
