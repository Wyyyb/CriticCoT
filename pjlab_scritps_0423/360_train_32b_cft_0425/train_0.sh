#!/bin/bash

#source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
#conda activate lf_train

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/360-LLaMA-Factory-sp/
PROJECT_NAME="critic_cot"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=f318ffd0dcf5d31701fd33aee12e57e9cf15444f
export WANDB_PROJECT=$PROJECT_NAME
# export WANDB_MODE=disabled
export WANDB_DISABLED="true"
export MASTER_PORT=$(shuf -i 30000-40000 -n 1)
export MASTER_ADDR="127.0.0.1"

FORCE_TORCHRUN=1 llamafactory-cli train ../pjlab_scritps_0423/360_train_32b_cft_0425/qwen2.5-32b_0424_0.yaml
