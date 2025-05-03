#!/bin/bash
#SBATCH --partition=belt_road       # 分区名称
#SBATCH --nodes=1                   # 请求1个节点
#SBATCH --ntasks-per-node=1         # 每节点1个任务
#SBATCH --gres=gpu:8                # 请求8个GPU
#SBATCH --mem=512G                   # 请求32GB内存
#SBATCH --time=96:00:00             # 最长运行8小时
#SBATCH --job-name=lf_train            # 作业名称
#SBATCH --output=lf_train_%j.log    # Slurm的标准输出和错误日志，%j表示作业ID

source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate t2

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/LLaMA-Factory/
PROJECT_NAME="critic_cot"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=f318ffd0dcf5d31701fd33aee12e57e9cf15444f
export WANDB_PROJECT=$PROJECT_NAME
# export WANDB_MODE=disabled
export WANDB_DISABLED="true"
export MASTER_PORT=$(shuf -i 30000-40000 -n 1)
export MASTER_ADDR="127.0.0.1"

# FORCE_TORCHRUN=1 llamafactory-cli train ../pjlab_scripts_0428/360_train_phi4_0503/phi4_qwen3_corr_0503.yaml > phi4_qwen3_corr_0503_output.txt 2>&1
llamafactory-cli train ../pjlab_scripts_0428/360_train_phi4_0503/phi4_qwen3_corr_0503.yaml > phi4_qwen3_corr_0503_output.txt 2>&1
