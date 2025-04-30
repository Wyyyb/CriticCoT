#!/bin/bash
#SBATCH --partition=belt_road       # 分区名称
#SBATCH --nodes=1                   # 请求1个节点
#SBATCH --ntasks-per-node=1         # 每节点1个任务
#SBATCH --gres=gpu:8                # 请求8个GPU
#SBATCH --mem=32G                   # 请求32GB内存
#SBATCH --time=96:00:00             # 最长运行8小时
#SBATCH --job-name=qwen_gen         # 作业名称
#SBATCH --output=qwen_gen_%j.log    # Slurm的标准输出和错误日志，%j表示作业ID


# 激活您的环境（如果有）
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /data/yubowang/CriticCoT/process_data_0429_pjlab
# 运行您的Python脚本，并将标准输出和标准错误重定向到指定文件
python -u qwen3_32b_gen_critique_0429.py

