#!/bin/bash
#SBATCH --partition=belt_road       # 分区名称
#SBATCH --nodes=1                   # 请求1个节点
#SBATCH --ntasks-per-node=1         # 每节点1个任务
#SBATCH --gres=gpu:8                # 请求8个GPU
#SBATCH --mem=32G                   # 请求32GB内存
#SBATCH --time=08:00:00             # 最长运行8小时
#SBATCH --job-name=qwen_gen         # 作业名称
#SBATCH --output=qwen_gen_%j.log    # Slurm的标准输出和错误日志，%j表示作业ID

# 加载必要的模块（如果需要）
# module load anaconda/2023.03
# module load cuda/11.7

# 激活您的环境（如果有）
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 运行您的Python脚本，并将标准输出和标准错误重定向到指定文件
python -u qwen_32b_gen_deepmath_solution_0420.py > qwen_32b_gen_deepmath_solution_0420_output.txt 2>&1