#!/bin/bash
#SBATCH --partition=belt_road       # 分区名称
#SBATCH --nodes=1                   # 请求1个节点
#SBATCH --ntasks-per-node=1         # 每节点1个任务
#SBATCH --gres=gpu:8                # 请求2个GPU
#SBATCH --mem=64G                   # 请求32GB内存
#SBATCH --time=24:00:00             # 最长运行16小时
#SBATCH --job-name=qwen_gen         # 作业名称
#SBATCH --output=qwen_gen_%j.log    # Slurm的标准输出和错误日志，%j表示作业ID

# 初始化conda
source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh

# 激活cft环境
conda activate cft

# 让Slurm管理GPU分配
# 不要手动设置CUDA_VISIBLE_DEVICES，让Slurm自动设置

# 打印当前分配到的GPU设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 运行Python脚本并实时输出到文件（禁用输出缓冲）
python -u qwen_32b_distill_gen_deepmath_solution_0420.py > qwen_32b_distill_gen_deepmath_solution_0420_output.txt 2>&1




