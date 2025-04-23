#!/bin/bash
#SBATCH --partition=belt_road       # 分区名称
#SBATCH --nodes=1                   # 请求1个节点
#SBATCH --ntasks-per-node=1         # 每节点1个任务
#SBATCH --gres=gpu:8                # 请求8个GPU
#SBATCH --mem=32G                   # 请求32GB内存
#SBATCH --time=24:00:00             # 最长运行8小时
#SBATCH --job-name=qwen_gen         # 作业名称
#SBATCH --output=qwen_gen_%j.log    # Slurm的标准输出和错误日志，%j表示作业ID

source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="../../eval_results_0423/qwen_32b_distill_aime25/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/DeepSeek-R1-Distill-Qwen-32B"
output_path="../../eval_results_0423/qwen_32b_distill_aime25/"

mkdir -p $output_path

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#bash evaluate_qwen_test.sh $model_path $output_path $summary_path > /mnt/hwfile/opendatalab/yubo/CriticCoT/pjlab_scritps_0423/eval_aime/qwen_32b_distil_output.txt 2>&1

bash evaluate_qwen_test.sh $model_path $output_path $summary_path
