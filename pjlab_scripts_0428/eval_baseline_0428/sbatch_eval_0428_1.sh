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
conda activate cft

summary_path="../../baseline_eval_results_0428_1/qwen_32b_distill/summary.txt"
model_path="/mnt/hwfile/opendatalab/yubo/models/DeepSeek-R1-Distill-Qwen-32B"
output_path="../../baseline_eval_results_0428_1/qwen_32b_distill/"

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts

mkdir -p $output_path

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash evaluate_distilled_qwen.sh $model_path $output_path $summary_path  > ../../baseline_eval_results_0428_1/qwen_32b_distill/output.txt 2>&1





