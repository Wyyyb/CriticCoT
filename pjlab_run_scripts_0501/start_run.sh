#!/bin/bash
#SBATCH --partition=belt_road       # 分区名称
#SBATCH --nodes=1                   # 请求1个节点
#SBATCH --ntasks-per-node=1         # 每节点1个任务
#SBATCH --gres=gpu:8                # 请求8个GPU
#SBATCH --mem=96G                   # 请求32GB内存
#SBATCH --time=120:00:00             # 最长运行8小时
#SBATCH --job-name=train_32b         # 作业名称
#SBATCH --output=train_32b_%j.log    # Slurm的标准输出和错误日志，%j表示作业ID


cd /mnt/hwfile/opendatalab/yubo/CriticCoT/pjlab_run_scripts_0501
# 设置环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 确保使用8张GPU卡

# 创建输出日志文件名，使用当前时间（月日时分格式）
LOG_TIMESTAMP=$(date +"%m%d%H%M")
LOG_FILE="${LOG_TIMESTAMP}_output.txt"
echo "Logging output to $LOG_FILE"

echo "Starting main loop, logging to $LOG_FILE"

# 主循环：每隔2分钟执行一次run_1.sh，如果run_1.sh运行结束得快，则执行GPU占用代码
echo "Starting main loop to run run_1.sh every 2 minutes..." | tee -a "$LOG_FILE"
while true; do
    start_time=$(date +%s)
    echo "$(date): Running run_1.sh" | tee -a "$LOG_FILE"

    # 运行run_1.sh并检查其退出状态
    if [ -f "run_1.sh" ] && [ -s "run_1.sh" ] && [ -x "run_1.sh" ]; then
        bash run_1.sh | tee -a "$LOG_FILE"
        run_status=$?
        echo "run_1.sh completed with status: $run_status" | tee -a "$LOG_FILE"
    else
        echo "run_1.sh doesn't exist, is empty, or not executable" | tee -a "$LOG_FILE"
        run_status=0
    fi

    # 检查是否需要运行GPU占用代码
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    # 如果run_1.sh执行时间小于10秒，运行GPU占用代码
    if [ $elapsed_time -lt 10 ]; then
        echo "run_1.sh completed quickly. Running GPU memory occupation code..." | tee -a "$LOG_FILE"

        # 计算剩余时间
        remaining_time=$((10 - elapsed_time))

        # 运行Python脚本来占用GPU，输出重定向到日志文件
        python run_on_gpu.py $remaining_time | tee -a "$LOG_FILE"

        echo "GPU occupation complete." | tee -a "$LOG_FILE"
    else
        echo "run_1.sh took more than 10 seconds, skipping GPU occupation." | tee -a "$LOG_FILE"
    fi

    # 计算等待时间，确保总循环时间为20s
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    wait_time=$((20 - elapsed_time))

    if [ $wait_time -gt 0 ]; then
        echo "Waiting for $wait_time seconds before next execution..." | tee -a "$LOG_FILE"
        sleep $wait_time
    else
        echo "Cycle took more than 20s, starting next cycle immediately." | tee -a "$LOG_FILE"
    fi
done