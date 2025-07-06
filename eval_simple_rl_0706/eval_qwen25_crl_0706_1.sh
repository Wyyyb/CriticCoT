#!/bin/bash

eval "$(/data/minimax-dialogue/feishan/miniconda3/bin/conda shell.bash hook)"

conda activate yb_verl

cd /data/minimax-dialogue/feishan/CriticCoT/simpleRL-reason

# 定义检查点编号数组
#checkpoint_numbers=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
checkpoint_numbers=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)


# GPU 设置
export CUDA_VISIBLE_DEVICES=0

# 遍历每个检查点进行评估
for ckpt_num in "${checkpoint_numbers[@]}"; do
    # 定义路径
    summary_path="/data/minimax-dialogue/feishan/CriticCoT/verl_data/eval_results/qwen25_math_7b_crl/summary.txt"
    model_path="/data/minimax-dialogue/feishan/CriticCoT/verl_data/checkpoints/simple_rl_qwen2-5_math_7b_crl/global_step_${ckpt_num}/actor/huggingface/"
    output_path="/data/minimax-dialogue/feishan/CriticCoT/verl_data/eval_results/qwen25_math_7b_crl/ckpt-${ckpt_num}/"

    # 检查模型是否存在
    if ls ${model_path}/*.safetensors 1> /dev/null 2>&1; then
        echo "Processing checkpoint ${ckpt_num}"

        # 切换到评估脚本目录
        cd /data/minimax-dialogue/feishan/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
        mkdir -p $output_path

        # 运行评估脚本
        bash evaluate_qwen.sh $model_path $output_path $summary_path

        echo "Finished processing checkpoint ${ckpt_num}"
    else
        echo "Checkpoint ${ckpt_num} not found at ${model_path}. Skipping."
    fi
done

echo "All checkpoints processed successfully!"