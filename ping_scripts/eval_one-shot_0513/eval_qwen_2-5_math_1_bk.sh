#!/bin/bash

# 激活conda环境
conda activate cft

# 设置要处理的检查点数字列表
checkpoint_numbers=(4 8 12 16 20 24 28 32)
gpu_count=8  # 可用GPU总数

for i in "${!checkpoint_numbers[@]}"; do
    ckpt_num="${checkpoint_numbers[$i]}"
    # 计算使用哪个GPU (0-7)
    gpu_id=$((i % gpu_count))
    
    # 为每个任务创建子shell并在后台运行
    (
        summary_path="../eval_results_0513_1/one-shot_exp_ckpt_${ckpt_num}/summary.txt"
        model_path="/data/yubo/CriticCoT/ms-swift/output_models_0513_rm_thinking/v0-20250513-200706/checkpoint-${ckpt_num}"
        output_path="../eval_results_0513_1/one-shot_exp_ckpt_${ckpt_num}/"

        echo "Processing checkpoint ${ckpt_num} on GPU ${gpu_id}..."

        cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
        mkdir -p $output_path

        export CUDA_VISIBLE_DEVICES=${gpu_id}

        bash evaluate_qwen.sh $model_path $output_path $summary_path
        
        echo "Finished processing checkpoint ${ckpt_num} on GPU ${gpu_id}"
    ) &
done
# 等待所有后台任务完成
wait

# 设置要处理的检查点数字列表
checkpoint_numbers=(36 40 44 48 52 56 60 64)
gpu_count=8  # 可用GPU总数

for i in "${!checkpoint_numbers[@]}"; do
    ckpt_num="${checkpoint_numbers[$i]}"
    # 计算使用哪个GPU (0-7)
    gpu_id=$((i % gpu_count))

    # 为每个任务创建子shell并在后台运行
    (
        summary_path="../eval_results_0513_1/one-shot_exp_ckpt_${ckpt_num}/summary.txt"
        model_path="/data/yubo/CriticCoT/ms-swift/output_models_0513_rm_thinking/v0-20250513-200706/checkpoint-${ckpt_num}"
        output_path="../eval_results_0513_1/one-shot_exp_ckpt_${ckpt_num}/"

        echo "Processing checkpoint ${ckpt_num} on GPU ${gpu_id}..."

        cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
        mkdir -p $output_path

        export CUDA_VISIBLE_DEVICES=${gpu_id}

        bash evaluate_qwen.sh $model_path $output_path $summary_path

        echo "Finished processing checkpoint ${ckpt_num} on GPU ${gpu_id}"
    ) &
done
# 等待所有后台任务完成
wait

echo "All checkpoints processed successfully!"