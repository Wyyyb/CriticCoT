#!/bin/bash

# 激活conda环境
# conda activate cft

# 设置要处理的检查点数字列表
checkpoint_numbers=(2 4 6 8 10 12 14)

 # 使用所有4张GPU
export CUDA_VISIBLE_DEVICES=6,7

# 串行处理每个检查点，每次都使用全部4张卡
for ckpt_num in "${checkpoint_numbers[@]}"; do
    summary_path="../eval_results_0513_3/summary.txt"
    model_path="/data/yubo/CriticCoT/ms-swift/output_models_0513_one_shot_balance/v0-20250514-021302/checkpoint-${ckpt_num}"
    output_path="../eval_results_0513_3/balance_one-shot_exp_ckpt_${ckpt_num}/"

    echo "Processing checkpoint ${ckpt_num}"

    cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
    mkdir -p $output_path

    bash evaluate_qwen.sh $model_path $output_path $summary_path

    echo "Finished processing checkpoint ${ckpt_num}"
done

echo "All checkpoints processed successfully!"


#!/bin/bash

# 激活conda环境
# conda activate cft

# 设置要处理的检查点数字列表
checkpoint_numbers=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72)

 # 使用所有4张GPU
export CUDA_VISIBLE_DEVICES=6,7

# 串行处理每个检查点，每次都使用全部4张卡
for ckpt_num in "${checkpoint_numbers[@]}"; do
    summary_path="../eval_results_0513_2//summary.txt"
    model_path="/data/yubo/CriticCoT/ms-swift/output_models_0513_filtered_full_38k/v2-20250514-010049/checkpoint-${ckpt_num}"
    output_path="../eval_results_0513_2/one-shot_exp_ckpt_${ckpt_num}/"

    echo "Processing checkpoint ${ckpt_num}"

    cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
    mkdir -p $output_path

    bash evaluate_qwen.sh $model_path $output_path $summary_path

    echo "Finished processing checkpoint ${ckpt_num}"
done

echo "All checkpoints processed successfully!"