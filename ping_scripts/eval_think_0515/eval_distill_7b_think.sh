#!/bin/bash

# 激活conda环境
# conda activate cft

# 设置要处理的检查点数字列表
checkpoint_numbers=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40)
# checkpoint_numbers=(2 4 6 8 10 12 14 16 18 20)
# checkpoint_numbers=(4 8 12 16 20 24 28 32 36 40)

 # 使用所有4张GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 串行处理每个检查点，每次都使用全部4张卡
for ckpt_num in "${checkpoint_numbers[@]}"; do
    summary_path="../eval_results_0514_qwen_2-5_math_7b_think_0/summary.txt"
    model_path="/data/yubo/CriticCoT/ms-swift/output_models_0514_qwen_2-5_math_7b_think/v0-20250515-035910/checkpoint-${ckpt_num}"
    output_path="../eval_results_0514_qwen_2-5_math_7b_think_0/ckpt_${ckpt_num}/"

    echo "Processing checkpoint ${ckpt_num}"

    cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
    mkdir -p $output_path

    bash evaluate_qwen.sh $model_path $output_path $summary_path

    echo "Finished processing checkpoint ${ckpt_num}"
done

echo "All checkpoints processed successfully!"