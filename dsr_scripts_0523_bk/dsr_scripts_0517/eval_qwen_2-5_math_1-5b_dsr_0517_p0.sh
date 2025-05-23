#!/bin/bash

# 激活conda环境
# conda activate cft

# 设置要处理的检查点数字列表
# checkpoint_numbers=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40)
# checkpoint_numbers=(2 4 6 8 10 12 14 16 18 20)
# checkpoint_numbers=(4 8 12 16 20 24 28 32 36 40)
checkpoint_numbers=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96 100 104 108 112 116 120 124 128 132 136 140 144 148 152 156 160 164 168 172 176 180 184 188 192 196 200)
# checkpoint_numbers=(2 6 10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70 74 78 82 86 90 94 98 102 106 110 114 118 122 126 130 134 138 142 146 150 154 158 162 166 170 174 178 182 186 190 194 198)
 # 使用所有4张GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 串行处理每个检查点，每次都使用全部4张卡
for ckpt_num in "${checkpoint_numbers[@]}"; do
    summary_path="../eval_results_dsr_of_0517_p0/summary.txt"
    model_path="/data/yubo/CriticCoT/ms-swift/output_models_dsr_1.5b_base_of_0517_p0/v0-20250517-184201/checkpoint-${ckpt_num}"
    output_path="../eval_results_dsr_of_0517_p0/dsr_1-5b_p0_ckpt_${ckpt_num}/"

    echo "Processing checkpoint ${ckpt_num}"

    cd /data/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/scripts
    mkdir -p $output_path

    bash evaluate_qwen.sh $model_path $output_path $summary_path

    echo "Finished processing checkpoint ${ckpt_num}"
done

echo "All checkpoints processed successfully!"