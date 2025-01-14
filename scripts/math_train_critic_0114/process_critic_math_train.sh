#!/bin/bash
set -ex

source /gpfs/public/research/miniconda3/bin/activate
conda activate lf_yubo

cd ../../Qwen2.5-Math-Eval/

input_dir_base="math_eval_result_0114_critic_MATH_train/qwen_eval_res_0114_critic/"
summary_path="math_eval_result_0114_critic_MATH_train/sta_critic_summary_0114.txt"

# 遍历所有qwen2.5开头的子文件夹
for input_dir in ${input_dir_base}/qwen2.5*/; do
    if [ -d "$input_dir" ]; then
        echo "Processing directory: $input_dir"
        python postprocess_critic_result.py \
            --input_dir "$input_dir" \
            --summary_path "$summary_path" \
            --candidate_num 8
    fi
done