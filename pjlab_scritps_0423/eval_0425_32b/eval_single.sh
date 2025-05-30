#!/bin/bash
set -ex

source /mnt/petrelfs/wangyubo.p/miniconda3/etc/profile.d/conda.sh
conda activate cft

summary_path="/mnt/hwfile/opendatalab/yubo/CriticCoT/0425_eval_results_32b_cft_deepmath/summary.txt"
models_dir="/mnt/hwfile/opendatalab/yubo/CriticCoT/360-LLaMA-Factory-sp/output_models_0425/deepmath_qwen_32b_distill_cft_0424"
start=0
end=160

export CUDA_VISIBLE_DEVICES=0,1,6,7

cd /mnt/hwfile/opendatalab/yubo/CriticCoT/Qwen2.5-Math-Eval-0203/multi_eval_0425

# 获取models_dir的文件夹名称（不是完整路径）
models_dir_name=$(basename "$models_dir")

# 获取summary_path的上一层目录
summary_parent_dir=$(dirname "$summary_path")

for checkpoint_dir in ${models_dir}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then  # 确保是目录
        # 获取checkpoint号码
        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)
        echo ${checkpoint_num}

        # 检查checkpoint_num是否[0, 30)
        if [ "$checkpoint_num" -ge ${start} ] && [ "$checkpoint_num" -lt ${end} ]; then
            # 设置输出目录
            output_dir="${summary_parent_dir}/${models_dir_name}-checkpoint-${checkpoint_num}/"

            echo "Processing checkpoint-${checkpoint_num}"
            echo "Model path: ${checkpoint_dir}"
            echo "Output dir: ${output_dir}"

            # 执行评估脚本
            bash eval_full.sh "$checkpoint_dir" "$output_dir" "$summary_path"
        else
            echo "Skipping checkpoint-${checkpoint_num}"
        fi
    fi
done
